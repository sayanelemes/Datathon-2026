"""Хранилище задач и очередь.

memory — как раньше: очередь и статусы в памяти процесса, результат — DATA_DIR/{job_id}/result.json
         (RunPod / ноутбук: один процесс uvicorn с воркерами-потоками).
redis  — очередь (список) и статусы в Redis, результат — тот же result.json на общем томе
         (docker compose: отдельные контейнеры api и worker). Включается переменной REDIS_URL.

В обоих режимах при старте незаконченные задачи снова ставятся в очередь (не зависают в processing).
"""
from __future__ import annotations

import json
import logging
import os
import itertools
import queue
import shutil
import threading
import time
from typing import Any, Optional

log = logging.getLogger("store")


def _job_dir(data: str, jid: str) -> str:
    return os.path.join(data, jid)


def _write_json(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)                         # атомарно: API никогда не читает полузаписанный файл


def _read_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class BaseStore:
    kind = "base"

    def __init__(self, data_dir: str):
        self.data = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._lock = threading.Lock()

    # --- файлы ---
    def meta_path(self, jid: str) -> str:
        return os.path.join(_job_dir(self.data, jid), "job.json")

    def result_path(self, jid: str) -> str:
        return os.path.join(_job_dir(self.data, jid), "result.json")

    def create(self, jid: str, path: str, t0: float, stage: str, src: str = "api", name: str = "") -> dict[str, Any]:
        """src: api — видео жюри (приоритет в очереди), web — загрузка с сайта."""
        os.makedirs(_job_dir(self.data, jid), exist_ok=True)
        job = {"job_id": jid, "status": "processing", "progress": 0, "stage": stage, "name": name,
               "_path": path, "_t0": t0, "_src": src}
        _write_json(self.meta_path(jid), {"job_id": jid, "_path": path, "_t0": t0, "_src": src, "name": name})
        self._put(job)
        self.enqueue(jid, src)
        return job

    def save_result(self, job: dict[str, Any]) -> None:
        with self._lock:
            _write_json(self.result_path(job["job_id"]), job)
        self._put({k: job[k] for k in ("job_id", "status", "progress", "stage", "error") if k in job}, final=True)

    def patch_result(self, jid: str, **fields: Any) -> Optional[dict[str, Any]]:
        """Дописать поля в готовый результат (видео с разметкой, решения проверяющего)."""
        with self._lock:
            job = _read_json(self.result_path(jid))
            if job is None:
                return None
            job.update(fields)
            _write_json(self.result_path(jid), job)
        return job

    def get(self, jid: str) -> Optional[dict[str, Any]]:
        if not jid or "/" in jid or ".." in jid:
            return None
        st = self._get_status(jid)
        if st and st.get("status") == "processing":
            return st
        res = _read_json(self.result_path(jid))
        if res is not None:
            return res
        return st

    def pending_on_disk(self) -> list[dict[str, Any]]:
        """Задачи без result.json (прерваны перезапуском)."""
        out = []
        for jid in sorted(os.listdir(self.data)):
            d = _job_dir(self.data, jid)
            if not os.path.isdir(d) or jid.startswith("_") or os.path.exists(self.result_path(jid)):
                continue
            meta = _read_json(self.meta_path(jid))
            if meta is None:                                  # старый формат: jobs/{id}/input.mov
                inp = [f for f in os.listdir(d) if f.startswith("input")]
                if not inp:
                    continue
                p = os.path.join(d, inp[0])
                meta = {"job_id": jid, "_path": p, "_t0": os.path.getmtime(p), "_src": "api"}
            if os.path.exists(meta["_path"]):
                out.append(meta)
        return out

    # --- реализуют наследники ---
    def _put(self, job: dict[str, Any], final: bool = False) -> None: ...
    def _get_status(self, jid: str) -> Optional[dict[str, Any]]: ...
    def update(self, jid: str, **fields: Any) -> None: ...
    def enqueue(self, jid: str, src: str = "api") -> None: ...
    def dequeue(self, timeout: float = 5.0) -> Optional[str]: ...
    def qsize(self) -> int: ...
    def ping(self) -> bool:
        return True

    def pending(self, src: str) -> int:
        return 0

    def purge(self, hours: float, upload_dir: str, results_dir: str) -> int:
        """Удаляет задачи старше hours: папку задачи, исходное видео, видео с разметкой."""
        if hours <= 0:
            return 0
        cutoff, n = time.time() - hours * 3600, 0
        for jid in os.listdir(self.data):
            d = _job_dir(self.data, jid)
            try:
                if jid.startswith("_") or not os.path.exists(self.result_path(jid)) or os.path.getmtime(self.result_path(jid)) > cutoff:
                    continue
                shutil.rmtree(d, ignore_errors=True)
                for folder in (upload_dir, results_dir):
                    for f in os.listdir(folder):
                        if f.startswith(jid):
                            os.remove(os.path.join(folder, f))
                n += 1
            except OSError:
                pass
        return n

    def resume(self) -> int:
        n = 0
        for meta in self.pending_on_disk():
            if self._claim_resume(meta["job_id"]):
                src = meta.get("_src", "api")
                self._put({"job_id": meta["job_id"], "status": "processing", "progress": 0, "name": meta.get("name", ""),
                           "stage": "В очереди (после перезапуска)", "_path": meta["_path"], "_t0": meta["_t0"], "_src": src})
                self.enqueue(meta["job_id"], src)
                log.info("возобновлена задача %s", meta["job_id"])
                n += 1
        return n

    def _claim_resume(self, jid: str) -> bool:
        return True


class MemoryStore(BaseStore):
    kind = "memory"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.q: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()   # 0 — жюри, 1 — сайт
        self._seq = itertools.count()

    def _put(self, job, final=False):
        with self._lock:
            if final:
                self.jobs.pop(job["job_id"], None)           # результат читается с диска
            else:
                self.jobs[job["job_id"]] = dict(job)

    def _get_status(self, jid):
        with self._lock:
            j = self.jobs.get(jid)
            return dict(j) if j else None

    def update(self, jid, **fields):
        with self._lock:
            if jid in self.jobs:
                self.jobs[jid].update(fields)

    def enqueue(self, jid, src="api"):
        self.q.put((0 if src == "api" else 1, next(self._seq), jid))

    def dequeue(self, timeout=5.0):
        try:
            return self.q.get(timeout=timeout)[2]
        except queue.Empty:
            return None

    def pending(self, src):
        with self._lock:
            return sum(1 for j in self.jobs.values() if j.get("_src") == src and j.get("status") == "processing")

    def qsize(self):
        return self.q.qsize()


class RedisStore(BaseStore):
    kind = "redis"

    def __init__(self, data_dir: str, url: str, prefix: str = "cg"):
        super().__init__(data_dir)
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=10, socket_connect_timeout=5)
        self.p = prefix
        self.qkey = f"{prefix}:queue"            # видео жюри (API)
        self.qkey_web = f"{prefix}:queue:web"    # загрузки с сайта — после жюри

    def _k(self, jid: str) -> str:
        return f"{self.p}:job:{jid}"

    def _put(self, job, final=False):
        self.r.set(self._k(job["job_id"]), json.dumps(job, ensure_ascii=False), ex=7 * 86400)

    def _get_status(self, jid):
        raw = self.r.get(self._k(jid))
        return json.loads(raw) if raw else None

    def update(self, jid, **fields):
        st = self._get_status(jid) or {"job_id": jid}
        st.update(fields)
        self._put(st)

    def enqueue(self, jid, src="api"):
        self.r.lpush(self.qkey if src == "api" else self.qkey_web, jid)

    def dequeue(self, timeout=5.0):
        item = self.r.brpop([self.qkey, self.qkey_web], timeout=max(int(timeout), 1))   # BRPOP берёт ключи по порядку
        return item[1] if item else None

    def qsize(self):
        return int(self.r.llen(self.qkey)) + int(self.r.llen(self.qkey_web))

    def pending(self, src):
        return int(self.r.llen(self.qkey if src == "api" else self.qkey_web))

    def ping(self):
        try:
            return bool(self.r.ping())
        except Exception:  # noqa: BLE001
            return False

    def _claim_resume(self, jid):
        st = self._get_status(jid)
        if st and st.get("status") == "processing" and st.get("_worker_ts", 0) > time.time() - 60:
            return False                                   # задачу прямо сейчас обрабатывает другой воркер
        if jid in self.r.lrange(self.qkey, 0, -1) or jid in self.r.lrange(self.qkey_web, 0, -1):
            return False
        return bool(self.r.set(f"{self.p}:resume:{jid}", "1", nx=True, ex=90))


def make(data_dir: str) -> BaseStore:
    url = os.getenv("REDIS_URL")
    if url:
        s = RedisStore(data_dir, url, os.getenv("REDIS_PREFIX", "cg"))
        log.info("очередь: Redis %s", url.split("@")[-1])
        return s
    log.info("очередь: в памяти процесса")
    return MemoryStore(data_dir)
