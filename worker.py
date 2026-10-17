"""Воркер: берёт job_id из очереди, запускает анализ, сохраняет результат, рисует аннотированное видео.

Режимы:
  python3 worker.py      — отдельный процесс (docker compose, очередь Redis: REDIS_URL);
  app.py сам запускает потоки-воркеры, если очередь в памяти (RunPod / ноутбук, как раньше).
Аннотированное видео рисуется в отдельном потоке ПОСЛЕ ответа: оно не съедает лимит 5 минут
и не задерживает следующие видео в очереди.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
import traceback
from typing import Any

import settings

log = logging.getLogger("worker")
VIS_ENABLED = settings.env_bool("VIS_ENABLED", bool((settings.rules_config().get("visualization") or {}).get("enabled", True)))
# видео с разметкой для задач жюри не рисуем: API его не отдаёт, а копия тестового видео на диске не нужна
# и перекодирование отнимает CPU у следующего видео жюри (лимит 5 минут)
VIS_FOR_API = settings.env_bool("VIS_FOR_API", False)
KEEP_VIDEO = settings.env_bool("KEEP_VIDEO", False)        # 0 — исходное видео удаляется сразу после анализа/разметки


def drop_upload(path: str) -> None:
    if KEEP_VIDEO or not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def user_error(e: Exception) -> str:
    """Пользователю — понятная причина, без внутренних подробностей (трассировка — в журнале)."""
    msg = str(e)
    if isinstance(e, (ValueError, RuntimeError)) and re.search("[а-яА-Я]", msg):
        return msg[:300]
    return "Внутренняя ошибка анализа. Повторите загрузку; если ошибка повторится — сообщите команде."

_render_q: "queue.Queue[tuple[Any, str, str, dict]]" = queue.Queue()
_render_started = False
_render_lock = threading.Lock()


def annotated_url(jid: str) -> str:
    return f"/results/{jid}_annotated.mp4"


def _render_loop() -> None:
    import visualization
    while True:
        store, jid, path, vis = _render_q.get()
        out = os.path.join(settings.RESULTS_DIR, f"{jid}_annotated.mp4")
        t = time.time()
        try:
            job = store.get(jid) or {}
            visualization.render(path, out, vis.get("scene"), vis.get("zones"), job.get("violations", []),
                                 job.get("rule_events", []), settings.rules_config().get("visualization") or {})
            store.patch_result(jid, annotated_video_url=annotated_url(jid), annotated_status="ready")
            log.info("видео с разметкой %s готово за %.0f с", jid, time.time() - t)
        except Exception as e:  # noqa: BLE001 — видео с разметкой необязательно
            log.exception("видео с разметкой %s не получилось", jid)
            store.patch_result(jid, annotated_status="failed", annotated_error=str(e)[:300])
        finally:
            drop_upload(path)


def _ensure_renderer() -> None:
    global _render_started
    with _render_lock:
        if not _render_started:
            threading.Thread(target=_render_loop, daemon=True, name="render").start()
            _render_started = True


def process(store, jid: str) -> None:
    import pipeline
    job = store.get(jid)
    if not job or not job.get("_path"):
        log.error("задача %s без файла — пропуск", jid)
        return
    path, t0 = job["_path"], job.get("_t0") or time.time()
    src, name = job.get("_src", "api"), job.get("name", "")
    vis_on = VIS_ENABLED and (src == "web" or VIS_FOR_API)
    last = [0.0]

    def progress(p: int, msg: str) -> None:
        now = time.time()
        if now - last[0] >= 0.5 or p >= 90:              # Redis не дёргаем чаще 2 раз в секунду
            last[0] = now
            store.update(jid, progress=p, stage=msg, _worker_ts=now)

    store.update(jid, stage="Анализ начат", _worker_ts=time.time())
    alive = threading.Event()

    def heartbeat() -> None:                           # «задача жива» — иначе после сбоя её подберёт resume()
        while not alive.wait(10):
            try:
                store.update(jid, _worker_ts=time.time())
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    vis = None
    try:
        res = pipeline.analyze(path, progress, t0=t0)
        vis = res.pop("_vis", None)
        out = {"job_id": jid, **res, "status": "done", "progress": 100, "stage": "Готово", "name": name, "_src": src,
               "annotated_video_url": None, "annotated_status": "rendering" if vis_on else "off"}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        out = {"job_id": jid, "status": "failed", "error": user_error(e), "stage": "Ошибка", "progress": 100,
               "name": name, "_src": src}
    alive.set()
    try:
        store.save_result(out)
    except Exception:  # noqa: BLE001 — воркер не должен умирать из-за диска
        log.exception("не удалось сохранить результат %s", jid)
        return
    if vis_on and out["status"] == "done":
        _ensure_renderer()
        _render_q.put((store, jid, path, vis or {}))          # видео удалится после разметки
    else:
        drop_upload(path)


def run_forever(store, stop: threading.Event | None = None) -> None:
    while stop is None or not stop.is_set():
        try:
            jid = store.dequeue(timeout=5)
        except Exception as e:  # noqa: BLE001 — Redis перезапускается: ждём
            log.warning("очередь недоступна: %r", e)
            time.sleep(3)
            continue
        if jid:
            process(store, jid)


def main() -> None:
    import store as store_mod
    settings.setup_logging()
    st = store_mod.make(settings.DATA_DIR)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    n = int(os.getenv("WORKERS", 1))
    st.resume()
    log.info("воркер запущен: %d поток(а), очередь %s", n, st.kind)
    threads = [threading.Thread(target=run_forever, args=(st,), daemon=True) for _ in range(n)]
    for t in threads:
        t.start()
    while True:                                        # задачи упавшего воркера (heartbeat устарел) — снова в очередь
        time.sleep(60)
        try:
            st.resume()
        except Exception as e:  # noqa: BLE001
            log.warning("resume: %r", e)


if __name__ == "__main__":
    main()
