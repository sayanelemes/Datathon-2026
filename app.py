"""Сервис (FastAPI): API по ТЗ (X-API-Key) + веб-интерфейс.

Безопасность и данные:
- задачи жюри (API) и сайта (web) разделены: по адресам сайта нельзя прочитать результат жюри и наоборот;
- job_id — 96 бит случайности (не угадать), проверяется по шаблону (нет обхода путей);
- очередь с приоритетом: видео жюри всегда раньше загрузок с сайта; у сайта лимиты на число загрузок;
- файл пишется потоком с лимитом размера и проверяется как видео до постановки в очередь;
- исходное видео удаляется после анализа (KEEP_VIDEO=0), задачи старше RETENTION_HOURS удаляются;
- файлы создаются только для владельца (umask 077); ключ ≥16 символов, сравнение за постоянное время;
- заголовки безопасности (CSP, nosniff, запрет встраивания), результаты не кэшируются.

Очередь в памяти (по умолчанию, RunPod/ноутбук): видео анализируют потоки-воркеры этого же процесса.
Очередь Redis (REDIS_URL, docker compose): API только ставит задачи, анализ — в контейнере worker.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

import settings
import store as store_mod
from schemas import AnalyzeAccepted, JobExtended, JobPending, JobStrict

settings.setup_logging()
log = logging.getLogger("app")

API_KEY = os.getenv("API_KEY", "")
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", 200))
WORKERS = int(os.getenv("WORKERS", 1))   # сколько видео анализируются одновременно (vLLM сам объединяет запросы в батчи)
API_EXTENDED = settings.env_bool("API_EXTENDED", False)   # 1 — confidence, legal_reference, annotated_video_url в ответе API
ALLOWED_EXT = (".mov", ".mp4", ".m4v", ".avi", ".mkv")
STATIC = os.path.join(settings.BASE_DIR, "static")
JID = re.compile(r"[0-9a-f]{24}")
MAX_DUR = float(os.getenv("MAX_DUR", 900))                       # видео длиннее 15 минут не принимаем
VALIDATE_VIDEO = settings.env_bool("VALIDATE_VIDEO", True)       # проверить, что файл читается как видео
RETENTION_HOURS = float(os.getenv("RETENTION_HOURS", 72))        # задачи старше — удаляются целиком (0 — не удалять)
WEB_MAX_PENDING = int(os.getenv("WEB_MAX_PENDING", 3))           # загрузок с сайта в очереди одновременно
WEB_PER_IP_HOUR = int(os.getenv("WEB_PER_IP_HOUR", 10))          # загрузок с сайта с одного адреса в час
_WEB_HITS: dict[str, list[float]] = {}
_WEB_LOCK = threading.Lock()

if os.getenv("VLM_MOCK") != "1" and (len(API_KEY) < 16 or API_KEY.startswith("change-me")):
    sys.exit("API_KEY не задан или слишком простой (нужно ≥16 символов). "
             "Запускайте через start.sh — он берёт ключ из /workspace/2mp_api_key, или задайте его в .env.")

os.umask(0o077)
for d in (settings.DATA_DIR, settings.UPLOAD_DIR, settings.RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass

STORE = store_mod.make(settings.DATA_DIR)
INLINE_WORKERS = settings.env_bool("INLINE_WORKERS", STORE.kind == "memory")

app = FastAPI(title="2MP — контроль кассовой дисциплины", version="2.0",
              description="POST /api/v1/analyze (multipart, поле file, заголовок X-API-Key) → job_id; "
                          "GET /api/v1/jobs/{job_id} → статус и нарушения.")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def public(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def check_key(request: Request) -> bool:
    return bool(API_KEY) and hmac.compare_digest(request.headers.get("x-api-key", "").encode(), API_KEY.encode())


def reviews_path(jid: str) -> str:
    return os.path.join(settings.DATA_DIR, jid, "reviews.json")


def load_job(jid: str, src: str) -> dict[str, Any] | None:
    """Задача только своего источника: сайт не видит задачи жюри, API — задачи сайта."""
    if not JID.fullmatch(jid or ""):
        return None
    job = STORE.get(jid)
    if job and job.get("_src", "api") != src:
        return None
    if job and os.path.exists(reviews_path(jid)):
        try:
            with open(reviews_path(jid), encoding="utf-8") as f:
                job["reviews"] = json.load(f)
        except (OSError, ValueError):
            pass
    return job


async def create_job(request: Request, src: str = "api") -> JSONResponse:
    try:
        if int(request.headers.get("content-length") or 0) > (MAX_MB + 1) * 1024 * 1024:
            return JSONResponse({"error": f"файл больше {MAX_MB} МБ"}, 413)
    except ValueError:
        pass
    try:
        form = await request.form()
    except Exception as e:  # noqa: BLE001 — битый multipart
        return JSONResponse({"error": f"ожидается multipart/form-data с полем file ({e})"}, 400)
    f = form.get("file")
    if f is None or not hasattr(f, "read"):
        return JSONResponse({"error": "поле file обязательно"}, 400)
    ext = os.path.splitext(os.path.basename(f.filename or ""))[1].lower() or ".mov"
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": f"неподдерживаемый формат {ext}; ожидается .mov"}, 415)
    jid = secrets.token_hex(12)
    path = os.path.join(settings.UPLOAD_DIR, jid + ext)       # uploads/{job_id}.mov
    size = 0
    with open(path, "wb") as out:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                out.close()
                os.remove(path)
                return JSONResponse({"error": "файл слишком большой"}, 413)
            out.write(chunk)
    if size == 0:
        os.remove(path)
        return JSONResponse({"error": "пустой файл"}, 400)
    if VALIDATE_VIDEO:                                         # это вообще видео? — до очереди, а не через минуту
        import cv2
        cap = cv2.VideoCapture(path)
        fps, n, ok = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.read()[0]
        cap.release()
        dur = n / fps if fps else 0
        if not ok or dur <= 0 or dur > MAX_DUR:
            os.remove(path)
            msg = "файл не читается как видео (нужен .mov или .mp4)" if not ok or dur <= 0 else f"видео длиннее {MAX_DUR / 60:.0f} минут"
            return JSONResponse({"error": msg}, 400)
    name = re.sub(r"[^\w.\- ()]+", "_", os.path.basename(f.filename or "запись"))[:120]
    q = STORE.qsize()
    STORE.create(jid, path, time.time(), f"В очереди ({q} перед вами)" if q else "Начинаю анализ", src=src, name=name)
    log.info("задача %s (%s): %.1f МБ", jid, src, size / 1e6)
    return JSONResponse(AnalyzeAccepted(job_id=jid).model_dump())


def jury_view(job: dict[str, Any]) -> dict[str, Any]:
    """Ответ API: строго поля ТЗ (по умолчанию) или расширенный формат (API_EXTENDED=1)."""
    st = job.get("status")
    if st != "done":                       # processing — только статус; failed — статус и причина
        out = JobPending(job_id=job["job_id"], status=st if st in ("processing", "failed") else "processing",
                         error=job.get("error", "") if st == "failed" else None)
        return out.model_dump(exclude_none=True)
    vs = []
    for v in job.get("violations", []):
        frames = [{"timestamp_sec": f["timestamp_sec"], "image_base64": f["image_base64"]}
                  for f in v.get("evidence_frames", [])][:3]
        base = {"type": v["type"], "timecode_sec": v["timecode_sec"], "description": v["description"], "evidence_frames": frames}
        if API_EXTENDED:
            base.update(confidence=v.get("confidence", 0.0), legal_reference=v.get("legal_reference"),
                        legal_confidence=v.get("legal_confidence", "low"), object_type=v.get("object_type", "unknown"),
                        verification=v.get("verification", "confirmed"))
        vs.append(base)
    verdict = "violation" if vs else "no_violation"
    try:
        if API_EXTENDED:
            return JobExtended(job_id=job["job_id"], verdict=verdict, violations=vs,
                               annotated_video_url=job.get("annotated_video_url")).model_dump()
        return JobStrict(job_id=job["job_id"], verdict=verdict, violations=vs).model_dump()
    except ValidationError:
        log.exception("ответ задачи %s не прошёл схему — отдаю как есть", job["job_id"])
        return {"job_id": job["job_id"], "status": "done", "verdict": verdict, "violations": vs}


# ---- API для жюри ----
@app.post("/api/v1/analyze", summary="Загрузить видео (.mov) на анализ")
async def api_analyze(request: Request):
    if not check_key(request):
        return JSONResponse({"error": "invalid API key"}, 401)
    return await create_job(request)


@app.get("/api/v1/jobs/{job_id}", summary="Статус и результат анализа")
async def api_job(job_id: str, request: Request):
    if not check_key(request):
        return JSONResponse({"error": "invalid API key"}, 401)
    job = load_job(job_id, "api")
    if not job:
        return JSONResponse({"error": "job not found"}, 404)
    return JSONResponse(jury_view(job))


# ---- для веб-интерфейса (без ключа) ----
def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


@app.post("/web/analyze", include_in_schema=False)
async def web_analyze(request: Request):
    """Сайт без входа (ТЗ, FAQ 27) — поэтому лимиты: видео жюри не должны ждать чужих загрузок."""
    ip, now = client_ip(request), time.time()
    with _WEB_LOCK:
        hits = [t for t in _WEB_HITS.get(ip, []) if now - t < 3600]
        if len(hits) >= WEB_PER_IP_HOUR:
            return JSONResponse({"error": "Слишком много загрузок с вашего адреса за час. Попробуйте позже."}, 429)
        if STORE.pending("web") >= WEB_MAX_PENDING:
            return JSONResponse({"error": "Сервис сейчас занят проверкой других записей. Попробуйте через пару минут."}, 429)
        _WEB_HITS[ip] = hits + [now]
    return await create_job(request, "web")


@app.get("/web/jobs/{job_id}", include_in_schema=False)
async def web_job(job_id: str):
    job = load_job(job_id, "web")
    return JSONResponse(public(job)) if job else JSONResponse({"error": "not found"}, 404)


@app.post("/web/jobs/{job_id}/review", include_in_schema=False)
async def web_review(job_id: str, request: Request):
    job = load_job(job_id, "web")
    if not job or job.get("status") != "done":
        return JSONResponse({"error": "not found"}, 404)
    try:
        b = await request.json()
        i = int(b.get("i"))
        decision = {"confirm": "confirmed", "reject": "rejected"}.get(b.get("decision"), b.get("decision"))
        assert 0 <= i < len(job.get("violations", [])) and decision in ("confirmed", "rejected")
    except Exception:  # noqa: BLE001
        return JSONResponse({"error": "неверные данные решения"}, 400)
    rv = dict(job.get("reviews") or {})
    rv[str(i)] = {"decision": decision, "note": str(b.get("note", ""))[:500]}
    os.makedirs(os.path.dirname(reviews_path(job_id)), exist_ok=True)
    tmp = reviews_path(job_id) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rv, f, ensure_ascii=False)
    os.replace(tmp, reviews_path(job_id))
    return JSONResponse({"ok": True})


@app.get("/results/{name}", include_in_schema=False)
async def results(name: str):
    m = re.fullmatch(r"([0-9a-f]{24})_annotated\.mp4", name)
    if not m or not load_job(m.group(1), "web"):          # видео с разметкой — только для загрузок с сайта
        return JSONResponse({"error": "not found"}, 404)
    p = os.path.join(settings.RESULTS_DIR, name)
    if not os.path.exists(p):
        return JSONResponse({"error": "видео ещё готовится или не создавалось"}, 404)
    return FileResponse(p, media_type="video/mp4")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
async def health():
    import pipeline
    pose = pipeline.pose
    return {"ok": STORE.ping(), "queue": STORE.qsize(), "queue_backend": STORE.kind, "workers": WORKERS if INLINE_WORKERS else 0,
            "model": pipeline.VLM_MODEL, "profile": os.getenv("PROFILE", "-"), "mock": pipeline.VLM_MOCK,
            "pose": getattr(pose, "PROVIDER", "off") if pose else "off", "offline": os.getenv("HF_HUB_OFFLINE") == "1",
            "api_extended": API_EXTENDED}


CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
       "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


class Secure(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        r = await call_next(request)
        r.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                          "Content-Security-Policy": CSP, "Permissions-Policy": "camera=(), microphone=(), geolocation=()"})
        if request.url.path.startswith(("/api/", "/web/", "/results/")):
            r.headers["Cache-Control"] = "no-store"
        return r


app.add_middleware(Secure)


def _purge_loop() -> None:
    while RETENTION_HOURS > 0:
        try:
            n = STORE.purge(RETENTION_HOURS, settings.UPLOAD_DIR, settings.RESULTS_DIR)
            if n:
                log.info("удалено старых задач: %d", n)
        except Exception as e:  # noqa: BLE001
            log.warning("purge: %r", e)
        time.sleep(600)


threading.Thread(target=_purge_loop, daemon=True, name="purge").start()


def _start_inline_workers() -> None:
    import worker
    STORE.resume()
    for _ in range(WORKERS):
        threading.Thread(target=worker.run_forever, args=(STORE,), daemon=True).start()
    log.info("воркеров в процессе: %d", WORKERS)


if INLINE_WORKERS:
    _start_inline_workers()
else:
    log.info("API без воркеров: анализ выполняет отдельный процесс worker.py (очередь %s)", STORE.kind)
