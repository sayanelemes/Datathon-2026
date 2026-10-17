"""API: ключ, загрузка, статусы, строгая схема ТЗ и расширенный формат."""
import importlib
import io
import time

import pytest
from fastapi.testclient import TestClient

KEY = {"X-API-Key": "test-key-0123456789"}


def fake_result(path, progress=lambda p, m: None, t0=None):
    progress(50, "тест")
    return {"verdict": "violation", "duration": 10.0, "zone": None, "tracks": [], "rule_events": [],
            "violations": [{"type": "Мобильный телефон на рабочем месте кассира", "timecode_sec": 4.2,
                            "description": "Кассир держит телефон. Норма: п. 12.",
                            "evidence_frames": [{"timestamp_sec": 4.0, "image_base64": "AAAA"},
                                                {"timestamp_sec": 4.2, "image_base64": "BBBB"}],
                            "confidence": 0.81, "legal_reference": {"source": "adilet.zan.kz", "document": "Правила",
                                                                    "article": "п. 12", "title": "", "quote": "текст",
                                                                    "url": None, "relevance": "", "confidence": "high"},
                            "legal_confidence": "high", "object_type": "phone", "verification": "confirmed",
                            "rule": {"id": "PHONE_AT_CASH_DESK"}, "bbox": None}],
            "_vis": {}}


@pytest.fixture()
def client(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline, "analyze", fake_result)
    import app as app_mod
    with TestClient(app_mod.app) as c:
        yield c, app_mod


def wait_done(c, jid, headers=KEY, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        r = c.get(f"/api/v1/jobs/{jid}", headers=headers).json()
        if r["status"] != "processing":
            return r
        time.sleep(0.1)
    raise AssertionError("задача не завершилась")


def upload(c, name="v.mov", data=b"\x00" * 1000, headers=KEY):
    return c.post("/api/v1/analyze", headers=headers, files={"file": (name, io.BytesIO(data), "video/quicktime")})


def test_auth(client):
    c, _ = client
    assert upload(c, headers={}).status_code == 401
    assert upload(c, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/v1/jobs/abc", headers={"X-API-Key": "wrong"}).status_code == 401


def test_analyze_and_strict_schema(client):
    c, app_mod = client
    r = upload(c)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processing" and len(body["job_id"]) == 24          # 96 бит — не угадать
    res = wait_done(c, body["job_id"])
    assert set(res) == {"job_id", "status", "verdict", "violations"}
    assert res["status"] == "done" and res["verdict"] == "violation"
    v = res["violations"][0]
    assert set(v) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert 1 <= len(v["evidence_frames"]) <= 3
    assert set(v["evidence_frames"][0]) == {"timestamp_sec", "image_base64"}
    import os
    import settings
    assert not os.path.exists(os.path.join(settings.UPLOAD_DIR, body["job_id"] + ".mov"))   # видео удалено после анализа


def test_extended_schema(client, monkeypatch):
    c, app_mod = client
    jid = upload(c).json()["job_id"]
    wait_done(c, jid)
    monkeypatch.setattr(app_mod, "API_EXTENDED", True)
    res = c.get(f"/api/v1/jobs/{jid}", headers=KEY).json()
    v = res["violations"][0]
    assert v["confidence"] == 0.81 and v["legal_reference"]["article"] == "п. 12" and v["legal_confidence"] == "high"
    assert "annotated_video_url" in res


def test_failed_job(client, monkeypatch):
    c, _ = client
    import pipeline

    def boom(*a, **k):
        raise RuntimeError("Модель анализа недоступна")
    monkeypatch.setattr(pipeline, "analyze", boom)
    jid = upload(c).json()["job_id"]
    res = wait_done(c, jid)
    assert res == {"job_id": jid, "status": "failed", "error": "Модель анализа недоступна"}


def test_bad_requests(client):
    c, _ = client
    assert c.post("/api/v1/analyze", headers=KEY).status_code == 400
    assert upload(c, name="x.txt").status_code == 415
    assert upload(c, data=b"").status_code == 400
    assert c.get("/api/v1/jobs/deadbeef", headers=KEY).status_code == 404
    assert c.get("/api/v1/jobs/" + "a" * 24, headers=KEY).status_code == 404
    assert c.get("/api/v1/jobs/../../etc", headers=KEY).status_code in (404, 400)
    assert c.get("/results/evil.mp4").status_code == 404


def test_web_endpoints(client):
    c, _ = client
    jid = c.post("/web/analyze", files={"file": ("v.mov", io.BytesIO(b"1" * 10), "video/quicktime")}).json()["job_id"]
    wait_done(c, jid)
    assert c.post(f"/web/jobs/{jid}/review", json={"i": 0, "decision": "confirm", "note": "ок"}).json() == {"ok": True}
    j = c.get(f"/web/jobs/{jid}").json()
    assert j["reviews"]["0"]["decision"] == "confirm" and not any(k.startswith("_") for k in j)
    h = c.get("/health").json()
    assert h["queue_backend"] == "memory" and h["api_extended"] is False


def test_sources_are_separated(client):
    """Сайт без ключа не видит результаты жюри; API не отдаёт задачи сайта; размеченное видео — только сайта."""
    c, _ = client
    jid = upload(c).json()["job_id"]
    wait_done(c, jid)
    assert c.get(f"/web/jobs/{jid}").status_code == 404
    assert c.post(f"/web/jobs/{jid}/review", json={"i": 0, "decision": "confirmed"}).status_code == 404
    assert c.get(f"/results/{jid}_annotated.mp4").status_code == 404
    wid = c.post("/web/analyze", files={"file": ("v.mov", io.BytesIO(b"1" * 10), "video/quicktime")}).json()["job_id"]
    assert c.get(f"/api/v1/jobs/{wid}", headers=KEY).status_code == 404


def test_security_headers_and_review_validation(client):
    c, _ = client
    h = c.get("/").headers
    assert "Content-Security-Policy" in h and h["X-Frame-Options"] == "DENY"
    wid = c.post("/web/analyze", files={"file": ("v.mov", io.BytesIO(b"1" * 10), "video/quicktime")}).json()["job_id"]
    wait_done(c, wid)
    assert c.post(f"/web/jobs/{wid}/review", json={"i": 99, "decision": "confirmed"}).status_code == 400
    assert c.post(f"/web/jobs/{wid}/review", json={"i": 0, "decision": "<script>"}).status_code == 400
