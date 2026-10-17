"""Проверка API на соответствие разделу 10 ТЗ — так, как его будет вызывать жюри.

  python3 tools/check_api.py --url https://<POD_ID>-8000.proxy.runpod.net --key <ключ> video1.mov video2.mov ...

Проверяет: ключ (401), POST поле file → job_id, статусы processing/done/failed, время < 300 с от загрузки,
схему ответа строго по ТЗ, verdict ⇔ violations, 1–3 кадра JPEG base64, таймкоды в пределах видео.
Все видео отправляются одновременно (как может сделать жюри).
"""
import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

LIMIT = 300
TOP = {"job_id", "status", "verdict", "violations"}
VIOL = {"type", "timecode_sec", "description", "evidence_frames"}
FRAME = {"timestamp_sec", "image_base64"}
errors = []


def err(name, msg):
    errors.append(f"{name}: {msg}")
    print(f"  ✗ {name}: {msg}")


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(name, r, dur=None):
    if set(r) != TOP:
        err(name, f"поля ответа {sorted(r)} ≠ {sorted(TOP)}")
    if r.get("verdict") not in ("violation", "no_violation"):
        err(name, f"verdict = {r.get('verdict')!r}")
    vs = r.get("violations")
    if not isinstance(vs, list):
        return err(name, "violations не список")
    if (r.get("verdict") == "violation") != bool(vs):
        err(name, f"verdict={r.get('verdict')}, а нарушений {len(vs)}")
    for i, v in enumerate(vs):
        p = f"{name} нарушение {i}"
        if set(v) != VIOL:
            err(p, f"поля {sorted(v)} ≠ {sorted(VIOL)}")
        if not isinstance(v.get("type"), str) or not v["type"].strip():
            err(p, "type пустой")
        if not num(v.get("timecode_sec")) or v["timecode_sec"] < 0 or (dur and v["timecode_sec"] > dur + 0.5):
            err(p, f"timecode_sec = {v.get('timecode_sec')!r}")
        if not isinstance(v.get("description"), str) or len(v["description"].strip()) < 10:
            err(p, "description пустое")
        fr = v.get("evidence_frames")
        if not isinstance(fr, list) or not 1 <= len(fr) <= 3:
            err(p, f"кадров {len(fr) if isinstance(fr, list) else fr!r}, нужно 1–3")
            continue
        for k, f in enumerate(fr):
            if set(f) != FRAME:
                err(p, f"кадр {k}: поля {sorted(f)}")
            if not num(f.get("timestamp_sec")):
                err(p, f"кадр {k}: timestamp_sec = {f.get('timestamp_sec')!r}")
            try:
                b = base64.b64decode(f.get("image_base64", ""), validate=True)
                if b[:3] != b"\xff\xd8\xff":
                    err(p, f"кадр {k}: не JPEG")
            except Exception:  # noqa
                err(p, f"кадр {k}: image_base64 не декодируется (нужен чистый base64 без data:)")


def duration(path):
    try:
        import cv2
        c = cv2.VideoCapture(path)
        fps, n = c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_FRAME_COUNT)
        return n / fps if fps else None
    except Exception:  # noqa
        return None


def run(url, key, path):
    name = path.rsplit("/", 1)[-1]
    t0 = time.time()
    with open(path, "rb") as fh:
        r = requests.post(f"{url}/api/v1/analyze", headers={"X-API-Key": key}, files={"file": (name, fh, "video/quicktime")}, timeout=120)
    if r.status_code != 200 or "job_id" not in r.json():
        err(name, f"POST → {r.status_code} {r.text[:200]}")
        return name, None, None
    jid = r.json()["job_id"]
    first = None
    while time.time() - t0 < LIMIT + 30:
        j = requests.get(f"{url}/api/v1/jobs/{jid}", headers={"X-API-Key": key}, timeout=30).json()
        first = first or j.get("status")
        if j.get("status") in ("done", "failed"):
            break
        if j.get("status") != "processing":
            err(name, f"неизвестный статус {j.get('status')!r}")
        time.sleep(2)
    dt = time.time() - t0
    if first != "processing" and j.get("status") != "done":
        err(name, f"первый статус {first!r}")
    if j.get("status") == "failed":
        err(name, f"failed: {j.get('error')}")
    elif j.get("status") != "done":
        err(name, "нет результата за 5 минут")
    else:
        if dt > LIMIT:
            err(name, f"готово за {dt:.0f} с — больше 300 с")
        validate(name, j, duration(path))
    return name, dt, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("videos", nargs="+")
    a = ap.parse_args()
    url = a.url.rstrip("/")

    print("1. Доступ по ключу")
    r1 = requests.post(f"{url}/api/v1/analyze", files={"file": ("x.mov", b"0")}, timeout=30).status_code
    r2 = requests.get(f"{url}/api/v1/jobs/none", headers={"X-API-Key": "wrong"}, timeout=30).status_code
    r3 = requests.post(f"{url}/api/v1/analyze", headers={"X-API-Key": "wrong"}, files={"file": ("x.mov", b"0")}, timeout=30).status_code
    for what, code in (("POST без ключа", r1), ("GET с неверным ключом", r2), ("POST с неверным ключом", r3)):
        print(f"  {'✓' if code in (401, 403) else '✗'} {what} → {code}")
        if code not in (401, 403):
            err("ключ", f"{what} → {code}")

    print(f"\n2. {len(a.videos)} видео одновременно (лимит {LIMIT} с от загрузки)")
    with ThreadPoolExecutor(len(a.videos)) as ex:
        res = list(ex.map(lambda p: run(url, a.key, p), a.videos))
    for name, dt, j in res:
        if j and j.get("status") == "done":
            vs = j["violations"]
            print(f"  ✓ {name}: {dt:.0f} с, {j['verdict']}, нарушений {len(vs)}"
                  + (": " + ", ".join(f"{v['timecode_sec']} с «{v['type']}» ({len(v['evidence_frames'])} кадр.)" for v in vs) if vs else ""))
    print("\n3. Схема ответа по ТЗ:", "совпадает" if not errors else f"ошибок {len(errors)}")
    if res and res[-1][2]:
        ex = json.loads(json.dumps(res[-1][2]))
        for v in ex.get("violations", []):
            for f in v["evidence_frames"]:
                f["image_base64"] = f["image_base64"][:12] + "..."
        print(json.dumps(ex, ensure_ascii=False, indent=1)[:900])
    print("\nИтог:", "API соответствует разделу 10 ТЗ" if not errors else "ЕСТЬ ОШИБКИ:\n  " + "\n  ".join(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
