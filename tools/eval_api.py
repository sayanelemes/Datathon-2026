"""Прогон видео через API — так же, как это сделает жюри — и подсчёт качества.

  python tools/eval_api.py --url http://localhost:8000 --key КЛЮЧ --videos /workspace/dataset \
         --labels labels.csv [--only val.txt] [--tol 2.0]

Считает:
  - вердикт по видео: верно/неверно, пропуски (FN) и ложные срабатывания (FP);
  - для найденных нарушений — ошибку таймкода относительно ближайшей метки;
  - время обработки каждого видео (лимит ТЗ — 300 с).
Сохраняет подробный отчёт в eval_report.csv.
"""
import argparse, csv, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.common import find_videos, read_labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--key", required=True)
ap.add_argument("--videos", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--only", default=None, help="файл со списком видео для проверки (например, отложенные)")
ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода, с")
ap.add_argument("--out", default="eval_report.csv")
ap.add_argument("--parallel", type=int, default=2, help="сколько видео отправлять одновременно (= WORKERS сервиса)")
args = ap.parse_args()

H = {"X-API-Key": args.key}
vids = find_videos(args.videos)
labels = read_labels(args.labels)
names = [n for n in vids if n in labels]
if args.only:
    keep = {os.path.splitext(x.strip())[0] for x in open(args.only, encoding="utf-8") if x.strip()}
    names = [n for n in names if n in keep]
if not names:
    sys.exit("Нет видео с разметкой. Проверьте пути.")

rows = []
tp = fp = fn = tn = 0
tc_err, types_ok, types_all, slow = [], 0, 0, 0
lock = threading.Lock()


def run(n):
    global tp, fp, fn, tn, types_ok, types_all, slow
    t0 = time.time()
    try:
        with open(vids[n], "rb") as f:
            r = requests.post(f"{args.url}/api/v1/analyze", headers=H, files={"file": (os.path.basename(vids[n]), f)}, timeout=120)
        r.raise_for_status()
        jid = r.json()["job_id"]
        while True:
            j = requests.get(f"{args.url}/api/v1/jobs/{jid}", headers=H, timeout=30).json()
            if j.get("status") in ("done", "failed") or time.time() - t0 > 330:
                break
            time.sleep(2)
    except Exception as e:  # noqa
        jid, j = "-", {"status": "failed", "error": repr(e)}
    el = time.time() - t0
    truth = labels[n]
    want = bool(truth)
    got = j.get("verdict") == "violation"
    if j.get("status") != "done":
        got = None
    errs, t_ok, t_all = [], 0, 0
    for vv in j.get("violations", []) or []:
        if truth:
            m = min(truth, key=lambda x: abs(x["t"] - vv["timecode_sec"]))
            errs.append(abs(m["t"] - vv["timecode_sec"]))
    status = "OK" if got == want else ("ПРОПУСК" if want else "ЛОЖНОЕ") if got is not None else "НЕ ОБРАБОТАНО"
    found = "; ".join(f'{x["type"]} @ {x["timecode_sec"]}' for x in j.get("violations", []) or [])
    with lock:
        slow += el > 300
        tp += want and got is True
        fn += want and got is not True
        fp += (not want) and got is True
        tn += (not want) and got is False
        if errs:
            tc_err.append(min(errs))
        print(f"{status:14s} {n:30s} метки={[m['t'] for m in truth]}  найдено=[{found}]  {el:.0f} с"
              + (f"  ({j.get('error')})" if j.get("status") == "failed" else ""), flush=True)
        rows.append({"video": n, "status": status, "labels": [m["t"] for m in truth], "found": found,
                     "timecode_err": round(min(errs), 2) if errs else "", "seconds": round(el, 1), "job_id": jid})


print(f"Видео: {len(names)}, одновременно: {args.parallel}", flush=True)
with ThreadPoolExecutor(args.parallel) as ex:
    list(ex.map(run, names))
rows.sort(key=lambda r: (r["status"] == "OK", r["video"]))

with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

N = len(names)
print("\n========== ИТОГ ==========")
print(f"Видео: {N}   Верный вердикт: {tp + tn}/{N} ({100 * (tp + tn) / N:.0f}%)")
print(f"Пропуски (самое плохое): {fn}   Ложные срабатывания: {fp}")
if tc_err:
    within = sum(e <= args.tol for e in tc_err)
    print(f"Таймкод: средняя ошибка {sum(tc_err) / len(tc_err):.1f} с, в пределах {args.tol} с: {within}/{len(tc_err)}")
print(f"Дольше 5 минут: {slow}")
print(f"Отчёт: {args.out}")
