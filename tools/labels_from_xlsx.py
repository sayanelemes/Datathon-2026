"""Разметка организаторов (open_labels.xlsx) → labels.csv для tools/eval_api.py + отложенная выборка val.txt.

  python3 tools/labels_from_xlsx.py /root/dataset/data/open/open_labels.xlsx /root/dataset/data/open/videos \
          --out labels.csv --val val.txt --val-share 0.2

Учитывает: латинскую X и кириллическую Х в названиях, время текстом («22 секунда», «1минута 8 секунда»).
Имена в labels.csv — как у реальных файлов видео (иначе eval их не сопоставит).
"""
import argparse
import csv
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.common import find_videos  # noqa: E402

CYR = str.maketrans({"Х": "X", "х": "x", "Е": "E", "е": "e", "А": "A", "а": "a", "О": "O", "о": "o", "С": "C", "с": "c",
                     "Р": "P", "р": "p", "К": "K", "к": "k", "М": "M", "Т": "T", "Н": "H", "В": "B"})


def norm(name):
    return re.sub(r"[\s_]+", "", os.path.splitext(str(name))[0].translate(CYR)).lower()


def seconds(v):
    if v is None or str(v).strip() in ("", "-", "None"):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*мин", s)
    sec = re.search(r"(\d+(?:\.\d+)?)\s*сек", s)
    if m or sec:
        return (float(m.group(1)) * 60 if m else 0.0) + (float(sec.group(1)) if sec else 0.0)
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", s.strip())          # 1:08
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    return float(nums[0]) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("videos")
    ap.add_argument("--out", default="labels.csv")
    ap.add_argument("--val", default="val.txt")
    ap.add_argument("--val-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=26)
    a = ap.parse_args()

    import openpyxl
    ws = openpyxl.load_workbook(a.xlsx, data_only=True).active
    vids = find_videos(a.videos)
    by_norm = {norm(k): k for k in vids}
    rows, missing = [], []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or not r or not r[0]:
            continue
        name, ts, verdict = r[0], r[1] if len(r) > 1 else None, str(r[2] if len(r) > 2 else "").lower()
        real = by_norm.get(norm(name))
        if not real:
            missing.append(str(name))
            continue
        t = seconds(ts)
        viol = "без" not in verdict and ("наруш" in verdict or t is not None)
        if viol and t is None:
            print(f"[!] {name}: нарушение без таймкода — пропускаю метку времени")
        rows.append((real, t if viol else None))

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "t_sec", "type_id", "note"])
        for v, t in rows:
            w.writerow([v, "" if t is None else t, "", ""])

    # отложенная выборка: поровну из видео с нарушением и без (стратификация), фиксированный seed
    rnd = random.Random(a.seed)
    pos = sorted(v for v, t in rows if t is not None)
    neg = sorted(v for v, t in rows if t is None)
    val = rnd.sample(pos, round(len(pos) * a.val_share)) + rnd.sample(neg, round(len(neg) * a.val_share))
    with open(a.val, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(val)) + "\n")

    print(f"видео в папке: {len(vids)}, в разметке сопоставлено: {len(rows)} "
          f"(с нарушением {len(pos)}, без {len(neg)}); отложено в {a.val}: {len(val)}")
    if missing:
        print(f"[!] нет файла для: {', '.join(missing)}")
    extra = sorted(set(vids) - {v for v, _ in rows})
    if extra:
        print(f"[i] видео без разметки: {', '.join(extra)}")


if __name__ == "__main__":
    main()
