"""Черновая разметка типов нарушений с помощью локального Qwen.

  python tools/autolabel.py --videos /workspace/dataset --labels labels_raw.csv --out labels_draft.csv

На вход — разметка организаторов в формате video,t_sec (тип пустой).
Для каждой метки Qwen смотрит кадры t-3..t+3 с и пишет: что происходит и какой тип из configs/violations.yaml подходит.
Результат — labels_draft.csv. ЕГО НУЖНО ПРОСМОТРЕТЬ ЧЕЛОВЕКУ и поправить type_id, затем сохранить как labels.csv.
Новые типы, которых нет в rules.yaml, модель помечает как OTHER и описывает — их стоит добавить в configs/violations.yaml.
"""
import argparse, csv, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as P  # noqa: E402
from tools.common import find_videos, read_labels  # noqa: E402

PROMPT = P.BASE + (
    "\nКадры ниже — фрагмент вокруг момента, который эксперт отметил как НАРУШЕНИЕ (около t={t:.1f}с).\n"
    "Опиши подробно, что делают люди (особенно руки кассира, деньги, сумки, телефон, двери), "
    "и определи, какое нарушение из списка здесь показано. Если ни одно не подходит — type_id=OTHER "
    "и в new_type кратко назови, что это за нарушение.\n"
    'Ответ строго JSON: {{"steps": "что происходит по шагам", "type_id": "<id или OTHER>", '
    '"new_type": "<название, если OTHER, иначе пусто>", "key_time": <секунда самого явного момента>}}'
)

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--out", default="labels_draft.csv")
args = ap.parse_args()

vids = find_videos(args.videos)
labels = read_labels(args.labels)


def one(name, mark):
    v = P.Video(vids[name])
    ts = [mark["t"] - 3 + i * 0.5 for i in range(13)]
    fr = sorted(v.frames(ts).items())
    content = P.frames_content(fr, None, "Опиши и определи тип. Ответь JSON.")
    try:
        a = P.chat(PROMPT.format(rules=P.rules_text(), t=mark["t"]), content, max_tokens=700) or {}
    except Exception as e:  # noqa
        a = {"steps": f"ошибка: {e}"}
    return {"video": name, "t_sec": mark["t"], "type_id": a.get("type_id", ""), "new_type": a.get("new_type", ""),
            "key_time": a.get("key_time", ""), "note": a.get("steps", "")}


tasks = [(n, m) for n, ms in labels.items() if n in vids for m in ms]
no_viol = [n for n, ms in labels.items() if n in vids and not ms]
print(f"Меток: {len(tasks)}, видео без нарушений: {len(no_viol)}")
with ThreadPoolExecutor(8) as ex:
    rows = list(ex.map(lambda x: one(*x), tasks))
rows += [{"video": n, "t_sec": "", "type_id": "NONE", "new_type": "", "key_time": "", "note": "без нарушений"} for n in no_viol]

with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["video", "t_sec", "type_id", "new_type", "key_time", "note"])
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["video"], str(r["t_sec"]))))

from collections import Counter  # noqa: E402
print("Предложенные типы:", dict(Counter(r["type_id"] for r in rows)))
print("Новые типы:", sorted({r["new_type"] for r in rows if r["new_type"]}))
print(f"Готово: {args.out} — просмотрите и поправьте вручную.")
