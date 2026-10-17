"""Общие функции для скриптов: чтение разметки и поиск видеофайлов."""
import csv, glob, os

VIDEO_EXT = (".mov", ".mp4", ".MOV", ".MP4", ".avi", ".mkv")


def find_videos(folder):
    """{имя без расширения: путь}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if p.endswith(VIDEO_EXT):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def read_labels(path):
    """CSV с колонками video,t_sec[,type_id[,note]].
    Пустой t_sec — в видео нарушений нет. Несколько строк на видео — несколько нарушений.
    Возвращает {video: [ {t, type_id, note}, ... ]}."""
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            name = os.path.splitext(os.path.basename(row.get("video", "")))[0]
            if not name:
                continue
            labels.setdefault(name, [])
            t = row.get("t_sec", "").replace(",", ".")
            if t and t.lower() not in ("-", "none", "nan"):
                labels[name].append({"t": float(t), "type_id": row.get("type_id") or "", "note": row.get("note", "")})
    return labels
