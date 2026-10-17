"""Аннотированное видео: зоны, люди (track_id и роль), скелет RTMPose, телефоны и предметы,
линия передачи, красный бейдж нарушения и таймлайн событий. Результат — results/{job_id}_annotated.mp4.

Пишем H.264 через ffmpeg (системный или imageio-ffmpeg), чтобы видео играло в браузере;
без ffmpeg — mp4v средствами OpenCV. Подписи латиницей: шрифты OpenCV не умеют кириллицу.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from bisect import bisect_left
from typing import Any

import cv2
import numpy as np

from rule_engine.base import CASHIER, CLIENT, Scene

log = logging.getLogger("vis")

COL = {CASHIER: (200, 200, 0), CLIENT: (0, 210, 255), "UNKNOWN": (170, 170, 170)}        # BGR: бирюзовый, жёлтый
ZCOL = {"cashier_zone": (200, 200, 0), "client_zone": (0, 210, 255), "transfer_zone": (0, 160, 0),
        "restricted_zone": (60, 60, 230)}
OCOL = {"phone": (0, 120, 255), "small_object": (255, 0, 200)}
RED = (40, 40, 230)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)]
TRANSLIT = {"CASH_CONCEALMENT": "CASH CONCEALMENT", "NO_RECOUNT_IN_FRONT": "NO RECOUNT", "PAYOUT_NO_RECOUNT": "PAYOUT NO RECOUNT",
            "CASHIER_LEFT_DURING_SERVICE": "CASHIER LEFT", "CASH_LEFT_UNATTENDED": "CASH UNATTENDED",
            "PHONE_AT_CASH_DESK": "PHONE", "BAG_ACCEPTED_NO_CHECK": "BAG NOT CHECKED", "BAG_OPENED_NO_COLLECTOR": "BAG OPENED",
            "UNAUTHORIZED_PERSON": "UNAUTHORIZED PERSON", "OTHER": "SUSPICIOUS"}


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


class Writer:
    def __init__(self, path: str, w: int, h: int, fps: float):
        self.proc = None
        self.cv = None
        exe = ffmpeg_exe()
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}",
                   "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, fr: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(fr.tobytes())
        else:
            self.cv.write(fr)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait(timeout=600) != 0:
                raise RuntimeError("ffmpeg завершился с ошибкой")
        elif self.cv:
            self.cv.release()


def _txt(img, text, org, color=(255, 255, 255), scale=0.5, bg=None, th=1):
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - tht - 4), (x + tw + 2, y + base), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def draw_frame(img: np.ndarray, t: float, sample, zones, s: float, violations: list[dict[str, Any]], dur: float,
               cfg: dict[str, Any], transfers: list[dict[str, Any]], kp_thr: float) -> np.ndarray:
    h, w = img.shape[:2]
    # зоны
    if zones is not None and not zones.empty:
        over = img.copy()
        for name, poly in zones.polys.items():
            pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
            cv2.fillPoly(over, [pts], ZCOL.get(name, (128, 128, 128)))
        cv2.addWeighted(over, float(cfg.get("zone_alpha", 0.18)), img, 1 - float(cfg.get("zone_alpha", 0.18)), 0, img)
        for name, poly in zones.polys.items():
            pts = np.array([[x * w, y * h] for x, y in poly], np.int32)
            cv2.polylines(img, [pts], True, ZCOL.get(name, (128, 128, 128)), 1, cv2.LINE_AA)
            _txt(img, name.replace("_zone", ""), (pts[:, 0].min() + 4, max(pts[:, 1].min() + 16, 16)), (0, 0, 0), 0.45,
                 ZCOL.get(name, (200, 200, 200)))
    hot = [v for v in violations if abs(v["timecode_sec"] - t) <= 2.0]
    hot_ids = {i for v in hot for i in ((v.get("rule_event") or {}).get("involved_track_ids") or [])}
    if sample is not None:
        for p in sample.persons:
            if p.score < float(cfg.get("min_person_score", 0.4)):
                continue
            red = p.id in hot_ids or (hot and not hot_ids and p.role == CASHIER)
            col = RED if red else COL.get(p.role, COL["UNKNOWN"])
            b = [int(v * s) for v in p.box]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2)
            _txt(img, f"#{p.id} {p.role}", (b[0] + 2, max(b[1] - 4, 12)), (0, 0, 0), 0.5, col)
            for a, c in SKELETON:
                if p.sc[a] > kp_thr and p.sc[c] > kp_thr:
                    cv2.line(img, tuple(int(v * s) for v in p.kp[a]), tuple(int(v * s) for v in p.kp[c]), col, 2, cv2.LINE_AA)
            for k in range(17):
                if p.sc[k] > kp_thr:
                    cv2.circle(img, tuple(int(v * s) for v in p.kp[k]), 3, (255, 255, 255), -1)
        for o in sample.objects:
            if o.score < float(cfg.get("min_object_score", 0.35)):
                continue
            b = [int(v * s) for v in o.box]
            col = OCOL.get(o.cls, (255, 255, 255))
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 2)
            _txt(img, f"{o.cls} {o.score:.2f}", (b[0], max(b[1] - 3, 12)), (0, 0, 0), 0.45, col)
        # линия передачи между ближайшими запястьями кассира и клиента
        for tr in transfers:
            if tr["start_timecode_sec"] - 0.3 <= t <= tr["end_timecode_sec"] + 0.3:
                ids = set(tr["involved_track_ids"])
                ps = [p for p in sample.persons if p.id in ids]
                cs = [p for p in ps if p.role == CASHIER]
                os_ = [p for p in ps if p.role != CASHIER]
                if cs and os_:
                    pairs = [(a, b) for a in (9, 10) for b in (9, 10) if cs[0].sc[a] > kp_thr and os_[0].sc[b] > kp_thr]
                    if pairs:
                        a, b = min(pairs, key=lambda ab: np.linalg.norm(cs[0].kp[ab[0]] - os_[0].kp[ab[1]]))
                        p1 = tuple(int(v * s) for v in cs[0].kp[a])
                        p2 = tuple(int(v * s) for v in os_[0].kp[b])
                        cv2.arrowedLine(img, p1, p2, (0, 255, 0), 3, cv2.LINE_AA, tipLength=0.2)
                        _txt(img, "TRANSFER", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 6), (0, 0, 0), 0.5, (0, 255, 0))
    # бейдж нарушения
    y = 28
    for v in hot:
        label = f"ALERT {TRANSLIT.get((v.get('rule') or {}).get('id', ''), 'VIOLATION')}  t={v['timecode_sec']:.1f}s"
        _txt(img, label, (12, y), (255, 255, 255), 0.7, RED, 2)
        y += 30
    # таймлайн
    bh, y0 = 18, h - 26
    x0, x1 = 10, w - 10
    cv2.rectangle(img, (x0, y0), (x1, y0 + bh), (40, 40, 40), -1)
    for v in violations:
        x = int(x0 + (x1 - x0) * v["timecode_sec"] / max(dur, 1e-6))
        cv2.rectangle(img, (x - 3, y0), (x + 3, y0 + bh), RED, -1)
    xc = int(x0 + (x1 - x0) * t / max(dur, 1e-6))
    cv2.line(img, (xc, y0 - 4), (xc, y0 + bh + 4), (255, 255, 255), 2)
    _txt(img, f"{t:6.1f}s", (x1 - 70, y0 - 6), (255, 255, 255), 0.45, (40, 40, 40))
    return img


def render(video_path: str, out_path: str, scene: Scene | None, zones, violations: list[dict[str, Any]],
           rule_events: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """Рисует аннотированное видео. scene — сцена Следопыта (координаты в кадрах CACHE_SIDE) или None."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / src_fps if src_fps else 0
    out_fps = min(float(cfg.get("fps", 15)), src_fps)
    step = max(int(round(src_fps / out_fps)), 1)
    out_fps = src_fps / step
    max_side = int(cfg.get("max_side", 960))
    times = scene.times if scene else []
    transfers = [e for e in rule_events if e.get("violation_type") == "transfer_of_item"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part.mp4"
    writer = None
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            ok, fr = cap.retrieve()
            if not ok:
                break
            t = idx / src_fps
            idx += 1
            hh, ww = fr.shape[:2]
            k = min(max_side / max(hh, ww), 1.0)
            img = cv2.resize(fr, (int(ww * k) // 2 * 2, int(hh * k) // 2 * 2), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = Writer(tmp, img.shape[1], img.shape[0], out_fps)
            sample, s = None, 1.0
            if scene and times:
                j = bisect_left(times, t)
                cand = [i for i in (j - 1, j) if 0 <= i < len(times)]
                i = min(cand, key=lambda i: abs(times[i] - t))
                if abs(times[i] - t) <= 1.0 / scene.fps:
                    sample = scene.frames[i]
                s = img.shape[1] / scene.w
            writer.write(draw_frame(img, t, sample, zones, s, violations, dur, cfg, transfers,
                                    scene.kp_thr if scene else 0.35))
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("не удалось прочитать кадры видео")
    os.replace(tmp, out_path)
    return out_path
