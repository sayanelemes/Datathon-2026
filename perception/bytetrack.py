"""ByteTrack (Zhang et al., 2022): фильтр Калмана + двухпроходное сопоставление по IoU.

Компактная реализация на numpy без внешних зависимостей (scipy — если есть, для венгерского алгоритма).
Первый проход — уверенные детекции, второй — слабые (держат трек при перекрытии рукой, стойкой и т.п.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # noqa: BLE001
    linear_sum_assignment = None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(iou: np.ndarray, min_iou: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    rows, cols = iou.shape
    pairs: list[tuple[int, int]] = []
    if rows and cols:
        if linear_sum_assignment is not None:
            r, c = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(r, c) if iou[i, j] >= min_iou]
        else:  # жадно по убыванию IoU
            used_r, used_c = set(), set()
            for i, j in sorted(((i, j) for i in range(rows) for j in range(cols)), key=lambda p: -iou[p]):
                if iou[i, j] < min_iou:
                    break
                if i not in used_r and j not in used_c:
                    pairs.append((i, j))
                    used_r.add(i)
                    used_c.add(j)
    mr = {i for i, _ in pairs}
    mc = {j for _, j in pairs}
    return pairs, set(range(rows)) - mr, set(range(cols)) - mc


class Kalman:
    """Постоянная скорость; состояние (cx, cy, a, h, vx, vy, va, vh), как в SORT/ByteTrack."""
    _std_pos, _std_vel = 1 / 20, 1 / 160

    def __init__(self) -> None:
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        F = self.F.copy()
        for i in range(4):
            F[i, 4 + i] = dt
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        Q = np.diag(np.square(std)) * max(dt, 1e-3)
        return F @ mean, F @ cov @ F.T + Q

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        R = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], max(b[3] - b[1], 1e-6)
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    last_t: float
    hits: int = 1
    payload: Any = None
    state: str = "tracked"          # tracked | lost
    pred_t: float = 0.0             # время последнего прогноза

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


class ByteTracker:
    def __init__(self, cfg: dict[str, Any] | None = None, fps: float = 5.0):
        cfg = cfg or {}
        self.high = float(cfg.get("track_thresh", 0.5))
        self.low = float(cfg.get("low_thresh", 0.1))
        self.new_thr = float(cfg.get("new_track_thresh", 0.6))
        self.match_iou = float(cfg.get("match_iou", 0.2))
        self.low_match_iou = float(cfg.get("low_match_iou", 0.5))
        self.lost_match_iou = float(cfg.get("lost_match_iou", 0.5))
        self.buffer = float(cfg.get("track_buffer_sec", 2.0))
        self.kf = Kalman()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.fps = fps

    def update(self, t: float, boxes: np.ndarray, scores: np.ndarray, payloads: list[Any] | None = None) -> list[tuple[int, int]]:
        """boxes (N,4) xyxy, scores (N,) → [(индекс детекции, track_id)]."""
        payloads = payloads if payloads is not None else [None] * len(boxes)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        # прогноз
        for tr in self.tracks:
            dt = max((t - tr.pred_t) * self.fps, 1e-3)      # шаг в «кадрах трекера»
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov, dt)
            tr.pred_t = t
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]
        result: list[tuple[int, int]] = []

        pool = list(self.tracks)
        tb = np.array([tr.box for tr in pool]).reshape(-1, 4)
        iou = iou_matrix(tb, boxes[hi])
        if len(pool) and len(hi):
            # потерянный трек «возвращаем» только при сильном перекрытии: иначе другой человек,
            # зашедший на место ушедшего кассира, унаследует его track_id
            lost = np.array([tr.state == "lost" for tr in pool])
            iou[lost] = np.where(iou[lost] >= self.lost_match_iou, iou[lost], 0.0)
        pairs, un_tr, un_hi = match(iou, self.match_iou)
        for i, j in pairs:
            self._apply(pool[i], t, boxes[hi[j]], scores[hi[j]], payloads[hi[j]])
            result.append((int(hi[j]), pool[i].id))
        # второй проход: слабые детекции только к ещё «живым» трекам
        rem = [pool[i] for i in sorted(un_tr) if pool[i].state == "tracked"]
        rb = np.array([tr.box for tr in rem]).reshape(-1, 4)
        pairs2, _, _ = match(iou_matrix(rb, boxes[lo]), self.low_match_iou)
        for i, j in pairs2:
            self._apply(rem[i], t, boxes[lo[j]], scores[lo[j]], payloads[lo[j]])
            result.append((int(lo[j]), rem[i].id))
        matched_ids = {tid for _, tid in result}
        for tr in self.tracks:
            if tr.id not in matched_ids:
                tr.state = "lost"
        # новые треки из несопоставленных уверенных детекций
        for j in sorted(un_hi):
            k = hi[j]
            if scores[k] < self.new_thr:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(boxes[k]))
            tr = Track(self.next_id, mean, cov, float(scores[k]), t, payload=payloads[k], pred_t=t)
            self.next_id += 1
            self.tracks.append(tr)
            result.append((int(k), tr.id))
        # удаляем давно потерянные
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.buffer]
        return result

    def _apply(self, tr: Track, t: float, box: np.ndarray, score: float, payload: Any) -> None:
        tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, xyxy_to_xyah(box))
        tr.score, tr.last_t, tr.payload, tr.state = float(score), t, payload, "tracked"
        tr.hits += 1


def track_sequence(seq: list[tuple[float, list[dict[str, Any]]]], cfg: dict[str, Any] | None = None,
                   fps: float = 5.0, key: str = "id") -> list[tuple[float, list[dict[str, Any]]]]:
    """seq: [(t, [ {box, score?, ...}, ... ])] → те же объекты с полем key (track_id).
    Объекты без трека (слабые, не сопоставленные) отбрасываются."""
    trk = ByteTracker(cfg, fps)
    out = []
    for t, items in seq:
        if not items:
            trk.update(t, np.zeros((0, 4)), np.zeros(0))
            out.append((t, []))
            continue
        boxes = np.array([it["box"] for it in items], dtype=float)
        scores = np.array([float(it.get("score", 1.0)) for it in items])
        kept = []
        for k, tid in trk.update(t, boxes, scores):
            items[k][key] = tid
            kept.append(items[k])
        out.append((t, kept))
    return out
