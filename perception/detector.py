"""YOLOX (COCO-80, Megvii, Apache 2.0) через onnxruntime: телефоны и мелкие предметы (и люди — по желанию).

Модель: официальные ONNX из релиза YOLOX 0.1.1rc0 (yolox_s / yolox_tiny / yolox_nano), скачивает
tools/fetch_models.py в MODELS_DIR. Нет файла — детектор объектов выключен, остальной пайплайн работает.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("detector")


@dataclass
class Detection:
    cls: str            # person | phone | small_object
    coco_id: int
    box: list[float]    # x1, y1, x2, y2 в пикселях
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"cls": self.cls, "coco_id": self.coco_id, "box": [round(v, 1) for v in self.box], "score": round(self.score, 3)}


def preprocess(img: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Letterbox как в YOLOX: масштаб без искажения, паддинг 114 справа/снизу, BGR, без нормализации."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(h * r), int(w * r)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None], r


def decode(outputs: np.ndarray, size: int, strides: tuple[int, ...] = (8, 16, 32)) -> np.ndarray:
    """Сетка YOLOX: (N, 85) — cx, cy, w, h в пикселях входа, obj, 80 классов."""
    grids, exp_strides = [], []
    for s in strides:
        hs = ws = size // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        exp_strides.append(np.full((hs * ws, 1), s))
    grid = np.concatenate(grids, 0)
    st = np.concatenate(exp_strides, 0)
    out = outputs.copy()
    out[:, :2] = (out[:, :2] + grid) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def postprocess(raw: np.ndarray, ratio: float, size: int, class_map: dict[int, str],
                score_thr: float, nms_thr: float) -> list[Detection]:
    pred = decode(raw, size)
    ids = np.array(sorted(class_map), dtype=int)
    if not len(ids):
        return []
    scores = pred[:, 4:5] * pred[:, 5 + ids]           # obj × cls только по нужным классам
    best = scores.argmax(1)
    best_score = scores[np.arange(len(scores)), best]
    m = best_score >= score_thr
    if not m.any():
        return []
    p, cls_idx, sc = pred[m], ids[best[m]], best_score[m]
    boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1) / ratio
    out: list[Detection] = []
    for c in np.unique(cls_idx):                        # NMS по каждому классу
        mc = np.where(cls_idx == c)[0]
        for k in nms(boxes[mc], sc[mc], nms_thr):
            j = mc[k]
            out.append(Detection(class_map[int(c)], int(c), [float(v) for v in boxes[j]], float(sc[j])))
    return out


class ObjectDetector:
    def __init__(self, model_path: str, cfg: dict[str, Any], device: str = "auto"):
        import onnxruntime as ort
        self.size = int(cfg.get("input_size", 640))
        self.score_thr = float(cfg.get("score_thr", 0.3))
        self.nms_thr = float(cfg.get("nms_thr", 0.45))
        self.class_map: dict[int, str] = {}
        for name, idx in (cfg.get("classes") or {}).items():
            for i in idx:
                self.class_map[int(i)] = name
        if not cfg.get("use_for_persons"):
            self.class_map = {k: v for k, v in self.class_map.items() if v != "person"}
        prov = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=prov)
        self.input = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        if isinstance(shp[-1], int) and shp[-1] != self.size:
            log.info("размер входа модели %s ≠ input_size %s — беру из модели", shp[-1], self.size)
            self.size = int(shp[-1])
        self.provider = "cuda" if "CUDAExecutionProvider" in self.sess.get_providers() else "cpu"
        log.info("YOLOX COCO: %s, вход %d, классы %s, считает на %s", os.path.basename(model_path), self.size,
                 sorted(set(self.class_map.values())), self.provider)

    def __call__(self, img: np.ndarray) -> list[Detection]:
        x, r = preprocess(img, self.size)
        raw = self.sess.run(None, {self.input: x})[0][0]
        return postprocess(raw, r, self.size, self.class_map, self.score_thr, self.nms_thr)


_det: ObjectDetector | None = None
_det_failed = False


def model_path(cfg: dict[str, Any]) -> str | None:
    from settings import MODELS_DIR
    p = os.getenv("OBJ_DET_MODEL") or cfg.get("model") or "yolox_s.onnx"
    if not os.path.isabs(p):
        p = os.path.join(MODELS_DIR, p)
    return p if os.path.exists(p) else None


def get(cfg: dict[str, Any]) -> ObjectDetector | None:
    """Ленивая загрузка; None — детектор недоступен (нет файла или onnxruntime)."""
    global _det, _det_failed
    if _det is not None or _det_failed:
        return _det
    if os.getenv("USE_OBJ_DET", "1") != "1":
        _det_failed = True
        return None
    path = model_path(cfg)
    if not path:
        log.warning("нет ONNX детектора объектов (OBJ_DET_MODEL / MODELS_DIR) — телефоны и предметы не детектируются; "
                    "скачать: python3 tools/fetch_models.py")
        _det_failed = True
        return None
    try:
        _det = ObjectDetector(path, cfg, os.getenv("POSE_DEVICE", "auto"))
    except Exception as e:  # noqa: BLE001 — детектор необязателен
        log.warning("детектор объектов не загрузился: %r", e)
        _det_failed = True
    return _det
