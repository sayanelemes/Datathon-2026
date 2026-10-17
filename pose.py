"""Следопыт: YOLOX (люди — rtmlib, телефоны/предметы — COCO) → RTMPose → ByteTrack → зоны → роли → Rule Engine.

Что даёт пайплайну:
  tracks      — рамки людей и предметов по времени (для отрисовки на видео в интерфейсе);
  candidates  — моменты от правил с подсказкой для перепроверки Qwen (формат pipeline);
  rule_events — результаты правил в формате ТЗ (violation_type, таймкоды, track_id, кадры, уверенность, raw_evidence);
  scene       — сцена целиком (ключевые точки) для аннотированного видео.
Лица не распознаются: используются только точки тела (плечи, запястья, бёдра) и рамки.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

import rule_engine
import settings
from perception import detector as objdet
from perception.bytetrack import track_sequence
from rule_engine import Frame, Obj, Person, Scene
from zones import Zones

log = logging.getLogger("pose")

POSE_FPS = float(os.getenv("POSE_FPS", 5))
POSE_MODE = os.getenv("POSE_MODE", "balanced")     # lightweight | balanced | performance
KP_T = float((settings.rules_config().get("pose") or {}).get("keypoint_thr", 0.35))

_model = None
PROVIDER = "-"                                       # чем реально считает RTMPose (для /health и selftest)


def available() -> bool:
    try:
        import rtmlib  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _providers(m: Any) -> list[str]:
    for part in ("det_model", "pose_model"):
        sess = getattr(getattr(m, part, None), "session", None)
        if sess is not None:
            return sess.get_providers()
    return []


def _load():
    global _model, PROVIDER
    if _model is None:
        from rtmlib import Body
        dev = os.getenv("POSE_DEVICE", "auto")   # cpu — оставить видеопамять для Qwen на слабых GPU
        if dev in ("auto", "cuda"):
            try:
                import onnxruntime as ort
                if hasattr(ort, "preload_dlls"):   # CUDA/cuDNN из pip-пакетов nvidia-* (те же, что у PyTorch в образе vLLM)
                    try:
                        ort.preload_dlls()
                    except Exception as e:  # noqa: BLE001
                        log.info("ort.preload_dlls: %s", e)
                dev = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
            except Exception:  # noqa: BLE001
                dev = "cpu"
        try:
            _model = Body(mode=POSE_MODE, backend="onnxruntime", device=dev)
        except Exception as e:  # noqa: BLE001
            if dev == "cpu":
                raise
            log.warning("RTMPose на GPU не запустился, перехожу на CPU: %s", e)
            _model = Body(mode=POSE_MODE, backend="onnxruntime", device="cpu")
        prov = _providers(_model)
        PROVIDER = "cuda" if "CUDAExecutionProvider" in prov else "cpu"
        log.info("RTMPose: режим %s, считает на %s %s", POSE_MODE, PROVIDER, prov)
    return _model


def _det_cfg() -> dict[str, Any]:
    return settings.rules_config().get("detector") or {}


def detect_objects(frame: np.ndarray) -> list[objdet.Detection]:
    d = objdet.get(_det_cfg())
    return d(frame) if d is not None else []


def detect(frame: np.ndarray, objects: list[objdet.Detection] | None = None) -> list[dict[str, Any]]:
    """-> список людей: {box: [x1,y1,x2,y2] px, kp: (17,2), sc: (17,), score}"""
    m = _load()
    person_boxes = [o for o in (objects or []) if o.cls == "person"]
    if person_boxes and _det_cfg().get("use_for_persons") and hasattr(m, "pose_model"):
        boxes = np.array([o.box for o in person_boxes])
        dscore = [o.score for o in person_boxes]
        kps, scs = m.pose_model(frame, bboxes=boxes)
    elif hasattr(m, "det_model") and hasattr(m, "pose_model"):
        boxes = m.det_model(frame)
        if boxes is None or len(boxes) == 0:
            return []
        dscore = [0.9] * len(boxes)                # YOLOX rtmlib уже отфильтровал по порогу и не отдаёт score
        kps, scs = m.pose_model(frame, bboxes=boxes)
    else:
        kps, scs = m(frame)
        boxes, dscore = None, [0.9] * len(kps)
    out = []
    for i in range(len(kps)):
        kp, sc = np.asarray(kps[i])[:17], np.asarray(scs[i])[:17]
        if boxes is not None:
            b = [float(x) for x in boxes[i][:4]]
        else:
            ok = sc > KP_T
            if ok.sum() < 3:
                continue
            p = kp[ok]
            b = [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
        out.append({"box": b, "kp": kp, "sc": sc, "score": float(dscore[i])})
    return out


def build_scene(seq: list[tuple[float, list[dict]]], obj_seq: list[tuple[float, list[dict]]],
                zones: Zones, w: int, h: int, crop_zone: list[float] | None = None) -> tuple[Scene, dict[int, str]]:
    """Трекинг (ByteTrack) людей и предметов, роли. seq/obj_seq — сырые детекции по времени."""
    cfg = settings.rules_config()
    tcfg = cfg.get("tracker") or {}
    min_area = float(tcfg.get("min_box_area", 0.0)) * w * h
    seq = [(t, [p for p in ps if (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) >= min_area]) for t, ps in seq]
    seq = track_sequence(seq, tcfg, POSE_FPS)
    ocfg = {"track_thresh": 0.4, "low_thresh": 0.2, "new_track_thresh": 0.35, "match_iou": 0.1, "low_match_iou": 0.3,
            "track_buffer_sec": 1.0, **(cfg.get("object_tracker") or {})}
    obj_seq = track_sequence(obj_seq, ocfg, POSE_FPS)
    objs_at = {t: os_ for t, os_ in obj_seq}
    frames = [Frame(t, [Person(p["id"], p["box"], p["kp"], p["sc"], p.get("score", 1.0)) for p in ps],
                    [Obj(o["cls"], o["box"], o["score"], o.get("id")) for o in objs_at.get(t, [])])
              for t, ps in seq]
    scene = Scene(frames, zones, w, h, POSE_FPS, KP_T)
    crop_px = [crop_zone[0] * w / 1000, crop_zone[1] * h / 1000, crop_zone[2] * w / 1000, crop_zone[3] * h / 1000] if crop_zone else None
    roles = rule_engine.assign_roles(scene, cfg, crop_px)
    for fr in scene.frames:
        for p in fr.persons:
            p.role = roles.get(p.id, rule_engine.UNKNOWN)
    return scene, roles


def ui_tracks(scene: Scene) -> list[dict[str, Any]]:
    w, h = scene.w, scene.h
    n = lambda b: [round(b[0] * 1000 / w), round(b[1] * 1000 / h), round(b[2] * 1000 / w), round(b[3] * 1000 / h)]  # noqa: E731
    return [{"t": fr.t,
             "p": [{"id": p.id, "r": p.role.lower(), "b": n(p.box)} for p in fr.persons],
             **({"o": [{"c": o.cls, "id": o.id, "b": n(o.box), "s": round(o.score, 2)} for o in fr.objects]} if fr.objects else {})}
            for fr in scene.frames]


# ---------------- главный вход ----------------
def analyze(video, zone: list[float] | None = None, zones: Zones | None = None) -> dict[str, Any]:
    """video: pipeline.Video; zone — рамка крупного плана (0..1000); zones — зоны кассы."""
    zones = zones or Zones()
    times = [i / POSE_FPS for i in range(int(video.dur * POSE_FPS) + 1)]
    h = w = None
    seq, obj_seq = [], []
    for t, fr in video.iter(times):          # потоком, без хранения кадров
        h, w = fr.shape[:2]
        objs = detect_objects(fr)
        seq.append((t, detect(fr, objs)))
        obj_seq.append((t, [{"cls": o.cls, "box": o.box, "score": o.score} for o in objs if o.cls != "person"]))
    if not seq:
        return {"tracks": [], "candidates": [], "rule_events": []}
    scene, roles = build_scene(seq, obj_seq, zones, w, h, zone)
    events = rule_engine.run(scene, settings.rules_config())
    log.info("Следопыт: %d кадров, треков %d (кассир: %s), предметов %d, событий правил %d",
             len(seq), len(roles), [i for i, r in roles.items() if r == rule_engine.CASHIER],
             sum(len(f.objects) for f in scene.frames), len(events))
    return {"tracks": ui_tracks(scene), "candidates": [e.to_candidate() for e in events],
            "rule_events": [e.to_dict() for e in events], "roles": {str(k): v for k, v in roles.items()},
            "scene": scene}
