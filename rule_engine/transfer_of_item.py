"""transfer_of_item: передача предмета от кассира клиенту.

Условия: кассир и клиент рядом; руки обоих в transfer_zone; (по желанию) small_object между руками и
движется от кассира к клиенту; длительность >= min_duration_sec. Само по себе это обычная операция —
правило только находит момент, тип и факт нарушения подтверждает Qwen (requires_vlm).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import CASHIER, CLIENT, UNKNOWN, RuleResult, Scene, evidence_times, point_segment, segments

NAME = "transfer_of_item"


def run(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    near_k = float(cfg.get("near_factor", 1.0))
    between_k = float(cfg.get("object_between_factor", 0.6))
    use_zone = scene.zones.has("transfer_zone")
    flags: list[bool] = []
    info: list[dict[str, Any] | None] = []
    for fr in scene.frames:
        best = None
        for c in fr.by_role(CASHIER):
            sw = scene.shoulder(c)
            for o in fr.by_role(CLIENT, UNKNOWN):
                for a in scene.wrists(c):
                    for b in scene.wrists(o):
                        d = float(np.linalg.norm(a - b))
                        if d > near_k * sw:
                            continue
                        if use_zone and not (scene.in_zone("transfer_zone", a) and scene.in_zone("transfer_zone", b)):
                            continue
                        obj = None
                        for ob in fr.objects:
                            if ob.cls != "small_object":
                                continue
                            dist, u = point_segment(np.array(ob.center), a, b)
                            if dist <= between_k * sw:
                                obj = (ob.id, u, ob.score)
                                break
                        if best is None or d < best["d"]:
                            best = {"d": d, "cashier": c.id, "client": o.id, "obj": obj, "sw": sw}
        flags.append(best is not None)
        info.append(best)

    out: list[RuleResult] = []
    times = scene.times
    for s, e in segments(flags, times, float(cfg.get("min_duration_sec", 0.5)), scene.dt, max_gap=scene.dt):
        rows = [i for i, t in zip(info, times) if i and s <= t <= e]
        objs = [r["obj"] for r in rows if r["obj"] is not None]
        progress, obj_id = None, None
        if objs:
            by_id: dict[Any, list[float]] = {}
            for oid, u, _ in objs:
                by_id.setdefault(oid, []).append(u)
            obj_id, us = max(by_id.items(), key=lambda kv: len(kv[1]))
            progress = us[-1] - us[0] if len(us) >= 2 else 0.0
        if cfg.get("require_object") and not objs:
            continue
        moving = progress is not None and progress >= float(cfg.get("min_progress", 0.15))
        conf = float(cfg.get("confidence_with_object", 0.55)) if moving else float(cfg.get("confidence", 0.35))
        if objs and not moving:
            conf = (conf + float(cfg.get("confidence_with_object", 0.55))) / 2
        ids = sorted({r["cashier"] for r in rows} | {r["client"] for r in rows})
        desc = f"по трекингу: с {s:.1f} до {e:.1f} с руки кассира и клиента сходятся" + (" в зоне передачи" if use_zone else "")
        if objs:
            desc += ", между руками предмет" + (" движется от кассира к клиенту" if moving else "")
        desc += " — проверь, что передаётся (деньги, карта, документ) и пересчитаны ли деньги при клиенте"
        out.append(RuleResult(
            NAME, round(s, 2), round(e, 2), ids, evidence_times(s, e), round(conf, 3),
            {"frames": len(rows), "in_transfer_zone": use_zone, "object_track_id": obj_id,
             "object_progress": None if progress is None else round(progress, 3),
             "mean_hand_distance_sw": round(float(np.mean([r["d"] / r["sw"] for r in rows])), 3) if rows else None},
            maps_to=cfg.get("maps_to", "PAYOUT_NO_RECOUNT"), requires_vlm=bool(cfg.get("requires_vlm", True)),
            description=desc))
    return out
