"""phone_use_during_service: кассир пользуется телефоном при клиенте.

Условия: кассир в cashier_zone; клиент присутствует; телефон (YOLOX COCO «cell phone») рядом с рукой
или лицом кассира; длительность >= min_duration_sec. Подтверждает Qwen.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import CASHIER, CLIENT, NOSE, RuleResult, Scene, evidence_times, segments

NAME = "phone_use_during_service"


def client_times(scene: Scene) -> list[float]:
    return [fr.t for fr in scene.frames if fr.by_role(CLIENT)]


def client_near(ct: list[float], t: float, window: float) -> bool:
    return any(abs(t - x) <= window for x in ct)


def run(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    near_k = float(cfg.get("near_factor", 0.8))
    zone = scene.zones.has("cashier_zone")
    flags, info = [], []
    for fr in scene.frames:
        hit = None
        for c in fr.by_role(CASHIER):
            if zone and not scene.in_zone("cashier_zone", c.center):
                continue
            sw = scene.shoulder(c)
            pts = scene.wrists(c) + ([c.kp[NOSE]] if scene.ok(c, NOSE) else [])
            for ob in fr.objects:
                if ob.cls != "phone":
                    continue
                oc = np.array(ob.center)
                if pts:
                    d = min(float(np.linalg.norm(oc - p)) for p in pts)
                    near = d <= near_k * sw
                else:                                   # руки не видны — телефон внутри рамки кассира
                    d, near = None, c.box[0] <= oc[0] <= c.box[2] and c.box[1] <= oc[1] <= c.box[3]
                if near:
                    hit = {"cashier": c.id, "phone": ob.id, "score": ob.score, "d_sw": None if d is None else d / sw}
                    break
            if hit:
                break
        flags.append(hit is not None)
        info.append(hit)

    ct = client_times(scene)
    window = float(cfg.get("client_window_sec", 5.0))
    out = []
    for s, e in segments(flags, scene.times, float(cfg.get("min_duration_sec", 2.0)), scene.dt, max_gap=2 * scene.dt):
        rows = [i for i, t in zip(info, scene.times) if i and s <= t <= e]
        with_client = client_near(ct, s, window) or client_near(ct, e, window)
        conf = float(cfg.get("confidence", 0.6))
        if cfg.get("require_client", True) and not with_client:
            conf *= 0.7                                  # без клиента — слабее, но не теряем (полнота)
        desc = (f"по детектору: с {s:.1f} до {e:.1f} с у руки/лица кассира телефон"
                + (", клиент у кассы" if with_client else ", клиента рядом не видно"))
        out.append(RuleResult(
            NAME, round(s, 2), round(e, 2), sorted({r["cashier"] for r in rows}), evidence_times(s, e), round(conf, 3),
            {"frames": len(rows), "client_present": with_client, "cashier_in_zone": zone,
             "phone_track_ids": sorted({r["phone"] for r in rows if r["phone"] is not None}),
             "mean_phone_score": round(float(np.mean([r["score"] for r in rows])), 3) if rows else None},
            maps_to=cfg.get("maps_to", "PHONE_AT_CASH_DESK"), requires_vlm=bool(cfg.get("requires_vlm", True)),
            description=desc))
    return out
