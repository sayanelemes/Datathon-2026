"""Правила по рукам кассира (перенесены из pose.py без изменения логики): только подсказки для Qwen.

hand_below_desk — одна рука опущена ниже линии бёдер (под стол), другая у стола (возможное сокрытие купюр);
hand_near_head  — запястье у лица >= 2 с (возможно, телефон, даже если детектор его не видит).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import CASHIER, LHIP, LWR, NOSE, RHIP, RWR, RuleResult, Scene, evidence_times, segments

BELOW, HEAD = "hand_below_desk", "hand_near_head"


def _handover_times(scene: Scene) -> list[float]:
    out = []
    for fr in scene.frames:
        for c in fr.by_role(CASHIER):
            sw = scene.shoulder(c)
            if any(np.linalg.norm(a - b) < 0.8 * sw for o in fr.persons if o.id != c.id
                   for a in scene.wrists(c) for b in scene.wrists(o)):
                out.append(fr.t)
                break
    return out


def run_below(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    margin = float(cfg.get("low_margin", 0.15))
    flags, ids = [], []
    for fr in scene.frames:
        hit = None
        for c in fr.by_role(CASHIER):
            sw = scene.shoulder(c)
            hips = [c.kp[j][1] for j in (LHIP, RHIP) if c.sc[j] > scene.kp_thr]
            hip_y = float(np.mean(hips)) if hips else c.box[3] - 0.25 * (c.box[3] - c.box[1])
            lw_low = scene.ok(c, LWR) and c.kp[LWR][1] > hip_y + margin * sw
            rw_low = scene.ok(c, RWR) and c.kp[RWR][1] > hip_y + margin * sw
            lw_up = scene.ok(c, LWR) and not lw_low
            rw_up = scene.ok(c, RWR) and not rw_low
            if (lw_low and rw_up) or (rw_low and lw_up):
                hit = c.id
                break
        flags.append(hit is not None)
        ids.append(hit)
    ho = _handover_times(scene)
    after = float(cfg.get("after_transfer_sec", 10.0))
    out = []
    for s, e in segments(flags, scene.times, float(cfg.get("min_duration_sec", 1.0)), scene.dt):
        after_ho = any(0 <= s - h <= after for h in ho)
        conf = float(cfg.get("confidence_after_transfer", 0.55)) if after_ho else float(cfg.get("confidence", 0.4))
        out.append(RuleResult(
            BELOW, round(s, 2), round(e, 2), sorted({i for i, t in zip(ids, scene.times) if i and s <= t <= e}),
            evidence_times(s, e), conf, {"after_transfer": after_ho},
            maps_to=cfg.get("maps_to", "CASH_CONCEALMENT"), requires_vlm=True,
            description=f"по позе: с {s:.1f} до {e:.1f} с одна рука кассира опущена ниже уровня стола, другая у стола"
                        + (" (вскоре после передачи денег/предметов)" if after_ho else "")))
    return out


def run_head(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    k = float(cfg.get("near_factor", 0.6))
    flags, ids = [], []
    for fr in scene.frames:
        hit = None
        for c in fr.by_role(CASHIER):
            if scene.ok(c, NOSE) and any(np.linalg.norm(w - c.kp[NOSE]) < k * scene.shoulder(c) for w in scene.wrists(c)):
                hit = c.id
                break
        flags.append(hit is not None)
        ids.append(hit)
    out = []
    for s, e in segments(flags, scene.times, float(cfg.get("min_duration_sec", 2.0)), scene.dt):
        out.append(RuleResult(
            HEAD, round(s, 2), round(e, 2), sorted({i for i, t in zip(ids, scene.times) if i and s <= t <= e}),
            evidence_times(s, e), float(cfg.get("confidence", 0.45)), {},
            maps_to=cfg.get("maps_to", "PHONE_AT_CASH_DESK"), requires_vlm=True,
            description=f"по позе: с {s:.1f} до {e:.1f} с рука кассира у головы (возможно, телефон)"))
    return out
