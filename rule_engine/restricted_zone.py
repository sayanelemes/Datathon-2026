"""Служебная зона.

unauthorized_person_in_restricted_zone — любой человек не с ролью CASHIER в restricted_zone >= 3 с.
client_in_restricted_zone             — клиент пересёк стойку (был в client_zone) и находится в restricted_zone >= 2 с.
"""
from __future__ import annotations

from typing import Any

from .base import CASHIER, CLIENT, RuleResult, Scene, evidence_times, segments

UNAUTHORIZED = "unauthorized_person_in_restricted_zone"
CLIENT_IN = "client_in_restricted_zone"


def _per_track(scene: Scene) -> dict[int, list[tuple[float, Any]]]:
    tracks: dict[int, list[tuple[float, Any]]] = {}
    for fr in scene.frames:
        for p in fr.persons:
            tracks.setdefault(p.id, []).append((fr.t, p))
    return tracks


def run_unauthorized(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    if not scene.zones.has("restricted_zone"):
        return []
    out = []
    for tid, rows in _per_track(scene).items():
        if rows[0][1].role == CASHIER:
            continue
        times = [t for t, _ in rows]
        flags = [scene.in_zone("restricted_zone", p.center) for _, p in rows]
        for s, e in segments(flags, times, float(cfg.get("min_duration_sec", 3.0)), scene.dt, max_gap=scene.dt):
            role = rows[0][1].role
            out.append(RuleResult(
                UNAUTHORIZED, round(s, 2), round(e, 2), [tid], evidence_times(s, e), float(cfg.get("confidence", 0.65)),
                {"role": role, "seconds_in_zone": round(e - s + scene.dt, 1)},
                maps_to=cfg.get("maps_to", "UNAUTHORIZED_PERSON"), requires_vlm=bool(cfg.get("requires_vlm", False)),
                description=f"по трекингу: человек #{tid} (не кассир) в служебной зоне с {s:.1f} до {e:.1f} с"))
    return out


def run_client(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    if not scene.zones.has("restricted_zone"):
        return []
    has_client_zone = scene.zones.has("client_zone")
    out = []
    for tid, rows in _per_track(scene).items():
        if rows[0][1].role != CLIENT:
            continue
        times = [t for t, _ in rows]
        inside = [scene.in_zone("restricted_zone", p.center) for _, p in rows]
        before = [scene.in_zone("client_zone", p.center) if has_client_zone else not f for (_, p), f in zip(rows, inside)]
        for s, e in segments(inside, times, float(cfg.get("min_duration_sec", 2.0)), scene.dt, max_gap=scene.dt):
            crossed = any(b for t, b in zip(times, before) if t < s)
            if not crossed:
                continue
            out.append(RuleResult(
                CLIENT_IN, round(s, 2), round(e, 2), [tid], evidence_times(s, e), float(cfg.get("confidence", 0.7)),
                {"crossed_from": "client_zone" if has_client_zone else "outside", "seconds_in_zone": round(e - s + scene.dt, 1)},
                maps_to=cfg.get("maps_to", "UNAUTHORIZED_PERSON"), requires_vlm=bool(cfg.get("requires_vlm", False)),
                description=f"по трекингу: клиент #{tid} зашёл за стойку в служебную зону с {s:.1f} до {e:.1f} с"))
    return out
