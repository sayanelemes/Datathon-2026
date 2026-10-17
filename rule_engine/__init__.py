"""Rule Engine: роли (CASHIER / CLIENT / UNKNOWN) и правила-модули с настройками из configs/rules.yaml.

Каждое правило — функция run(scene, cfg) -> list[RuleResult]. Чтобы добавить правило: модуль с run(),
запись в RULES ниже и секция в configs/rules.yaml.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from . import cashier_left, hand_rules, phone_use, restricted_zone, transfer_of_item
from .base import CASHIER, CLIENT, UNKNOWN, Frame, Obj, Person, RuleResult, Scene, segments  # noqa: F401

log = logging.getLogger("rules")

RULES: dict[str, Callable[[Scene, dict[str, Any]], list[RuleResult]]] = {
    transfer_of_item.NAME: transfer_of_item.run,
    phone_use.NAME: phone_use.run,
    cashier_left.NAME: cashier_left.run,
    restricted_zone.UNAUTHORIZED: restricted_zone.run_unauthorized,
    restricted_zone.CLIENT_IN: restricted_zone.run_client,
    hand_rules.BELOW: hand_rules.run_below,
    hand_rules.HEAD: hand_rules.run_head,
}


def assign_roles(scene: Scene, cfg: dict[str, Any], crop_zone_px: list[float] | None = None) -> dict[int, str]:
    """CASHIER — дольше всех в cashier_zone; CLIENT — со стороны клиента; остальные — UNKNOWN.

    Кассир, вернувшийся с новым track_id, тоже CASHIER: трек в cashier_zone, не пересекающийся по времени
    с главным кассиром. Без зон — прежняя эвристика: дольше всех в кадре и меньше всех двигается.
    """
    rc = cfg.get("roles") or {}
    tracks: dict[int, list[tuple[float, tuple[float, float]]]] = {}
    for fr in scene.frames:
        for p in fr.persons:
            tracks.setdefault(p.id, []).append((fr.t, p.center))
    if not tracks:
        return {}
    dt = scene.dt
    min_track = float(rc.get("min_track_sec", 1.0))
    dur = {i: len(v) * dt for i, v in tracks.items()}
    roles: dict[int, str] = {}

    def share(i: int, zone: str) -> float:
        pts = tracks[i]
        return sum(scene.in_zone(zone, c) for _, c in pts) / max(len(pts), 1)

    main = None
    if scene.zones.has("cashier_zone"):
        in_zone = {i: share(i, "cashier_zone") * dur[i] for i in tracks}
        cands = [i for i in tracks if share(i, "cashier_zone") >= float(rc.get("cashier_min_zone_share", 0.3))]
        if cands:
            main = max(cands, key=lambda i: in_zone[i])
    if main is None:                                   # прежняя эвристика (без зон)
        best_s = -1.0
        for i, pts in tracks.items():
            a = np.array([c for _, c in pts])
            move = float(np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1))) if len(a) > 1 else 0.0
            s = len(a) / (1 + move / 10)
            if crop_zone_px:
                z = crop_zone_px
                s *= 0.5 + float(np.mean([(z[0] <= x <= z[2] and z[1] <= y <= z[3]) for x, y in a]))
            if s > best_s:
                main, best_s = i, s
    roles[main] = CASHIER
    spans = [(tracks[main][0][0], tracks[main][-1][0])]
    home = np.median(np.array([c for _, c in tracks[main]]), axis=0)          # где обычно сидит кассир
    widths = [fr_p.box[2] - fr_p.box[0] for fr in scene.frames for fr_p in fr.persons if fr_p.id == main]
    home_r = float(rc.get("cashier_home_radius", 0.6)) * (float(np.median(widths)) if widths else 100.0)
    for i in sorted(tracks, key=lambda k: tracks[k][0][0]):          # вернувшийся кассир (новый track_id)
        if i == main or dur[i] < min_track or not scene.zones.has("cashier_zone"):
            continue
        a, b = tracks[i][0][0], tracks[i][-1][0]
        start = np.array(tracks[i][0][1])
        from_client = scene.zones.has("client_zone") and share(i, "client_zone") > 0
        if (share(i, "cashier_zone") >= 0.5 and not from_client and float(np.linalg.norm(start - home)) <= home_r
                and all(b < s0 - 0.5 or a > e0 + 0.5 for s0, e0 in spans)):
            roles[i] = CASHIER                     # появился на месте кассира, не со стороны клиента
            spans.append((a, b))
    for i in tracks:
        if i in roles:
            continue
        if dur[i] < min_track:
            roles[i] = UNKNOWN
        elif scene.zones.has("client_zone"):
            roles[i] = CLIENT if share(i, "client_zone") >= float(rc.get("client_min_zone_share", 0.5)) else UNKNOWN
        else:
            roles[i] = CLIENT
    return roles


def run(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    out: list[RuleResult] = []
    for name, fn in RULES.items():
        rcfg = (cfg.get("rules") or {}).get(name) or {}
        if not rcfg.get("enabled", True):
            continue
        try:
            res = sorted(fn(scene, rcfg), key=lambda r: -r.preliminary_confidence)
        except Exception:  # noqa: BLE001 — одно сломанное правило не должно ронять анализ
            log.exception("правило %s упало", name)
            continue
        out += res[: int(rcfg.get("max_candidates", 3))]
    out.sort(key=lambda r: -r.preliminary_confidence)
    total = int((cfg.get("engine") or {}).get("max_candidates_total", 8))
    det = [r for r in out if not r.requires_vlm]          # детерминированные не теряем при обрезке
    rest = [r for r in out if r.requires_vlm][: max(total - len(det), 0)]
    res = sorted(det + rest, key=lambda r: r.start_timecode_sec)
    for r in res:
        log.info("правило %s: %.1f–%.1f с, conf %.2f, VLM %s", r.violation_type, r.start_timecode_sec,
                 r.end_timecode_sec, r.preliminary_confidence, "нужен" if r.requires_vlm else "только veto")
    return res
