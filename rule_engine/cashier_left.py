"""cashier_left_workspace: кассир покинул рабочее место, зона кассира осталась без сотрудника.

>= min_absence_sec — детерминированное нарушение (Qwen может только отклонить);
soft_absence_sec..min_absence_sec — кандидат для проверки Qwen.
"""
from __future__ import annotations

from typing import Any

from .base import CASHIER, RuleResult, Scene, segments
from .phone_use import client_near, client_times

NAME = "cashier_left_workspace"


def run(scene: Scene, cfg: dict[str, Any]) -> list[RuleResult]:
    zone = scene.zones.has("cashier_zone")
    occupied = []
    for fr in scene.frames:
        cs = fr.by_role(CASHIER)
        occupied.append(any(scene.in_zone("cashier_zone", c.center) for c in cs) if zone else bool(cs))
    if not any(occupied):
        return []
    times = scene.times
    first = times[occupied.index(True)]
    gone = [(not o) and t > first for o, t in zip(occupied, times)]
    hard = float(cfg.get("min_absence_sec", 10.0))
    soft = float(cfg.get("soft_absence_sec", 3.0))
    ct = client_times(scene)
    window = float(cfg.get("client_window_sec", 5.0))
    out = []
    for s, e in segments(gone, times, soft, scene.dt, max_gap=scene.dt):
        dur = e - s + scene.dt
        with_client = client_near(ct, s, window)
        deterministic = dur >= hard and (with_client or not cfg.get("require_client", True))
        conf = float(cfg.get("confidence", 0.75)) if deterministic else float(cfg.get("soft_confidence", 0.5))
        requires_vlm = bool(cfg.get("requires_vlm", False)) if deterministic else True
        last_seen = max((t for t, o in zip(times, occupied) if o and t < s), default=s)
        desc = (f"по трекингу: кассира нет на рабочем месте с {s:.1f} до {e:.1f} с ({dur:.0f} с)"
                + (", клиент у кассы" if with_client else ""))
        # без клиента сам уход не нарушение — проверяем, забрал ли кассир ключ от шкафчика (п. 80)
        maps = cfg.get("maps_to", "CASHIER_LEFT_DURING_SERVICE") if with_client else cfg.get("maps_to_no_client", "KEY_LEFT_UNATTENDED")
        if not with_client:
            desc += "; проверь, запер ли кассир шкафчик и забрал ли ключ"
        out.append(RuleResult(
            NAME, round(s, 2), round(e, 2), [], sorted({round(last_seen, 2), round(s + 0.5, 2), round(min(s + hard, e), 2)}),
            round(conf, 3), {"absence_sec": round(dur, 1), "client_present": with_client, "zone_based": zone,
                             "until_video_end": e >= times[-1] - scene.dt},
            maps_to=maps, requires_vlm=requires_vlm,
            description=desc, key_time_sec=round(max(s - 1.0, first), 2)))   # момент перед уходом: ключ / шкафчик
    return out
