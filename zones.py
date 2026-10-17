"""Зоны кассы: cashier_zone, client_zone, transfer_zone, restricted_zone.

Хранятся как многоугольники в нормированных координатах (0..1). Источник — configs/camera_*.yaml
и/или рамки, которые Qwen находит на каждом видео (финальная камера снята под другим углом).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger("zones")

ZONE_NAMES = ("cashier_zone", "client_zone", "transfer_zone", "restricted_zone")
Point = tuple[float, float]


def point_in_poly(x: float, y: float, poly: list[Point]) -> bool:
    """Луч вправо (even-odd). poly — список вершин."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / ((yj - yi) or 1e-12)
            if x < xc:
                inside = not inside
        j = i
    return inside


def box_poly(b: Iterable[float]) -> list[Point]:
    x1, y1, x2, y2 = b
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _valid_box(b: Any) -> list[float] | None:
    """Рамка Qwen 0..1000 → 0..1 или None."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        b = [min(max(float(v), 0.0), 1000.0) / 1000.0 for v in b]
    except (TypeError, ValueError):
        return None
    if b[2] - b[0] < 0.03 or b[3] - b[1] < 0.03:
        return None
    return b


@dataclass
class Zones:
    polys: dict[str, list[Point]] = field(default_factory=dict)   # нормированные 0..1
    source: str = "none"

    def has(self, name: str) -> bool:
        return len(self.polys.get(name) or []) >= 3

    def contains(self, name: str, x: float, y: float, w: float = 1.0, h: float = 1.0) -> bool:
        """Точка в пикселях кадра w×h (или нормированная, если w=h=1)."""
        poly = self.polys.get(name)
        if not poly or len(poly) < 3:
            return False
        return point_in_poly(x / w, y / h, poly)

    def to_json(self) -> dict[str, Any]:
        return {"source": self.source, "zones": {k: [[round(x, 4), round(y, 4)] for x, y in v] for k, v in self.polys.items()}}

    @property
    def empty(self) -> bool:
        return not any(self.has(n) for n in ZONE_NAMES)


def from_yaml(cfg: dict[str, Any]) -> Zones:
    polys: dict[str, list[Point]] = {}
    for name, pts in (cfg.get("zones") or {}).items():
        if name not in ZONE_NAMES:
            log.warning("неизвестная зона %s в конфиге камеры — пропущена", name)
            continue
        try:
            poly = [(float(p[0]), float(p[1])) for p in pts]
        except (TypeError, ValueError, IndexError):
            log.warning("зона %s: неверный формат точек", name)
            continue
        if len(poly) >= 3 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in poly):
            polys[name] = poly
        else:
            log.warning("зона %s: нужно >=3 точек в диапазоне 0..1", name)
    return Zones(polys, "yaml" if polys else "none")


def from_qwen(boxes: dict[str, Any], cfg: dict[str, Any]) -> Zones:
    """boxes — ответ Qwen: {cashier, counter, client} в координатах 0..1000 кадра."""
    auto = cfg.get("auto") or {}
    polys: dict[str, list[Point]] = {}
    cashier = _valid_box(boxes.get("cashier"))
    counter = _valid_box(boxes.get("counter"))
    client = _valid_box(boxes.get("client"))
    if cashier:
        polys["cashier_zone"] = box_poly(cashier)
    if client:
        polys["client_zone"] = box_poly(client)
    if counter:
        m = float(auto.get("transfer_margin", 0.1))
        dx, dy = (counter[2] - counter[0]) * m, (counter[3] - counter[1]) * m
        polys["transfer_zone"] = box_poly([max(0, counter[0] - dx), max(0, counter[1] - dy),
                                           min(1, counter[2] + dx), min(1, counter[3] + dy)])
    if auto.get("restricted_from", "cashier") == "cashier" and cashier:
        polys["restricted_zone"] = list(polys["cashier_zone"])
    return Zones(polys, "qwen" if polys else "none")


def merge(primary: Zones, fallback: Zones) -> Zones:
    """Зоны из primary важнее; недостающие берутся из fallback."""
    polys = dict(fallback.polys)
    polys.update(primary.polys)
    src = "+".join(s for s in (primary.source, fallback.source) if s != "none") or "none"
    return Zones(polys, src)


def resolve(cfg: dict[str, Any], qwen_boxes: dict[str, Any] | None) -> Zones:
    mode = str(cfg.get("zones_mode", "auto")).lower()
    if mode == "off":
        return Zones()
    manual = from_yaml(cfg)
    if mode == "yaml":
        return manual
    auto = from_qwen(qwen_boxes or {}, cfg)
    return merge(manual, auto)
