"""Общие структуры Rule Engine: сцена (люди, предметы, зоны по времени) и результат правила."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from zones import Zones

CASHIER, CLIENT, UNKNOWN = "CASHIER", "CLIENT", "UNKNOWN"
# индексы COCO-17 (в Wholebody первые 17 — те же)
NOSE, LSH, RSH, LWR, RWR, LHIP, RHIP = 0, 5, 6, 9, 10, 11, 12


@dataclass
class Person:
    id: int
    box: list[float]
    kp: np.ndarray          # (17, 2) пиксели
    sc: np.ndarray          # (17,)
    score: float = 1.0
    role: str = UNKNOWN

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class Obj:
    cls: str                # phone | small_object
    box: list[float]
    score: float
    id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class Frame:
    t: float
    persons: list[Person] = field(default_factory=list)
    objects: list[Obj] = field(default_factory=list)

    def by_role(self, *roles: str) -> list[Person]:
        return [p for p in self.persons if p.role in roles]


@dataclass
class Scene:
    frames: list[Frame]
    zones: Zones
    w: int
    h: int
    fps: float
    kp_thr: float = 0.35

    @property
    def times(self) -> list[float]:
        return [f.t for f in self.frames]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps if self.fps else 0.2

    def in_zone(self, name: str, xy: tuple[float, float]) -> bool:
        return self.zones.contains(name, xy[0], xy[1], self.w, self.h)

    def ok(self, p: Person, *idx: int) -> bool:
        return all(p.sc[i] > self.kp_thr for i in idx)

    def shoulder(self, p: Person) -> float:
        """Ширина плеч в пикселях — масштаб человека (не зависит от разрешения и ракурса)."""
        if self.ok(p, LSH, RSH):
            d = float(np.linalg.norm(p.kp[LSH] - p.kp[RSH]))
            if d > 1:
                return d
        return max((p.box[2] - p.box[0]) * 0.4, 1.0)

    def wrists(self, p: Person) -> list[np.ndarray]:
        return [p.kp[i] for i in (LWR, RWR) if self.ok(p, i)]


@dataclass
class RuleResult:
    """То, что возвращает каждое правило (формат из ТЗ Rule Engine)."""
    violation_type: str                 # имя правила, например transfer_of_item
    start_timecode_sec: float
    end_timecode_sec: float
    involved_track_ids: list[int]
    evidence_frame_timestamps: list[float]
    preliminary_confidence: float
    raw_evidence: dict[str, Any]
    maps_to: str = "OTHER"              # тип из configs/violations.yaml
    requires_vlm: bool = True
    description: str = ""               # подсказка по-русски для перепроверки Qwen
    key_time_sec: float | None = None   # момент для кандидата (по умолчанию — середина)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_candidate(self) -> dict[str, Any]:
        t = self.key_time_sec if self.key_time_sec is not None else (self.start_timecode_sec + self.end_timecode_sec) / 2
        return {"t": round(t, 2), "s": round(self.preliminary_confidence, 3), "type_id": self.maps_to, "src": "rule",
                "rule": self.violation_type, "obs": self.description, "requires_vlm": self.requires_vlm,
                "rule_event": self.to_dict()}


def segments(flags: list[bool], times: list[float], min_len: float, dt: float, max_gap: float = 0.0) -> list[tuple[float, float]]:
    """[(t_start, t_end)] подряд идущих True длительностью >= min_len (длительность = end - start + dt).
    Разрывы короче max_gap секунд склеиваются (детектор иногда теряет руку на кадр)."""
    segs: list[tuple[float, float]] = []
    start = prev = None
    for f, t in zip(flags, times):
        if f:
            if start is None:
                start = t
            elif prev is not None and t - prev > max_gap + dt + 1e-6:
                if prev - start + dt >= min_len - 1e-6:
                    segs.append((start, prev))
                start = t
            prev = t
    if start is not None and prev is not None and prev - start + dt >= min_len - 1e-6:
        segs.append((start, prev))
    return segs


def evidence_times(s: float, e: float) -> list[float]:
    return sorted({round(s, 2), round((s + e) / 2, 2), round(e, 2)})


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(расстояние от p до отрезка ab, параметр проекции u ∈ [0,1], 0 — у a)."""
    ab = b - a
    L = float(ab @ ab)
    if L < 1e-9:
        return float(np.linalg.norm(p - a)), 0.0
    u = float(np.clip((p - a) @ ab / L, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab))), u
