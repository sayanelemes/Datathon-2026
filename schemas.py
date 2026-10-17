"""Pydantic-схемы: ответы Qwen (валидация) и API (строго по ТЗ и расширенный формат)."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ObjectType = Literal["banknote", "card", "phone", "paper", "unknown"]
_OBJ_ALIASES = {
    "banknote": "banknote", "banknotes": "banknote", "cash": "banknote", "money": "banknote", "bill": "banknote",
    "купюра": "banknote", "купюры": "banknote", "деньги": "banknote", "наличные": "banknote",
    "card": "card", "карта": "card", "карточка": "card", "bank card": "card",
    "phone": "phone", "smartphone": "phone", "mobile": "phone", "телефон": "phone", "смартфон": "phone",
    "paper": "paper", "document": "paper", "documents": "paper", "документ": "paper", "бумага": "paper", "чек": "paper",
}


def to_confidence(x: Any, default: float = 0.0) -> float:
    """0.85 / 85 / "85%" / "0,85" / "high" → 0..1."""
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        v = float(x)
    elif isinstance(x, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", x)
        if m:
            v = float(m.group(0).replace(",", "."))
            if "%" in x:
                v /= 100
        else:
            return {"high": 0.85, "высокая": 0.85, "medium": 0.5, "средняя": 0.5,
                    "low": 0.15, "низкая": 0.15}.get(x.strip().lower(), default)
    else:
        return default
    if v > 1:
        v /= 100
    return min(max(v, 0.0), 1.0)


def to_int(x: Any, default: int = 0) -> int:
    if isinstance(x, bool) or x is None:
        return default
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"-?\d+", str(x))
    return int(m.group(0)) if m else default


class QwenVerdict(BaseModel):
    """Ответ Qwen на скан/перепроверку. Терпит синонимы ключей и форматы чисел, но результат всегда валиден."""
    model_config = ConfigDict(extra="ignore")

    violation: bool = False
    type_id: str = "NONE"
    object_type: ObjectType = "unknown"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    observation: str = ""                      # = description из ТЗ
    frame_index: int = 0
    evidence_frames: list[int] = Field(default_factory=list)
    bbox: Optional[list[float]] = None

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, a: Any) -> Any:
        if not isinstance(a, dict):
            raise ValueError("ответ модели не JSON-объект")
        a = dict(a)
        if "type_id" not in a and "type" in a:
            a["type_id"] = a["type"]
        if not a.get("observation") and a.get("description"):
            a["observation"] = a["description"]
        if "violation" not in a:
            if "confidence" not in a:
                raise ValueError("нет полей violation и confidence")
            a["violation"] = to_confidence(a["confidence"]) >= 0.5
        return a

    @field_validator("violation", mode="before")
    @classmethod
    def _v(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "да", "1")
        return bool(v)

    @field_validator("type_id", mode="before")
    @classmethod
    def _t(cls, v: Any) -> str:
        v = str(v or "NONE").strip().upper()
        return "NONE" if v in ("", "NULL", "NONE", "N/A") else v

    @field_validator("object_type", mode="before")
    @classmethod
    def _o(cls, v: Any) -> str:
        return _OBJ_ALIASES.get(str(v or "").strip().lower(), "unknown")

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v: Any) -> float:
        return to_confidence(v)

    @field_validator("observation", mode="before")
    @classmethod
    def _obs(cls, v: Any) -> str:
        return str(v or "").strip()[:600]

    @field_validator("frame_index", mode="before")
    @classmethod
    def _fi(cls, v: Any) -> int:
        return max(to_int(v), 0)

    @field_validator("evidence_frames", mode="before")
    @classmethod
    def _ef(cls, v: Any) -> list[int]:
        if not isinstance(v, (list, tuple)):
            return []
        out = []
        for x in v:
            if isinstance(x, dict):
                x = x.get("frame_index", x.get("index"))
            i = to_int(x, -1)
            if i >= 0:
                out.append(i)
        return out[:9]

    @field_validator("bbox", mode="before")
    @classmethod
    def _bb(cls, v: Any) -> Optional[list[float]]:
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            return None
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None


def validate_verdict(a: Any, known_types: set[str] | None = None) -> Optional[dict[str, Any]]:
    """dict от модели → нормализованный dict или None (невалидный ответ)."""
    try:
        v = QwenVerdict.model_validate(a)
    except ValidationError:
        return None
    d = v.model_dump()
    if known_types is not None and d["type_id"] not in known_types and not d["violation"]:
        d["type_id"] = "NONE"
    return d


class LegalChoice(BaseModel):
    """Ответ Qwen при выборе нормы из найденных фрагментов."""
    model_config = ConfigDict(extra="ignore")
    choice: Optional[int] = None               # номер фрагмента 1..N или null
    confidence: Literal["high", "medium", "low"] = "low"
    relevance: str = ""
    quote: str = ""

    @field_validator("choice", mode="before")
    @classmethod
    def _ch(cls, v: Any) -> Optional[int]:
        if v in (None, "", "null", "none", 0, "0"):
            return None
        i = to_int(v, -1)
        return i if i > 0 else None

    @field_validator("confidence", mode="before")
    @classmethod
    def _cf(cls, v: Any) -> str:
        s = str(v or "low").strip().lower()
        if s in ("high", "medium", "low"):
            return s
        c = to_confidence(v, 0.0)
        return "high" if c >= 0.75 else "medium" if c >= 0.45 else "low"


# ---------------- API ----------------
class AnalyzeAccepted(BaseModel):
    job_id: str
    status: Literal["processing"] = "processing"


class EvidenceFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_sec: float
    image_base64: str


class LegalReference(BaseModel):
    source: str
    document: str
    article: str
    title: str = ""
    quote: str
    url: Optional[str] = None
    relevance: str = ""
    confidence: Literal["high", "medium", "low"]


class ViolationStrict(BaseModel):
    """Строго поля ТЗ (раздел 10) — именно их проверяет жюри."""
    model_config = ConfigDict(extra="forbid")
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)


class ViolationExtended(ViolationStrict):
    confidence: float
    legal_reference: Optional[LegalReference] = None
    legal_confidence: Literal["high", "medium", "low"] = "low"
    object_type: ObjectType = "unknown"
    verification: str = "confirmed"


class JobStrict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[ViolationStrict]


class JobExtended(BaseModel):
    job_id: str
    status: Literal["done"] = "done"
    verdict: Literal["violation", "no_violation"]
    violations: list[ViolationExtended]
    annotated_video_url: Optional[str] = None


class JobPending(BaseModel):
    job_id: str
    status: Literal["processing", "failed"]
    error: Optional[str] = None
