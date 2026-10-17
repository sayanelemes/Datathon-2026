"""Legal-RAG: норма РК для нарушения.

1. По типу и описанию нарушения ищем top-3 фрагмента в legal_base/ (векторный индекс).
2. Пункт из каталога configs/violations.yaml (проверен людьми) тоже кладём в кандидаты, если он есть в базе.
3. Qwen выбирает номер фрагмента — ТОЛЬКО номер. Документ, пункт, URL берутся из метаданных фрагмента,
   цитата — дословно из текста фрагмента (цитату Qwen принимаем, только если она точно есть в тексте).
4. Нет уверенности → legal_reference = null, legal_confidence = low. Ничего не выдумывается.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Optional

from schemas import LegalChoice

from .index import VectorIndex, build_or_load
from .loader import Chunk

log = logging.getLogger("legal")

_index: VectorIndex | None = None
_lock = threading.Lock()
_cache: dict[str, tuple[Optional[dict[str, Any]], str]] = {}
TOP_K = 3

SELECT_PROMPT = (
    "Ты помогаешь юристу банка. Нарушение кассовой дисциплины:\n{violation}\n\n"
    "Фрагменты нормативных актов Республики Казахстан:\n{fragments}\n\n"
    "Выбери ОДИН фрагмент, норму которого нарушает описанное действие. Выбирать можно только из фрагментов выше. "
    "Не придумывай статьи, документы, цитаты и ссылки. Если ни один фрагмент не подходит — choice = null.\n"
    "Ответ строго JSON без markdown: "
    '{{"choice": <номер фрагмента или null>, "confidence": "high" | "medium" | "low", '
    '"relevance": "одно предложение по-русски: почему норма относится к нарушению", '
    '"quote": "точная фраза из выбранного фрагмента, без изменений"}}'
)


def get_index() -> VectorIndex:
    global _index
    with _lock:
        if _index is None:
            import settings
            _index = build_or_load(settings.LEGAL_BASE_DIR, settings.LEGAL_INDEX_DIR)
        return _index


def reset() -> None:
    global _index
    with _lock:
        _index = None
        _cache.clear()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("ё", "е").replace("«", '"').replace("»", '"')).strip().lower()


def verified_quote(chunk: Chunk, quote: str | None, limit: int = 400) -> str:
    """Цитата Qwen, если она дословно есть в тексте; иначе — начало пункта до границы предложения."""
    if quote and len(quote.strip()) >= 15 and _norm(quote) in _norm(chunk.text):
        return quote.strip()
    t = chunk.text.strip()
    if len(t) <= limit:
        return t
    cut = t[:limit]
    m = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[: m + 1] if m > 80 else cut.rstrip() + "…").strip()


def subpoint_text(chunk: Chunk, point: str | None) -> str | None:
    """«п. 31, пп. 4» → дословный текст подпункта «4) …» из пункта (если он есть)."""
    m = re.search(r"пп\.\s*(\d{1,2})", point or "")
    if not m:
        return None
    sm = re.search(rf"(?:^|\s){m.group(1)}\)\s+(.+?)(?=\s\d{{1,2}}\)\s|$)", chunk.text)
    return f"{m.group(1)}) {sm.group(1).strip()}" if sm else None


def catalog_article(point: str | None) -> str | None:
    """«п. 20 (см. также п. 14)» → «20»; «п. 31, пп. 4» → «31»."""
    m = re.search(r"п\.\s*(\d{1,3}(?:-\d{1,2})?)", point or "")
    return m.group(1) if m else None


def reference(chunk: Chunk, quote: str, relevance: str, conf: str) -> dict[str, Any]:
    return {"source": chunk.source, "document": chunk.document, "article": chunk.article, "title": chunk.title,
            "quote": quote, "url": chunk.url, "relevance": relevance, "confidence": conf}


def candidates(idx: VectorIndex, query: str, art: str | None) -> list[tuple[Chunk, float]]:
    found = idx.search(query, TOP_K)
    if art and not any(c.article_num == art for c, _ in found):
        cat = next((c for c in idx.chunks if c.article_num == art and c.part == 0), None)
        if cat is not None:
            found = found[: TOP_K - 1] + [(cat, 0.0)]
    return found


def lookup(type_id: str, rule: dict[str, Any], description: str,
           chat: Callable[[str], Optional[dict[str, Any]]] | None = None) -> tuple[Optional[dict[str, Any]], str]:
    """→ (legal_reference | None, legal_confidence). chat(prompt) → dict | None — текстовый запрос к Qwen."""
    key = f"{type_id}|{'qwen' if chat else 'noqwen'}"
    if key in _cache:
        return _cache[key]
    try:
        idx = get_index()
    except Exception:  # noqa: BLE001 — норма не должна ронять анализ
        log.exception("нормативный индекс недоступен")
        return None, "low"
    if not idx.chunks:
        return None, "low"
    art = catalog_article(rule.get("point"))
    query = f'{rule.get("name", "")}. {rule.get("visual", "")}. {rule.get("norm", "")}. {description}'
    found = candidates(idx, query, art)
    cat_i = next((i for i, (c, _) in enumerate(found) if art and c.article_num == art), None)

    choice: LegalChoice | None = None
    if chat is not None:
        frags = "\n".join(f"[{i + 1}] {c.document}, {c.article}: {c.text[:900]}" for i, (c, _) in enumerate(found))
        try:
            raw = chat(SELECT_PROMPT.format(violation=f'{rule.get("name", type_id)}. {description}'[:800], fragments=frags))
            choice = LegalChoice.model_validate(raw) if isinstance(raw, dict) else None
        except Exception as e:  # noqa: BLE001
            log.warning("выбор нормы Qwen не удался: %r", e)
            choice = None
        if choice is not None and choice.choice is not None and not 1 <= choice.choice <= len(found):
            choice = None                                    # номер вне списка — считаем, что ответа нет

    res: tuple[Optional[dict[str, Any]], str]
    q_i = choice.choice - 1 if choice is not None and choice.choice else None
    if cat_i is not None and q_i == cat_i:                   # каталог и Qwen согласны
        c = found[cat_i][0]
        quote = choice.quote if choice.quote and _norm(choice.quote) in _norm(c.text) else subpoint_text(c, rule.get("point"))
        res = reference(c, verified_quote(c, quote), choice.relevance or rule.get("norm", ""), "high"), "high"
    elif cat_i is not None:                                  # каталог проверен людьми — он главнее
        c = found[cat_i][0]
        # без Qwen: «high», если поиск сам поставил пункт каталога первым (два независимых источника согласны)
        conf = "high" if choice is None and cat_i == 0 and found[0][1] > 0 else "medium"
        res = reference(c, verified_quote(c, subpoint_text(c, rule.get("point"))), rule.get("norm", ""), conf), conf
    elif q_i is not None and choice.confidence != "low":     # в каталоге пункта нет, Qwen уверен
        c = found[q_i][0]
        res = reference(c, verified_quote(c, choice.quote), choice.relevance, choice.confidence), choice.confidence
    else:
        res = None, "low"
    _cache[key] = res
    return res
