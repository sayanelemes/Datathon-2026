"""Загрузка нормативных документов из legal_base/ (TXT, MD, PDF) и разбиение на статьи/пункты и чанки.

Метаданные документа (название, URL, источник) — legal_base/sources.yaml:
  nbrk_231.txt:
    document: "Правила осуществления кассовых операций ..."
    url: "https://adilet.zan.kz/rus/docs/V1900019680"
    source: "adilet.zan.kz"
Без записи в sources.yaml документом считается имя файла, URL не указывается (не выдумываем).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import yaml

log = logging.getLogger("legal")

EXT = (".txt", ".md", ".pdf")
CHUNK_CHARS = int(os.getenv("LEGAL_CHUNK_CHARS", 1200))

# «20. Текст пункта», «20-1. …», «Статья 12. …», «Глава 3. Название»
RE_POINT = re.compile(r"^\s*(\d{1,3}(?:-\d{1,2})?)\.\s+(?=\S)")
RE_ARTICLE = re.compile(r"^\s*Стать[яи]\s+(\d{1,4}(?:-\d{1,2})?)\.?\s*(.*)$", re.I)
RE_CHAPTER = re.compile(r"^\s*(Глава|Параграф|Раздел)\s+(\d+[-\d]*)\.?\s*(.*)$", re.I)


@dataclass
class Chunk:
    doc_file: str
    document: str
    source: str
    url: str | None
    article: str          # «п. 20» / «статья 12» / «фрагмент 3»
    article_num: str      # «20» / «12» / ""
    title: str            # глава/раздел, к которому относится пункт
    text: str             # дословный текст из документа
    part: int = 0         # номер чанка внутри длинного пункта

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        except ImportError:
            pass
        if shutil.which("pdftotext"):
            return subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, check=True).stdout
        log.warning("PDF %s пропущен: нет pypdf и pdftotext", path)
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_sources(base: str) -> dict[str, dict[str, Any]]:
    p = os.path.join(base, "sources.yaml")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    sents = re.split(r"(?<=[.;:])\s+", text)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def split_articles(text: str) -> list[tuple[str, str, str, str]]:
    """→ [(article_label, article_num, title, text)] без потери текста."""
    items: list[list[str]] = []           # [label, num, title, text]
    title = ""
    cur: list[str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if cur is not None:
                cur[3] += "\n"
            continue
        m_ch = RE_CHAPTER.match(line)
        m_art = RE_ARTICLE.match(line)
        m_pt = RE_POINT.match(line)
        if m_ch and not m_pt:
            title = line.strip()
            cur = None
            continue
        if m_art:
            cur = [f"статья {m_art.group(1)}", m_art.group(1), title, line.strip()]
            items.append(cur)
        elif m_pt:
            cur = [f"п. {m_pt.group(1)}", m_pt.group(1), title, line.strip()]
            items.append(cur)
        elif cur is not None:
            cur[3] += ("" if cur[3].endswith("\n") else " ") + line.strip()
        else:                               # преамбула до первого пункта
            cur = ["преамбула", "", title, line.strip()]
            items.append(cur)
    return [(lab, num, ttl, re.sub(r"\s*\n\s*", " ", txt).strip()) for lab, num, ttl, txt in items if txt.strip()]


def load_chunks(base: str) -> list[Chunk]:
    if not os.path.isdir(base):
        log.warning("нет папки нормативной базы %s", base)
        return []
    meta = load_sources(base)
    chunks: list[Chunk] = []
    for name in sorted(os.listdir(base)):
        if not name.lower().endswith(EXT) or name.lower() == "readme.md":
            continue
        text = read_text(os.path.join(base, name))
        if not text.strip():
            continue
        m = meta.get(name) or {}
        document = str(m.get("document") or os.path.splitext(name)[0])
        source = str(m.get("source") or "legal_base")
        url = m.get("url") or None
        arts = split_articles(text)
        if len(arts) <= 1:                  # структуры нет — режем на фрагменты
            arts = [(f"фрагмент {i + 1}", "", "", t) for i, t in enumerate(_split_long(re.sub(r"\s+", " ", text), CHUNK_CHARS))]
        for label, num, title, body in arts:
            for k, part in enumerate(_split_long(body, CHUNK_CHARS)):
                chunks.append(Chunk(name, document, source, url, label, num, title, part, k))
        log.info("нормативная база: %s — %d фрагментов", name, sum(1 for c in chunks if c.doc_file == name))
    return chunks
