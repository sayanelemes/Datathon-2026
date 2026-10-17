"""Эмбеддинги и векторный индекс (локально, без интернета).

Эмбеддер по умолчанию — хэширующий TF-IDF по основам слов и символьным n-граммам (работает сразу, без
скачивания моделей, устойчив к падежам). Можно включить нейросетевой: LEGAL_EMBED_MODEL=<путь или имя
sentence-transformers, уже скачанной>. Индекс — FAISS (если установлен) или numpy (то же косинусное сходство).
Индекс сохраняется в LEGAL_INDEX_DIR и пересобирается, если поменялись файлы в legal_base/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import zlib
from dataclasses import asdict

import numpy as np

from .loader import Chunk, load_chunks

log = logging.getLogger("legal")
DIM = int(os.getenv("LEGAL_EMBED_DIM", 4096))


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    feats = []
    for w in re.findall(r"[a-zа-я0-9]+", text):
        if len(w) < 2:
            continue
        feats.append("w:" + w[:6])                          # грубая основа: «пересчитываются» ≈ «пересчёт»
        if len(w) >= 5:
            feats += ["g:" + w[i:i + 4] for i in range(len(w) - 3)]
    return feats


class HashingEmbedder:
    name = f"hash-tfidf-{DIM}"

    def __init__(self) -> None:
        self.idf = np.ones(DIM, dtype=np.float32)

    def _tf(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for f in _tokens(text):
            v[zlib.crc32(f.encode()) % DIM] += 0.5 if f.startswith("g:") else 1.0
        return np.log1p(v)

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(DIM, dtype=np.float32)
        for t in texts:
            df += self._tf(t) > 0
        self.idf = np.log((1 + len(texts)) / (1 + df)).astype(np.float32) + 1.0

    def encode(self, texts: list[str]) -> np.ndarray:
        m = np.stack([self._tf(t) * self.idf for t in texts]) if texts else np.zeros((0, DIM), np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(n, 1e-9)

    def state(self) -> dict:
        return {"idf": self.idf}

    def load_state(self, st: dict) -> None:
        self.idf = st["idf"]


class STEmbedder:
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device=os.getenv("LEGAL_EMBED_DEVICE", "cpu"))
        self.name = f"st:{model}"

    def fit(self, texts: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def state(self) -> dict:
        return {}

    def load_state(self, st: dict) -> None:
        pass


def make_embedder():
    name = os.getenv("LEGAL_EMBED_MODEL")
    if name:
        try:
            return STEmbedder(name)
        except Exception as e:  # noqa: BLE001
            log.warning("эмбеддер %s недоступен (%r) — использую hash-tfidf", name, e)
    return HashingEmbedder()


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder) -> None:
        self.chunks, self.vectors, self.embedder = chunks, vectors.astype(np.float32), embedder
        self.faiss = None
        try:
            import faiss
            self.faiss = faiss.IndexFlatIP(vectors.shape[1])
            if len(vectors):
                self.faiss.add(self.vectors)
        except Exception:  # noqa: BLE001 — faiss необязателен
            self.faiss = None

    @property
    def backend(self) -> str:
        return "faiss" if self.faiss is not None else "numpy"

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query]).astype(np.float32)
        k = min(k, len(self.chunks))
        if self.faiss is not None:
            sc, idx = self.faiss.search(q, k)
            return [(self.chunks[i], float(s)) for i, s in zip(idx[0], sc[0]) if i >= 0]
        sims = self.vectors @ q[0]
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


def fingerprint(base: str, embedder_name: str) -> str:
    h = hashlib.sha1(embedder_name.encode())
    if os.path.isdir(base):
        for n in sorted(os.listdir(base)):
            p = os.path.join(base, n)
            if os.path.isfile(p):
                st = os.stat(p)
                h.update(f"{n}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def build_or_load(base: str, cache_dir: str | None) -> VectorIndex:
    emb = make_embedder()
    fp = fingerprint(base, emb.name)
    if cache_dir:
        npz, js = os.path.join(cache_dir, f"index_{fp}.npz"), os.path.join(cache_dir, f"chunks_{fp}.json")
        if os.path.exists(npz) and os.path.exists(js):
            try:
                data = np.load(npz)
                emb.load_state({k: data[k] for k in data.files if k != "vectors"})
                with open(js, encoding="utf-8") as f:
                    chunks = [Chunk(**c) for c in json.load(f)]
                log.info("нормативный индекс загружен из кэша: %d фрагментов", len(chunks))
                return VectorIndex(chunks, data["vectors"], emb)
            except Exception as e:  # noqa: BLE001
                log.warning("кэш индекса повреждён (%r) — пересобираю", e)
    chunks = load_chunks(base)
    texts = [f"{c.title} {c.article} {c.text}" for c in chunks]
    emb.fit(texts)
    vecs = emb.encode(texts) if texts else np.zeros((0, DIM), np.float32)
    idx = VectorIndex(chunks, vecs, emb)
    if cache_dir and chunks:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(os.path.join(cache_dir, f"index_{fp}.npz"), vectors=vecs, **emb.state())
            with open(os.path.join(cache_dir, f"chunks_{fp}.json"), "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in chunks], f, ensure_ascii=False)
        except OSError as e:
            log.warning("не удалось сохранить индекс: %s", e)
    log.info("нормативный индекс: %d фрагментов, эмбеддер %s, поиск %s", len(chunks), emb.name, idx.backend)
    return idx
