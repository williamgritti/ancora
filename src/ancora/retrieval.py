"""Okapi BM25 retrieval in pure Python.

No numpy, no vector database, no network call. For a knowledge base of the
size a support bot actually has (a few hundred pages), lexical retrieval is
both competitive and auditable, and it never silently degrades because an
embedding endpoint is down.

An ``Embedder`` hook is provided for callers who do want dense retrieval;
scores are then blended, not replaced, so a dense model cannot drag in a
chunk that shares no vocabulary with the question.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

from .textutil import tokenize
from .types import Chunk, ScoredChunk

__all__ = ["BM25Index", "Embedder", "Retriever"]


@runtime_checkable
class Embedder(Protocol):
    """Optional dense-retrieval hook."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class BM25Index:
    """Okapi BM25 over a fixed set of chunks.

    ``k1`` controls term-frequency saturation and ``b`` the length
    normalisation; the defaults are the standard ones and are rarely worth
    tuning for knowledge bases of this size.
    """

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = [tokenize(c.text) for c in self.chunks]
        self._lengths: list[int] = [len(t) for t in self._tokens]
        self._avg_len: float = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._tf: list[Counter[str]] = [Counter(t) for t in self._tokens]

        df: Counter[str] = Counter()
        for token_set in ({t for t in toks} for toks in self._tokens):
            df.update(token_set)
        n = len(self.chunks)
        # BM25+ style idf floor keeps very common terms from going negative.
        self._idf: dict[str, float] = {
            term: max(math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)), 1e-6)
            for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def score(self, query: str) -> list[float]:
        """BM25 score of every chunk against ``query``."""
        terms = tokenize(query)
        scores = [0.0] * len(self.chunks)
        if not terms or not self._avg_len:
            return scores
        for i, tf in enumerate(self._tf):
            length = self._lengths[i] or 1
            total = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                total += idf * (freq * (self.k1 + 1)) / denom
            scores[i] = total
        return scores


class Retriever:
    """Ranks chunks for a question, optionally blending dense similarity."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder | None = None,
        dense_weight: float = 0.35,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        self.index = BM25Index(chunks)
        self.embedder = embedder
        self.dense_weight = dense_weight if embedder else 0.0
        self._vectors: Sequence[Sequence[float]] | None = None
        if embedder and chunks:
            self._vectors = embedder.embed([c.text for c in chunks])

    @property
    def chunks(self) -> list[Chunk]:
        return self.index.chunks

    def search(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        """Return the ``top_k`` best chunks, scores normalised to 0..1."""
        if not self.index.chunks or top_k <= 0:
            return []

        lexical = self.index.score(query)
        peak = max(lexical) if lexical else 0.0
        combined = [s / peak for s in lexical] if peak > 0 else [0.0] * len(lexical)

        if self.embedder is not None and self._vectors:
            qvec = self.embedder.embed([query])[0]
            dense = [_cosine(qvec, v) for v in self._vectors]
            w = self.dense_weight
            combined = [(1 - w) * lex + w * max(d, 0.0) for lex, d in zip(combined, dense)]

        ranked = sorted(
            (ScoredChunk(chunk=c, score=s) for c, s in zip(self.index.chunks, combined)),
            key=lambda sc: (-sc.score, sc.chunk.id),
        )
        return [sc for sc in ranked[:top_k] if sc.score > 0.0]
