"""Sentence-aware chunking that preserves character offsets.

Offsets matter: an operator auditing an escalated answer needs to jump to
the exact span of the source document, not to a fuzzy "somewhere in doc 3".
"""

from __future__ import annotations

from .types import Chunk, Document

__all__ = ["chunk_document", "chunk_documents"]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences, computed against the *original* text."""
    from .textutil import split_sentences

    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(text):
        idx = text.find(sentence, cursor)
        if idx == -1:
            # Sentence was normalised (stripped bullet chars); fall back to
            # locating its first few words instead of dropping it.
            probe = sentence[:24]
            idx = text.find(probe, cursor)
            if idx == -1:
                continue
        spans.append((idx, idx + len(sentence)))
        cursor = idx + len(sentence)
    return spans


def chunk_document(
    doc: Document,
    *,
    target_chars: int = 700,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Group whole sentences into chunks of roughly ``target_chars``.

    Chunks never split a sentence, because a half-sentence is both a bad
    retrieval unit and a bad citation. ``overlap_sentences`` repeats the
    tail of each chunk at the head of the next so that a fact stated across
    a sentence boundary stays retrievable.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    text = doc.text or ""
    spans = _sentence_spans(text)
    if not spans:
        stripped = text.strip()
        if not stripped:
            return []
        start = text.find(stripped)
        return [
            Chunk(
                id=f"{doc.id}#0",
                doc_id=doc.id,
                text=stripped,
                start=start,
                end=start + len(stripped),
                metadata=dict(doc.metadata),
            )
        ]

    chunks: list[Chunk] = []
    group: list[tuple[int, int]] = []
    index = 0

    def flush() -> None:
        nonlocal group, index
        if not group:
            return
        start, end = group[0][0], group[-1][1]
        chunks.append(
            Chunk(
                id=f"{doc.id}#{index}",
                doc_id=doc.id,
                text=text[start:end].strip(),
                start=start,
                end=end,
                metadata=dict(doc.metadata),
            )
        )
        index += 1

    for span in spans:
        prospective = (group[0][0] if group else span[0], span[1])
        if group and (prospective[1] - prospective[0]) > target_chars:
            flush()
            group = group[-overlap_sentences:] if overlap_sentences else []
        group.append(span)

    flush()
    return [c for c in chunks if c.text]


def chunk_documents(docs: list[Document], **kwargs: object) -> list[Chunk]:
    """Chunk many documents, preserving input order."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, **kwargs))  # type: ignore[arg-type]
    return out
