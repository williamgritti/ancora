"""Core data types for Ancora.

Everything here is a plain dataclass: no pydantic, no runtime magic. The
grounding pipeline is meant to be auditable, so the objects that flow
through it stay boring and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Outcome of the grounding gate for a whole answer."""

    GROUNDED = "grounded"
    """Every checked claim is supported by retrieved evidence."""

    UNGROUNDED = "ungrounded"
    """At least one claim has no support in the evidence."""

    NO_EVIDENCE = "no_evidence"
    """Retrieval returned nothing usable for the question."""

    UNSAFE_NUMERIC = "unsafe_numeric"
    """A number appears in the answer that appears in no cited chunk."""

    OFF_TOPIC = "off_topic"
    """Every claim is grounded, but the answer does not address the question."""


class Action(str, Enum):
    """What the caller should actually do with the answer."""

    ANSWER = "answer"
    """Send the answer to the user."""

    ESCALATE = "escalate"
    """Hand off to a human. The draft is attached for the operator."""

    REFUSE = "refuse"
    """Reply with the configured refusal text. No human available."""


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document.id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("Document.text must be a string")


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a document.

    ``start``/``end`` are character offsets into the parent document, which
    lets an operator jump straight to the source when auditing an answer.
    """

    id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its retrieval score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ClaimCheck:
    """The grounding result for one sentence of a candidate answer."""

    claim: str
    supported: bool
    support_score: float
    supporting_chunk_ids: tuple[str, ...] = ()
    unsupported_numbers: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class Answer:
    """The full, auditable result of one ask()."""

    question: str
    action: Action
    verdict: Verdict
    text: str
    """What to send. Refusal/escalation text when action is not ANSWER."""

    draft: str = ""
    """The model's raw candidate answer, kept even when it was rejected."""

    citations: tuple[Chunk, ...] = ()
    claim_checks: tuple[ClaimCheck, ...] = ()
    retrieved: tuple[ScoredChunk, ...] = ()
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_answerable(self) -> bool:
        return self.action is Action.ANSWER

    def sources(self) -> list[str]:
        """Human-readable source labels, de-duplicated, order preserved."""
        seen: list[str] = []
        for chunk in self.citations:
            label = str(chunk.metadata.get("source", chunk.doc_id))
            if label not in seen:
                seen.append(label)
        return seen

    def to_dict(self) -> dict[str, Any]:
        """Flatten to JSON-safe primitives for logging or an audit trail."""
        return {
            "question": self.question,
            "action": self.action.value,
            "verdict": self.verdict.value,
            "text": self.text,
            "draft": self.draft,
            "confidence": round(self.confidence, 4),
            "sources": self.sources(),
            "citations": [
                {
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                }
                for c in self.citations
            ],
            "claim_checks": [
                {
                    "claim": cc.claim,
                    "supported": cc.supported,
                    "support_score": round(cc.support_score, 4),
                    "supporting_chunk_ids": list(cc.supporting_chunk_ids),
                    "unsupported_numbers": list(cc.unsupported_numbers),
                    "reason": cc.reason,
                }
                for cc in self.claim_checks
            ],
            "diagnostics": self.diagnostics,
        }
