"""What to do when the gate rejects an answer.

Separated from the checker on purpose: whether a claim is grounded is a
factual question, whether an ungrounded answer should escalate or refuse is
a business decision, and the two change at different rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .types import Action, ClaimCheck, Verdict

__all__ = ["Policy", "decide"]

INSUFFICIENT_MARKERS = ("insuficiente", "insufficient", "não sei", "nao sei", "i don't know")


@dataclass
class Policy:
    """Thresholds and copy for the escalation decision."""

    min_retrieval_score: float = 0.12
    """Below this best-hit score the knowledge base is treated as silent."""

    max_unsupported_claims: int = 0
    """How many ungrounded sentences are tolerated. Zero is the point."""

    min_relevance: float = 0.25
    """Minimum topical overlap between question and answer.

    Guards the failure mode where a grounded sentence answers the wrong
    question. Set to 0.0 to disable.
    """

    escalate_to_human: bool = True
    """When False, rejected answers become a refusal instead of a handoff."""

    refusal_text: str = (
        "Não tenho essa informação confirmada na minha base. "
        "Prefiro não arriscar uma resposta errada."
    )
    escalation_text: str = (
        "Vou encaminhar você para um atendente humano para confirmar isso com precisão."
    )
    escalation_tags: Sequence[str] = field(default_factory=lambda: ("needs-human",))

    def __post_init__(self) -> None:
        if self.max_unsupported_claims < 0:
            raise ValueError("max_unsupported_claims must be >= 0")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        if not 0.0 <= self.min_retrieval_score <= 1.0:
            raise ValueError("min_retrieval_score must be between 0 and 1")


def decide(
    *,
    policy: Policy,
    best_retrieval_score: float,
    claim_checks: Sequence[ClaimCheck],
    draft: str,
) -> tuple[Action, Verdict, float]:
    """Map evidence quality and claim checks onto an action.

    Returns ``(action, verdict, confidence)`` where confidence is the mean
    support score across checked claims, damped by retrieval quality.
    """
    normalized_draft = (draft or "").strip().lower()

    # The model declining is a success, not a failure: it recognised the gap.
    if any(marker in normalized_draft for marker in INSUFFICIENT_MARKERS):
        action = Action.ESCALATE if policy.escalate_to_human else Action.REFUSE
        return action, Verdict.NO_EVIDENCE, 0.0

    if best_retrieval_score < policy.min_retrieval_score:
        action = Action.ESCALATE if policy.escalate_to_human else Action.REFUSE
        return action, Verdict.NO_EVIDENCE, 0.0

    if not claim_checks:
        # Nothing verifiable was said. Safe to pass through: greetings and
        # clarifying questions land here.
        return Action.ANSWER, Verdict.GROUNDED, min(best_retrieval_score, 1.0)

    numeric_failures = [c for c in claim_checks if c.unsupported_numbers]
    unsupported = [c for c in claim_checks if not c.supported]
    mean_support = sum(c.support_score for c in claim_checks) / len(claim_checks)
    confidence = round(mean_support * min(1.0, best_retrieval_score + 0.35), 4)

    if numeric_failures:
        action = Action.ESCALATE if policy.escalate_to_human else Action.REFUSE
        return action, Verdict.UNSAFE_NUMERIC, confidence

    if len(unsupported) > policy.max_unsupported_claims:
        action = Action.ESCALATE if policy.escalate_to_human else Action.REFUSE
        return action, Verdict.UNGROUNDED, confidence

    return Action.ANSWER, Verdict.GROUNDED, confidence
