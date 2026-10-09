"""The grounding gate: does the evidence actually support the answer?

The gate runs after generation and before delivery. It is deliberately
model-free — a second LLM call to "check" the first one shares the first
one's failure modes and doubles the cost. Instead each claim is verified
against the retrieved text by weighted lexical coverage plus a hard
numeric guard.

The numeric guard is the part that matters commercially. The expensive
hallucination in a support bot is not a vague sentence, it is an invented
price, deadline or percentage. A number that appears in no cited chunk is
rejected outright regardless of how well the rest of the sentence scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .textutil import extract_numbers, split_sentences, tokenize
from .types import ClaimCheck, Chunk

__all__ = ["GroundingChecker", "is_checkable_claim"]


# Sentences that carry no verifiable content. Checking them produces noise:
# a greeting has no support in a knowledge base and never should.
_NON_CLAIM_PATTERNS = (
    r"^(ola|oi|bom dia|boa tarde|boa noite|hello|hi|hey|ciao|buongiorno)\b",
    r"^(obrigad|thank|grazie|de nada|you're welcome|disponha)",
    r"^(posso ajudar|precisa de mais|anything else|algo mais|mais alguma)",
    r"^(claro|certamente|com certeza|sure|of course|certo)\b[.!,]?$",
    r"^(vou verificar|deixa comigo|um momento|one moment|let me check)",
    r"\?$",  # questions assert nothing
)
_NON_CLAIM_RE = re.compile("|".join(_NON_CLAIM_PATTERNS), re.IGNORECASE)

# Hedges do not make a sentence unverifiable, but they do lower the bar for
# how confidently it must be supported.
_HEDGE_RE = re.compile(
    r"\b(talvez|possivelmente|geralmente|normalmente|costuma|pode ser|"
    r"maybe|possibly|generally|usually|typically|might|may)\b",
    re.IGNORECASE,
)


def is_checkable_claim(sentence: str, *, min_content_tokens: int = 2) -> bool:
    """Whether a sentence asserts something worth verifying."""
    text = (sentence or "").strip()
    if not text:
        return False
    from .textutil import normalize

    if _NON_CLAIM_RE.search(normalize(text)):
        return False
    return len(tokenize(text)) >= min_content_tokens


class GroundingChecker:
    """Verifies candidate answers against retrieved evidence.

    ``support_threshold`` is the fraction of a claim's IDF-weighted content
    tokens that must appear in a single supporting chunk. 0.55 is a
    deliberate default: high enough to reject invention, low enough to
    tolerate paraphrase, which is what you want from a model that is
    summarising rather than quoting.
    """

    def __init__(
        self,
        *,
        support_threshold: float = 0.55,
        hedged_threshold: float = 0.40,
        require_numeric_support: bool = True,
        numeric_scope_threshold: float = 0.30,
        min_content_tokens: int = 2,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1")
        if not 0.0 <= hedged_threshold <= 1.0:
            raise ValueError("hedged_threshold must be between 0 and 1")
        if not 0.0 <= numeric_scope_threshold <= 1.0:
            raise ValueError("numeric_scope_threshold must be between 0 and 1")
        self.support_threshold = support_threshold
        self.hedged_threshold = hedged_threshold
        self.require_numeric_support = require_numeric_support
        self.numeric_scope_threshold = numeric_scope_threshold
        self.min_content_tokens = min_content_tokens

    # -- internals -------------------------------------------------------

    @staticmethod
    def _idf(chunks: Sequence[Chunk]) -> dict[str, float]:
        """Inverse document frequency over the evidence set.

        Rare words carry the meaning of a claim; without weighting, a
        sentence made of common words scores high against any chunk.
        """
        n = len(chunks) or 1
        df: Counter[str] = Counter()
        for chunk in chunks:
            df.update(set(tokenize(chunk.text)))
        return {term: math.log(1.0 + n / (freq or 1)) for term, freq in df.items()}

    @staticmethod
    def _lexical_tokens(tokens: Sequence[str]) -> list[str]:
        """Drop numeric tokens from a token list.

        Coverage must measure *topical* overlap only. If a claim's own figure
        counts toward selecting the chunk that will then license that figure,
        the guard becomes circular: a payments page containing "3 dias uteis"
        outranks the delivery page for the claim "delivery takes 3 days",
        purely because it shares the invented number.
        """
        return [t for t in tokens if not any(ch.isdigit() for ch in t)]

    def _coverage(
        self, claim_tokens: Sequence[str], chunk_tokens: set[str], idf: dict[str, float]
    ) -> float:
        """IDF-weighted fraction of a claim's non-numeric tokens in the chunk."""
        claim_tokens = self._lexical_tokens(claim_tokens)
        if not claim_tokens:
            return 0.0
        default_idf = math.log(2.0)
        total = sum(idf.get(t, default_idf) for t in claim_tokens)
        if total <= 0:
            return 0.0
        matched = sum(idf.get(t, default_idf) for t in claim_tokens if t in chunk_tokens)
        return matched / total

    # -- public API ------------------------------------------------------

    def check_claim(
        self, claim: str, evidence: Sequence[Chunk], idf: dict[str, float] | None = None
    ) -> ClaimCheck:
        """Verify one sentence against the evidence chunks."""
        idf = idf if idf is not None else self._idf(evidence)
        claim_tokens = tokenize(claim)
        threshold = self.hedged_threshold if _HEDGE_RE.search(claim) else self.support_threshold

        scored: list[tuple[float, "Chunk"]] = []
        best_score = 0.0
        supporting: list[str] = []
        for chunk in evidence:
            score = self._coverage(claim_tokens, set(tokenize(chunk.text)), idf)
            scored.append((score, chunk))
            if score > best_score:
                best_score = score
            if score >= threshold:
                supporting.append(chunk.id)

        # Numeric guard: every number asserted must appear in a chunk that is
        # *about the same subject* as the claim.
        #
        # Scoping matters. Checking against the union of all retrieved chunks
        # means an unrelated document licenses any figure it happens to
        # contain: a payments page that mentions "3 dias uteis" would silently
        # authorise "delivery takes 3 days". So only chunks that clear a
        # relaxed lexical bar count, plus the single best chunk, which keeps
        # a fact split across neighbouring chunks of one document verifiable.
        unsupported_numbers: tuple[str, ...] = ()
        if self.require_numeric_support:
            claimed = extract_numbers(claim)
            if claimed:
                # Relative floor: a chunk only licenses a figure if it is
                # nearly as on-topic as the best match. An absolute threshold
                # alone is too permissive once several documents share generic
                # vocabulary ("dias uteis" appears on both the delivery and
                # the payments page).
                floor = max(self.numeric_scope_threshold, best_score * 0.75)
                in_scope = [c for s, c in scored if s >= floor]
                if not in_scope and scored:
                    in_scope = [max(scored, key=lambda pair: pair[0])[1]]
                available: set[str] = set()
                for chunk in in_scope:
                    available.update(extract_numbers(chunk.text))
                missing = [n for n in claimed if n not in available]
                # A bare percentage is also satisfied by the plain number
                # appearing in the source ("30%" vs "30 por cento").
                missing = [
                    n for n in missing
                    if not (n.endswith("%") and n[:-1] in available)
                ]
                unsupported_numbers = tuple(dict.fromkeys(missing))

        if unsupported_numbers:
            return ClaimCheck(
                claim=claim,
                supported=False,
                support_score=best_score,
                supporting_chunk_ids=tuple(supporting),
                unsupported_numbers=unsupported_numbers,
                reason=(
                    "numbers not present in any relevant source: "
                    + ", ".join(unsupported_numbers)
                ),
            )

        supported = bool(supporting)
        return ClaimCheck(
            claim=claim,
            supported=supported,
            support_score=best_score,
            supporting_chunk_ids=tuple(supporting),
            reason="" if supported else f"lexical support {best_score:.2f} < {threshold:.2f}",
        )

    def relevance(self, question: str, answer: str, evidence: Sequence[Chunk]) -> float:
        """How much of the question's topic the answer actually addresses.

        Groundedness and relevance are different properties. A support bot
        can reply with a sentence copied verbatim from the manual - perfectly
        traceable - that answers a question nobody asked. The claim checker
        cannot see that, because it only ever compares the answer to the
        evidence, never to the question.

        Scored as IDF-weighted coverage of the question's content tokens by
        the *answer alone*. The evidence is deliberately excluded from the
        haystack: it was retrieved for this question, so it covers the
        question by construction and would make every answer look relevant.
        ``evidence`` is used only to weight terms by rarity.
        """
        question_tokens = self._lexical_tokens(tokenize(question))
        if not question_tokens:
            return 1.0
        idf = self._idf(evidence) if evidence else {}
        return self._coverage(question_tokens, set(tokenize(answer)), idf)

    def check_answer(
        self, answer: str, evidence: Sequence[Chunk]
    ) -> tuple[ClaimCheck, ...]:
        """Verify every checkable sentence of a candidate answer."""
        if not answer or not answer.strip():
            return ()
        idf = self._idf(evidence)
        return tuple(
            self.check_claim(sentence, evidence, idf)
            for sentence in split_sentences(answer)
            if is_checkable_claim(sentence, min_content_tokens=self.min_content_tokens)
        )
