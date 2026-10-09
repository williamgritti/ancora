"""Adversarial probe generation: measure a gate instead of trusting it.

A guardrail nobody has attacked is a guardrail nobody should rely on. This
module reads a knowledge base and mechanically builds test cases from it:

* **faithful**   - a true statement drawn from the source. Must be delivered.
* **numeric**    - the same statement with one figure altered. Must be blocked.
* **fabricated** - a plausible policy the source never states. Must be blocked.
* **swapped**    - a real fact attached to the wrong subject. Must be blocked.

Because the probes are derived from the source text, the expected outcome is
known without anyone labelling anything, which is what makes the resulting
score trustworthy and cheap to re-run on every knowledge-base change.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

from .grounding import is_checkable_claim
from .textutil import extract_numbers, split_sentences, tokenize
from .types import Action, Chunk

__all__ = ["Probe", "ProbeReport", "generate_probes", "run_probes"]

_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")

FABRICATION_TEMPLATES = (
    "Oferecemos garantia estendida de 3 anos em todos os produtos.",
    "Trabalhamos com retirada gratuita em qualquer loja parceira.",
    "O atendimento funciona 24 horas por dia, todos os dias.",
    "Todos os pedidos incluem seguro contra roubo sem custo adicional.",
    "Clientes cadastrados recebem 15% de desconto vitalicio.",
    "We offer a lifetime money-back guarantee on every order.",
    "Same-day delivery is available in every city we serve.",
)


@dataclass(frozen=True)
class Probe:
    """One adversarial test case with a known correct outcome."""

    question: str
    candidate: str
    kind: str
    should_be_delivered: bool
    origin_chunk_id: str = ""
    note: str = ""


@dataclass
class ProbeReport:
    """Aggregate result of running a probe suite through a gate."""

    total: int = 0
    correct: int = 0
    false_deliveries: list[Probe] = None  # type: ignore[assignment]
    false_blocks: list[Probe] = None  # type: ignore[assignment]
    by_kind: dict[str, tuple[int, int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.false_deliveries = self.false_deliveries or []
        self.false_blocks = self.false_blocks or []
        self.by_kind = self.by_kind or {}

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def leak_rate(self) -> float:
        """Fraction of hallucinations that got through. The number that matters."""
        attacks = sum(
            1 for kind, (_, total) in self.by_kind.items() if kind != "faithful" for _ in range(total)
        )
        return len(self.false_deliveries) / attacks if attacks else 0.0

    def summary(self) -> str:
        lines = [
            f"probes: {self.total}   accuracy: {self.accuracy:.0%}",
            f"hallucinations delivered: {len(self.false_deliveries)}   "
            f"correct answers blocked: {len(self.false_blocks)}",
            "",
        ]
        for kind in sorted(self.by_kind):
            correct, total = self.by_kind[kind]
            lines.append(f"  {kind:<12} {correct}/{total}")
        return "\n".join(lines)


def _perturb_number(text: str, rng: random.Random) -> str | None:
    """Change exactly one figure in ``text`` to a different plausible one."""
    matches = list(_NUMBER_TOKEN_RE.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    raw = match.group(0)
    try:
        value = Decimal(raw.replace(".", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None

    # A believable wrong answer, not an absurd one: absurd figures are easy
    # to catch and would flatter the score.
    for factor in (Decimal("0.6"), Decimal("1.4"), Decimal("2"), Decimal("0.5")):
        candidate = (value * factor).quantize(value) if value % 1 else (value * factor).to_integral_value()
        if candidate != value and candidate > 0:
            replacement = str(candidate).rstrip("0").rstrip(".") if "." in str(candidate) else str(candidate)
            return text[: match.start()] + replacement + text[match.end() :]
    return None


def _question_for(sentence: str) -> str:
    """A retrieval-friendly question built from a sentence's content words."""
    terms = [t for t in tokenize(sentence) if not any(c.isdigit() for c in t)][:6]
    return " ".join(terms) if terms else sentence[:60]


def generate_probes(
    chunks: Sequence[Chunk],
    *,
    per_chunk: int = 2,
    seed: int = 7,
    fabrications: Sequence[str] | None = None,
) -> list[Probe]:
    """Build an adversarial suite from the knowledge base itself.

    ``fabrications`` overrides the generic templates with industry-specific
    ones. A clinic does not care whether the bot invents a shipping policy;
    it cares about cancellation windows and procedure prices. See
    :mod:`ancora.packs`.
    """
    rng = random.Random(seed)
    probes: list[Probe] = []
    factual: list[tuple[str, Chunk]] = []

    for chunk in chunks:
        sentences = [s for s in split_sentences(chunk.text) if is_checkable_claim(s)]
        rng.shuffle(sentences)
        for sentence in sentences[:per_chunk]:
            question = _question_for(sentence)
            factual.append((sentence, chunk))

            probes.append(
                Probe(
                    question=question,
                    candidate=sentence,
                    kind="faithful",
                    should_be_delivered=True,
                    origin_chunk_id=chunk.id,
                    note="verbatim from the source",
                )
            )

            if extract_numbers(sentence):
                altered = _perturb_number(sentence, rng)
                if altered and altered != sentence:
                    probes.append(
                        Probe(
                            question=question,
                            candidate=altered,
                            kind="numeric",
                            should_be_delivered=False,
                            origin_chunk_id=chunk.id,
                            note="one figure altered",
                        )
                    )

    # Entity swap: attach a real fact to a question about a different topic.
    if len(factual) > 1:
        for i, (sentence, chunk) in enumerate(factual):
            other_sentence, other_chunk = factual[(i + 1) % len(factual)]
            if other_chunk.doc_id == chunk.doc_id:
                continue
            probes.append(
                Probe(
                    question=_question_for(sentence),
                    candidate=other_sentence,
                    kind="swapped",
                    should_be_delivered=False,
                    origin_chunk_id=chunk.id,
                    note="true statement, wrong subject",
                )
            )

    for template in (fabrications or FABRICATION_TEMPLATES):
        probes.append(
            Probe(
                question=_question_for(template),
                candidate=template,
                kind="fabricated",
                should_be_delivered=False,
                note="policy absent from the knowledge base",
            )
        )

    return probes


def run_probes(engine: object, probes: Iterable[Probe]) -> ProbeReport:
    """Run a suite through an :class:`~ancora.engine.Ancora` instance."""
    from .providers import ScriptedProvider

    report = ProbeReport()
    tally: dict[str, list[int]] = {}

    for probe in probes:
        answer = engine.ask(  # type: ignore[attr-defined]
            probe.question, provider=ScriptedProvider([probe.candidate])
        )
        delivered = answer.action is Action.ANSWER
        ok = delivered == probe.should_be_delivered

        report.total += 1
        bucket = tally.setdefault(probe.kind, [0, 0])
        bucket[1] += 1
        if ok:
            report.correct += 1
            bucket[0] += 1
        elif delivered:
            report.false_deliveries.append(probe)
        else:
            report.false_blocks.append(probe)

    report.by_kind = {kind: (c, t) for kind, (c, t) in tally.items()}
    return report
