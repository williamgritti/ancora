"""The Ancora engine: retrieve, generate, verify, decide."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import chunk_documents
from .grounding import GroundingChecker
from .policy import Policy, decide
from .providers import DEFAULT_SYSTEM_PROMPT, Provider, build_prompt
from .retrieval import Embedder, Retriever
from .types import Action, Answer, Chunk, Document, Verdict

__all__ = ["Ancora"]

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}


class Ancora:
    """A grounded question-answering engine with a hard refusal path.

    The contract: ``ask()`` either returns an answer every sentence of which
    is traceable to a cited source, or it returns an escalation. It never
    returns a confident guess.

        >>> kb = Ancora.from_texts({"faq": "O frete custa R$ 24,90."})
        >>> kb.ask("quanto custa o frete?", provider=EchoProvider()).is_answerable
        True
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        checker: GroundingChecker | None = None,
        policy: Policy | None = None,
        embedder: Embedder | None = None,
        top_k: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever = Retriever(chunks, embedder=embedder)
        self.checker = checker or GroundingChecker()
        self.policy = policy or Policy()
        self.top_k = top_k
        self.system_prompt = system_prompt

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_documents(
        cls, docs: Iterable[Document], *, target_chars: int = 700, **kwargs: object
    ) -> "Ancora":
        chunks = chunk_documents(list(docs), target_chars=target_chars)
        return cls(chunks, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_texts(cls, texts: dict[str, str], **kwargs: object) -> "Ancora":
        """Build from an ``{id: text}`` mapping."""
        docs = [
            Document(id=doc_id, text=text, metadata={"source": doc_id})
            for doc_id, text in texts.items()
        ]
        return cls.from_documents(docs, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_directory(
        cls, path: str | Path, *, pattern: str = "**/*", **kwargs: object
    ) -> "Ancora":
        """Load every text-like file under ``path`` as a document."""
        root = Path(path)
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")
        docs: list[Document] = []
        for file in sorted(root.glob(pattern)):
            if not file.is_file() or file.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if text.strip():
                rel = file.relative_to(root).as_posix()
                docs.append(Document(id=rel, text=text, metadata={"source": rel}))
        if not docs:
            raise ValueError(f"no readable text documents found under {root}")
        return cls.from_documents(docs, **kwargs)  # type: ignore[arg-type]

    # -- main entry point ------------------------------------------------

    def ask(self, question: str, *, provider: Provider, top_k: int | None = None) -> Answer:
        """Answer ``question``, or escalate. Never guesses."""
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        started = time.perf_counter()
        k = top_k or self.top_k
        retrieved = self.retriever.search(question, top_k=k)
        evidence = [sc.chunk for sc in retrieved]
        best_score = retrieved[0].score if retrieved else 0.0

        if not evidence:
            return self._reject(
                question,
                draft="",
                verdict=Verdict.NO_EVIDENCE,
                retrieved=(),
                claim_checks=(),
                confidence=0.0,
                elapsed=time.perf_counter() - started,
                note="retrieval returned no chunks",
            )

        prompt = build_prompt(question, [c.text for c in evidence])
        draft = (provider.complete(prompt, system=self.system_prompt) or "").strip()

        claim_checks = self.checker.check_answer(draft, evidence)
        action, verdict, confidence = decide(
            policy=self.policy,
            best_retrieval_score=best_score,
            claim_checks=claim_checks,
            draft=draft,
        )

        # Relevance is checked only on answers that already passed grounding:
        # a rejected answer has a more specific verdict worth keeping.
        relevance = 1.0
        if action is Action.ANSWER and self.policy.min_relevance > 0 and claim_checks:
            relevance = self.checker.relevance(question, draft, evidence)
            if relevance < self.policy.min_relevance:
                action = (
                    Action.ESCALATE if self.policy.escalate_to_human else Action.REFUSE
                )
                verdict = Verdict.OFF_TOPIC

        elapsed = time.perf_counter() - started
        if action is Action.ANSWER:
            cited_ids = {
                cid for check in claim_checks for cid in check.supporting_chunk_ids
            }
            citations = tuple(c for c in evidence if c.id in cited_ids) or tuple(evidence[:1])
            return Answer(
                question=question,
                action=Action.ANSWER,
                verdict=verdict,
                text=draft,
                draft=draft,
                citations=citations,
                claim_checks=claim_checks,
                retrieved=tuple(retrieved),
                confidence=confidence,
                diagnostics={
                    **self._diagnostics(elapsed, best_score, len(evidence)),
                    "relevance": round(relevance, 4),
                },
            )

        return self._reject(
            question,
            draft=draft,
            verdict=verdict,
            retrieved=tuple(retrieved),
            claim_checks=claim_checks,
            confidence=confidence,
            elapsed=elapsed,
            action=action,
            relevance=relevance,
        )

    # -- helpers ---------------------------------------------------------

    def _diagnostics(self, elapsed: float, best_score: float, n_evidence: int) -> dict[str, object]:
        return {
            "elapsed_ms": round(elapsed * 1000, 2),
            "best_retrieval_score": round(best_score, 4),
            "evidence_chunks": n_evidence,
            "index_size": len(self.retriever.chunks),
        }

    def _reject(
        self,
        question: str,
        *,
        draft: str,
        verdict: Verdict,
        retrieved: tuple,
        claim_checks: tuple,
        confidence: float,
        elapsed: float,
        action: Action | None = None,
        note: str = "",
        relevance: float | None = None,
    ) -> Answer:
        resolved = action or (
            Action.ESCALATE if self.policy.escalate_to_human else Action.REFUSE
        )
        text = (
            self.policy.escalation_text
            if resolved is Action.ESCALATE
            else self.policy.refusal_text
        )
        diagnostics = self._diagnostics(
            elapsed,
            retrieved[0].score if retrieved else 0.0,
            len(retrieved),
        )
        diagnostics["tags"] = list(self.policy.escalation_tags)
        if relevance is not None:
            diagnostics["relevance"] = round(relevance, 4)
        if note:
            diagnostics["note"] = note
        failed = [c.claim for c in claim_checks if not c.supported]
        if failed:
            diagnostics["rejected_claims"] = failed
        return Answer(
            question=question,
            action=resolved,
            verdict=verdict,
            text=text,
            draft=draft,
            citations=(),
            claim_checks=claim_checks,
            retrieved=retrieved,
            confidence=confidence,
            diagnostics=diagnostics,
        )

    def audit_log(self, answer: Answer) -> str:
        """One-line JSON record for an append-only audit trail."""
        return json.dumps(answer.to_dict(), ensure_ascii=False, sort_keys=True)
