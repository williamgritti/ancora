"""LLM provider protocol plus adapters.

Ancora never imports an SDK at module scope. The protocol is three lines,
the adapters are thin, and the deterministic ``ScriptedProvider`` lets the
whole pipeline be tested without a network call or an API key.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, runtime_checkable

__all__ = [
    "Provider",
    "ScriptedProvider",
    "EchoProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "build_prompt",
]

_BLOCK_MARKER_RE = re.compile(r"(?:^|\n)\[\d+\]\s*")

DEFAULT_SYSTEM_PROMPT = (
    "You answer strictly from the CONTEXT below. "
    "If the context does not contain the answer, reply exactly: INSUFICIENTE. "
    "Never invent numbers, prices, deadlines, names or policies. "
    "Quote figures exactly as they appear in the context. "
    "Answer in the same language as the question. Be brief."
)


def build_prompt(question: str, context_blocks: Sequence[str]) -> str:
    """Assemble the user-side prompt from question and evidence."""
    numbered = "\n\n".join(
        f"[{i + 1}] {block.strip()}" for i, block in enumerate(context_blocks)
    )
    return (
        f"CONTEXT:\n{numbered or '(no context retrieved)'}\n\n"
        f"QUESTION: {question.strip()}\n\n"
        "ANSWER (from context only):"
    )


@runtime_checkable
class Provider(Protocol):
    """Anything that turns a prompt into text."""

    def complete(self, prompt: str, *, system: str = "") -> str:
        ...


class ScriptedProvider:
    """Returns canned responses in order. For tests and demos."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self._calls: list[tuple[str, str]] = []

    @property
    def calls(self) -> list[tuple[str, str]]:
        return list(self._calls)

    def complete(self, prompt: str, *, system: str = "") -> str:
        self._calls.append((system, prompt))
        if not self._responses:
            return "INSUFICIENTE"
        return self._responses.pop(0)


class EchoProvider:
    """Returns the highest-ranked context block verbatim.

    Useful as a zero-cost baseline: it is grounded by construction, so any
    failure of the gate against it is a bug in the gate.

    Leading markdown headings are dropped. A heading is a topic label, not an
    answer, and echoing "# Entregas e Frete" at a customer who asked about
    shipping costs is not a useful baseline.
    """

    @staticmethod
    def _strip_leading_headings(text: str) -> str:
        lines = text.splitlines()
        while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
            lines.pop(0)
        return "\n".join(lines).strip() or text.strip()

    def complete(self, prompt: str, *, system: str = "") -> str:
        body = prompt.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0].strip()
        # Split on the "[n]" markers, not on blank lines: a chunk's own text
        # contains blank lines, so splitting on those returns whatever comes
        # before the first one — typically a markdown heading, not a fact.
        blocks = _BLOCK_MARKER_RE.split(body)
        first = next((b.strip() for b in blocks if b.strip()), "")
        return self._strip_leading_headings(first)


class OpenAIProvider:
    """Adapter for the ``openai`` SDK. Imported lazily."""

    def __init__(self, model: str = "gpt-4o-mini", client: object | None = None, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        if client is None:
            try:
                from openai import OpenAI  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "OpenAIProvider needs the openai package: pip install 'ancora[openai]'"
                ) from exc
            client = OpenAI()
        self.client = client

    def complete(self, prompt: str, *, system: str = "") -> str:
        response = self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.kwargs.get("temperature", 0.0),
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicProvider:
    """Adapter for the ``anthropic`` SDK. Imported lazily."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", client: object | None = None, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        if client is None:
            try:
                import anthropic  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "AnthropicProvider needs the anthropic package: pip install 'ancora[anthropic]'"
                ) from exc
            client = anthropic.Anthropic()
        self.client = client

    def complete(self, prompt: str, *, system: str = "") -> str:
        response = self.client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=int(self.kwargs.get("max_tokens", 1024)),
            temperature=float(self.kwargs.get("temperature", 0.0)),
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        return "".join(parts).strip()
