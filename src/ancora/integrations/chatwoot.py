"""Chatwoot handoff sink, on the standard library only.

Uses ``urllib.request`` rather than ``requests`` so that installing Ancora
into a production bot still pulls in nothing. The surface is small enough
that the convenience of a client library does not pay for a dependency.

Credentials come from the environment, never from a constructor default, and
never reach a log line un-masked.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from ..types import Answer
from .whatsapp import InboundMessage, mask_secret

__all__ = ["ChatwootHandoff", "LoggingHandoff"]

logger = logging.getLogger("ancora.chatwoot")


class LoggingHandoff:
    """A sink that only logs. The right default for a first deployment.

    Running with this for a week tells an operator how often the gate fires
    and on which questions, which is the information needed to decide whether
    a human queue is warranted at all.
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        self.level = level
        self.escalations: list[tuple[InboundMessage, Answer]] = []

    def open(self, *, message: InboundMessage, answer: Answer) -> None:
        self.escalations.append((message, answer))
        logger.log(
            self.level,
            "handoff needed verdict=%s question=%r blocked_draft=%r",
            answer.verdict.value,
            message.text,
            answer.draft,
        )


class ChatwootHandoff:
    """Opens a Chatwoot conversation and attaches the blocked draft.

    The draft goes in as a *private note*, visible to the agent and never to
    the customer. That is the whole point: the agent sees what the model
    wanted to say, corrects the one wrong figure, and sends it.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        account_id: str | None = None,
        inbox_id: str | None = None,
        api_token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("CHATWOOT_BASE_URL", "")).rstrip("/")
        self.account_id = account_id or os.environ.get("CHATWOOT_ACCOUNT_ID", "")
        self.inbox_id = inbox_id or os.environ.get("CHATWOOT_INBOX_ID", "")
        self._token = api_token or os.environ.get("CHATWOOT_API_TOKEN", "")
        self.timeout = timeout

        missing = [
            name
            for name, value in (
                ("CHATWOOT_BASE_URL", self.base_url),
                ("CHATWOOT_ACCOUNT_ID", self.account_id),
                ("CHATWOOT_API_TOKEN", self._token),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "ChatwootHandoff is not configured; missing: " + ", ".join(missing)
            )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ChatwootHandoff(base_url={self.base_url!r}, "
            f"account_id={self.account_id!r}, token={mask_secret(self._token)})"
        )

    # -- transport -------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/api/v1/accounts/{self.account_id}/{path.lstrip('/')}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api_access_token": self._token,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload) if payload else {}

    # -- sink ------------------------------------------------------------

    def open(self, *, message: InboundMessage, answer: Answer) -> None:
        """Create the conversation and attach the audit note.

        Never raises into the caller: a failing helpdesk must not take the
        bot down with it. The failure is logged and the customer still gets
        the escalation text.
        """
        note = self._build_note(message, answer)
        try:
            conversation = self._post(
                "conversations",
                {
                    "source_id": message.sender,
                    "inbox_id": self.inbox_id,
                    "additional_attributes": {
                        "ancora_verdict": answer.verdict.value,
                        "ancora_confidence": round(answer.confidence, 3),
                    },
                },
            )
            conversation_id = conversation.get("id")
            if conversation_id is None:
                logger.error("chatwoot returned no conversation id; note not attached")
                return
            self._post(
                f"conversations/{conversation_id}/messages",
                {"content": note, "message_type": "outgoing", "private": True},
            )
            logger.info("handoff opened conversation=%s", conversation_id)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            # Deliberately not re-raised. Logged without the token.
            logger.error("chatwoot handoff failed (%s): %s", type(exc).__name__, exc)

    @staticmethod
    def _build_note(message: InboundMessage, answer: Answer) -> str:
        lines = [
            "**Resposta bloqueada pelo filtro de fundamentação**",
            "",
            f"Pergunta do cliente: {message.text}",
            f"Motivo: {answer.verdict.value}",
            "",
        ]
        if answer.draft:
            lines += [
                "Rascunho que o modelo produziu (não enviado):",
                f"> {answer.draft}",
                "",
            ]
        rejected = [c for c in answer.claim_checks if not c.supported]
        if rejected:
            lines.append("Trechos reprovados:")
            lines += [f"- {c.claim} — {c.reason}" for c in rejected]
            lines.append("")
        if answer.sources():
            lines.append(f"Fontes consultadas: {', '.join(answer.sources())}")
        lines.append("")
        lines.append("O rascunho costuma estar quase certo: confira o dado e envie.")
        return "\n".join(lines)
