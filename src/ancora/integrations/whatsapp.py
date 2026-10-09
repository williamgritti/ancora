"""Evolution API inbound handling with a Chatwoot-style human handoff.

The shape of a support bot that does not embarrass its operator:

    inbound message
        -> Ancora gate
            -> grounded  : reply on WhatsApp, with sources if configured
            -> otherwise : stay silent to the customer, open a handoff, and
                           attach the blocked draft for the human agent

The draft is attached deliberately. A blocked answer is usually 90% correct
and the agent only has to fix one figure; making them retype it from nothing
is how an escalation path stops being used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..engine import Ancora
from ..providers import Provider
from ..types import Action, Answer

__all__ = [
    "InboundMessage",
    "Reply",
    "HandoffSink",
    "WhatsAppHandler",
    "parse_evolution_webhook",
    "mask_secret",
    "mask_identifier",
]

logger = logging.getLogger("ancora.whatsapp")


def mask_secret(value: str | None, *, keep: int = 4) -> str:
    """Render a token safe to log.

    Adapters get configured with API keys, and the natural place to print one
    is a debug log that later ends up in a screenshot or a support ticket.
    Everything in this module that touches a credential goes through here.
    """
    if not value:
        return "(unset)"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{'*' * (len(value) - keep)}{value[-keep:]}"


def mask_identifier(jid: str | None, *, keep: int = 4) -> str:
    """Mask a channel identifier for logs, keeping it useful for support.

    A WhatsApp JID is ``5551999998888@s.whatsapp.net``. Masking the last N
    characters of the whole string keeps ``pp.net``, which identifies
    nothing. The digits are what an operator needs to correlate a log line
    with a conversation, so the local part is masked and its tail kept.
    """
    if not jid:
        return "(unknown)"
    local, _, domain = jid.partition("@")
    tail = local[-keep:] if len(local) > keep else local
    masked = f"{'*' * max(len(local) - len(tail), 0)}{tail}"
    return f"{masked}@{domain}" if domain else masked


@dataclass(frozen=True)
class InboundMessage:
    """One customer message, normalised out of a provider payload."""

    text: str
    sender: str
    """Opaque channel identifier. Treated as a routing key, never parsed."""

    instance: str = ""
    message_id: str = ""
    sender_name: str = ""
    is_group: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_actionable(self) -> bool:
        """Whether this message should reach the engine at all."""
        return bool(self.text.strip()) and not self.is_group


@dataclass(frozen=True)
class Reply:
    """What the handler decided to do."""

    action: Action
    text: str
    """Empty when nothing should be sent to the customer."""

    handoff: bool = False
    answer: Answer | None = None

    @property
    def should_send(self) -> bool:
        return bool(self.text.strip())


@runtime_checkable
class HandoffSink(Protocol):
    """Anything that can put a conversation in front of a human.

    Chatwoot, a helpdesk, a Slack channel, a database table. The handler does
    not care, which keeps the integration testable without a network.
    """

    def open(self, *, message: InboundMessage, answer: Answer) -> None:
        ...


def _extract_text(message: dict[str, Any]) -> str:
    """Pull user text out of Evolution's several message shapes."""
    if not isinstance(message, dict):
        return ""
    if isinstance(message.get("conversation"), str):
        return message["conversation"]
    for key, field_name in (
        ("extendedTextMessage", "text"),
        ("imageMessage", "caption"),
        ("videoMessage", "caption"),
        ("documentMessage", "caption"),
    ):
        nested = message.get(key)
        if isinstance(nested, dict) and isinstance(nested.get(field_name), str):
            return nested[field_name]
    button = message.get("buttonsResponseMessage") or message.get("listResponseMessage")
    if isinstance(button, dict):
        for field_name in ("selectedDisplayText", "title"):
            if isinstance(button.get(field_name), str):
                return button[field_name]
    return ""


def parse_evolution_webhook(payload: dict[str, Any]) -> InboundMessage | None:
    """Normalise an Evolution API webhook body.

    Returns ``None`` for anything that is not an inbound customer message:
    status updates, delivery receipts, and the bot's own outgoing messages,
    which is the loop every first integration accidentally creates.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("event") not in (None, "messages.upsert", "MESSAGES_UPSERT"):
        return None

    data = payload.get("data")
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None

    key = data.get("key") if isinstance(data.get("key"), dict) else {}
    if key.get("fromMe"):
        return None  # our own message; replying to it would loop

    remote_jid = str(key.get("remoteJid") or "")
    if not remote_jid:
        return None

    text = _extract_text(data.get("message") or {})
    return InboundMessage(
        text=text,
        sender=remote_jid,
        instance=str(payload.get("instance") or ""),
        message_id=str(key.get("id") or ""),
        sender_name=str(data.get("pushName") or ""),
        is_group=remote_jid.endswith("@g.us"),
        raw=payload,
    )


class WhatsAppHandler:
    """Turns an inbound message into a decision, with the gate in the middle."""

    def __init__(
        self,
        engine: Ancora,
        provider: Provider,
        *,
        handoff: HandoffSink | None = None,
        cite_sources: bool = False,
        source_prefix: str = "Fonte: ",
        on_answer: Callable[[InboundMessage, Answer], None] | None = None,
    ) -> None:
        self.engine = engine
        self.provider = provider
        self.handoff = handoff
        self.cite_sources = cite_sources
        self.source_prefix = source_prefix
        self.on_answer = on_answer

    def handle(self, message: InboundMessage) -> Reply:
        """Run one message through the gate."""
        if not message.is_actionable:
            return Reply(action=Action.REFUSE, text="")

        answer = self.engine.ask(message.text, provider=self.provider)

        if self.on_answer is not None:
            self.on_answer(message, answer)

        logger.info(
            "ancora decision sender=%s action=%s verdict=%s confidence=%.2f",
            mask_identifier(message.sender),
            answer.action.value,
            answer.verdict.value,
            answer.confidence,
        )

        if answer.action is Action.ANSWER:
            text = answer.text
            if self.cite_sources and answer.sources():
                text = f"{text}\n\n{self.source_prefix}{', '.join(answer.sources())}"
            return Reply(action=Action.ANSWER, text=text, answer=answer)

        if self.handoff is not None:
            self.handoff.open(message=message, answer=answer)
            return Reply(
                action=Action.ESCALATE,
                text=self.engine.policy.escalation_text,
                handoff=True,
                answer=answer,
            )

        return Reply(
            action=Action.REFUSE,
            text=self.engine.policy.refusal_text,
            answer=answer,
        )

    def handle_webhook(self, payload: dict[str, Any]) -> Reply | None:
        """Convenience: parse and handle in one call. ``None`` if not a message."""
        message = parse_evolution_webhook(payload)
        return self.handle(message) if message is not None else None
