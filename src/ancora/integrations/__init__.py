"""Production adapters.

Each adapter is framework-agnostic and dependency-free: it parses a payload
into a plain request, hands it to the engine, and returns a plain decision.
Wiring that to an HTTP framework is three lines and stays the caller's
choice, because a guardrail library has no business owning your web stack.
"""

from .chatwoot import ChatwootHandoff, LoggingHandoff
from .whatsapp import (
    HandoffSink,
    InboundMessage,
    Reply,
    WhatsAppHandler,
    mask_identifier,
    mask_secret,
    parse_evolution_webhook,
)

__all__ = [
    "ChatwootHandoff",
    "HandoffSink",
    "InboundMessage",
    "LoggingHandoff",
    "Reply",
    "WhatsAppHandler",
    "mask_identifier",
    "mask_secret",
    "parse_evolution_webhook",
]
