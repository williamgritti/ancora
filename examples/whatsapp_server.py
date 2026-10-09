"""A complete WhatsApp bot with a grounding gate, on the standard library.

    export ANCORA_KB=./examples/knowledge-base
    export ANTHROPIC_API_KEY=...            # or run with the echo provider
    python examples/whatsapp_server.py

Point your Evolution API instance's webhook at http://<host>:8080/webhook
for the `messages.upsert` event.

Deliberately stdlib-only. Swapping http.server for FastAPI is three lines
and is the reader's choice; the point here is that the whole path — parse,
gate, reply or hand off — fits on one screen and needs no framework.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from ancora import Ancora, EchoProvider, Policy
from ancora.integrations import (
    LoggingHandoff,
    WhatsAppHandler,
    mask_identifier,
    parse_evolution_webhook,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

KB_PATH = os.environ.get("ANCORA_KB", "./examples/knowledge-base")
EVOLUTION_URL = os.environ.get("EVOLUTION_API_URL", "").rstrip("/")
EVOLUTION_KEY = os.environ.get("EVOLUTION_API_KEY", "")
EVOLUTION_INSTANCE = os.environ.get("EVOLUTION_INSTANCE", "")
PORT = int(os.environ.get("PORT", "8080"))


def build_provider():
    """Anthropic when a key is present, otherwise the offline echo provider."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from ancora import AnthropicProvider

        log.info("provider: anthropic")
        return AnthropicProvider()
    log.info("provider: echo (no ANTHROPIC_API_KEY set)")
    log.warning(
        "echo replies quote the whole source chunk verbatim. Grounded by "
        "construction, but too verbose for a real customer — set a real "
        "provider before pointing this at a live number."
    )
    return EchoProvider()


def send_whatsapp(to: str, text: str) -> None:
    """Send a reply through Evolution API. No-op when unconfigured."""
    if not (EVOLUTION_URL and EVOLUTION_KEY and EVOLUTION_INSTANCE):
        log.info("would send to %s: %s", mask_identifier(to), text)
        return
    request = urllib.request.Request(
        f"{EVOLUTION_URL}/message/sendText/{EVOLUTION_INSTANCE}",
        data=json.dumps({"number": to, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json", "apikey": EVOLUTION_KEY},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            log.info("sent to %s", mask_identifier(to))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Never log the request: it carries the apikey header.
        log.error("send failed (%s)", type(exc).__name__)


engine = Ancora.from_directory(
    KB_PATH,
    policy=Policy(
        escalate_to_human=True,
        escalation_text=(
            "Deixa eu confirmar isso com um atendente para não te passar "
            "informação errada. Já te retorno."
        ),
    ),
)
handler = WhatsAppHandler(
    engine,
    build_provider(),
    handoff=LoggingHandoff(),
    cite_sources=False,
)
log.info("knowledge base: %s chunks from %s", len(engine.retriever.chunks), KB_PATH)


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/") != "/webhook":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return

        # Acknowledge immediately: Evolution retries on a slow webhook, and a
        # retry storm turns one customer question into several replies.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

        try:
            reply = handler.handle_webhook(payload)
        except Exception:  # noqa: BLE001 - a bad message must not kill the server
            log.exception("handler raised")
            return

        if reply is not None and reply.should_send:
            inbound = parse_evolution_webhook(payload)
            if inbound is not None:
                send_whatsapp(inbound.sender, reply.text)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ancora whatsapp bot: ok")

    def log_message(self, fmt: str, *args: object) -> None:
        log.debug(fmt, *args)  # keep the default access log out of stdout


if __name__ == "__main__":
    log.info("listening on :%s/webhook", PORT)
    HTTPServer(("", PORT), WebhookHandler).serve_forever()
