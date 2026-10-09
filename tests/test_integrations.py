"""Tests for the Evolution API / Chatwoot adapter.

Everything here runs without a network: the handoff sink is a protocol, so
the whole path is exercised with an in-memory double.
"""

import logging

import pytest

from ancora import Ancora, ScriptedProvider
from ancora.integrations import (
    InboundMessage,
    LoggingHandoff,
    WhatsAppHandler,
    mask_identifier,
    mask_secret,
    parse_evolution_webhook,
)
from ancora.types import Action

KB = {
    "entregas": "O prazo de entrega padrao e de 5 dias uteis. O frete custa R$ 24,90.",
}


def webhook(text, *, from_me=False, jid="5551999998888@s.whatsapp.net", event="messages.upsert"):
    return {
        "event": event,
        "instance": "loja",
        "data": {
            "key": {"remoteJid": jid, "fromMe": from_me, "id": "MSG1"},
            "pushName": "Cliente",
            "message": {"conversation": text},
        },
    }


@pytest.fixture
def kb():
    return Ancora.from_texts(KB)


class TestParsing:
    def test_plain_conversation(self):
        m = parse_evolution_webhook(webhook("qual o prazo?"))
        assert m.text == "qual o prazo?"
        assert m.sender_name == "Cliente"
        assert m.is_actionable

    def test_own_message_ignored(self):
        """The loop every first integration accidentally creates."""
        assert parse_evolution_webhook(webhook("eco", from_me=True)) is None

    def test_group_message_not_actionable(self):
        m = parse_evolution_webhook(webhook("oi", jid="12345@g.us"))
        assert m.is_group and not m.is_actionable

    def test_non_message_event_ignored(self):
        assert parse_evolution_webhook(webhook("x", event="messages.update")) is None

    def test_data_as_list(self):
        payload = webhook("qual o prazo?")
        payload["data"] = [payload["data"]]
        assert parse_evolution_webhook(payload).text == "qual o prazo?"

    @pytest.mark.parametrize(
        "message,expected",
        [
            ({"conversation": "a"}, "a"),
            ({"extendedTextMessage": {"text": "b"}}, "b"),
            ({"imageMessage": {"caption": "c"}}, "c"),
            ({"videoMessage": {"caption": "d"}}, "d"),
            ({"documentMessage": {"caption": "e"}}, "e"),
            ({"buttonsResponseMessage": {"selectedDisplayText": "f"}}, "f"),
            ({"listResponseMessage": {"title": "g"}}, "g"),
        ],
    )
    def test_message_shapes(self, message, expected):
        payload = webhook("")
        payload["data"]["message"] = message
        assert parse_evolution_webhook(payload).text == expected

    def test_unknown_shape_yields_empty_not_crash(self):
        payload = webhook("")
        payload["data"]["message"] = {"stickerMessage": {"url": "x"}}
        assert parse_evolution_webhook(payload).text == ""

    @pytest.mark.parametrize("payload", [None, "string", [], {}, {"data": None}])
    def test_malformed_payloads_return_none(self, payload):
        assert parse_evolution_webhook(payload) is None

    def test_missing_jid_rejected(self):
        payload = webhook("x")
        payload["data"]["key"]["remoteJid"] = ""
        assert parse_evolution_webhook(payload) is None


class TestHandler:
    def test_grounded_answer_is_sent(self, kb):
        handler = WhatsAppHandler(kb, ScriptedProvider(["O prazo de entrega e de 5 dias uteis."]))
        reply = handler.handle_webhook(webhook("qual o prazo?"))
        assert reply.action is Action.ANSWER
        assert reply.should_send
        assert not reply.handoff

    def test_hallucination_triggers_handoff(self, kb):
        sink = LoggingHandoff()
        handler = WhatsAppHandler(kb, ScriptedProvider(["O prazo e de 3 dias uteis."]), handoff=sink)
        reply = handler.handle_webhook(webhook("qual o prazo?"))
        assert reply.action is Action.ESCALATE
        assert reply.handoff
        assert len(sink.escalations) == 1

    def test_customer_never_sees_the_blocked_draft(self, kb):
        sink = LoggingHandoff()
        handler = WhatsAppHandler(kb, ScriptedProvider(["O prazo e de 3 dias uteis."]), handoff=sink)
        reply = handler.handle_webhook(webhook("qual o prazo?"))
        assert "3 dias" not in reply.text
        assert "3 dias" in reply.answer.draft, "but the agent must still see it"

    def test_without_a_sink_it_refuses(self, kb):
        handler = WhatsAppHandler(kb, ScriptedProvider(["O prazo e de 3 dias uteis."]))
        reply = handler.handle(parse_evolution_webhook(webhook("qual o prazo?")))
        assert reply.action is Action.REFUSE
        assert reply.text == kb.policy.refusal_text

    def test_sources_appended_when_configured(self, kb):
        handler = WhatsAppHandler(
            kb, ScriptedProvider(["O prazo de entrega e de 5 dias uteis."]), cite_sources=True
        )
        assert "Fonte:" in handler.handle_webhook(webhook("qual o prazo?")).text

    def test_empty_message_sends_nothing(self, kb):
        handler = WhatsAppHandler(kb, ScriptedProvider(["x"]))
        reply = handler.handle(InboundMessage(text="   ", sender="5551@s.whatsapp.net"))
        assert not reply.should_send

    def test_group_message_sends_nothing(self, kb):
        handler = WhatsAppHandler(kb, ScriptedProvider(["x"]))
        assert not handler.handle_webhook(webhook("oi", jid="1@g.us")).should_send

    def test_non_message_webhook_returns_none(self, kb):
        handler = WhatsAppHandler(kb, ScriptedProvider(["x"]))
        assert handler.handle_webhook(webhook("x", event="presence.update")) is None

    def test_on_answer_hook_receives_every_decision(self, kb):
        seen = []
        handler = WhatsAppHandler(
            kb,
            ScriptedProvider(["O prazo e de 3 dias uteis."]),
            handoff=LoggingHandoff(),
            on_answer=lambda m, a: seen.append((m.text, a.verdict.value)),
        )
        handler.handle_webhook(webhook("qual o prazo?"))
        assert seen == [("qual o prazo?", "unsafe_numeric")]


class TestSecretHygiene:
    @pytest.mark.parametrize(
        "value,expected",
        [(None, "(unset)"), ("", "(unset)"), ("abcd", "****"), ("secrettoken", "*******oken")],
    )
    def test_mask_secret(self, value, expected):
        assert mask_secret(value) == expected

    def test_mask_identifier_keeps_useful_digits(self):
        masked = mask_identifier("5551999998888@s.whatsapp.net")
        assert masked.endswith("8888@s.whatsapp.net")
        assert "5551" not in masked

    def test_mask_identifier_handles_bare_value(self):
        assert mask_identifier("12345").endswith("2345")

    def test_mask_identifier_empty(self):
        assert mask_identifier("") == "(unknown)"

    def test_token_never_appears_in_logs(self, kb, caplog):
        sink = LoggingHandoff()
        handler = WhatsAppHandler(kb, ScriptedProvider(["O prazo e de 3 dias uteis."]), handoff=sink)
        with caplog.at_level(logging.DEBUG):
            handler.handle_webhook(webhook("qual o prazo?"))
        assert "5551999998888" not in caplog.text, "raw phone number leaked into logs"

    def test_chatwoot_requires_configuration(self, monkeypatch):
        from ancora.integrations import ChatwootHandoff

        for var in ("CHATWOOT_BASE_URL", "CHATWOOT_ACCOUNT_ID", "CHATWOOT_API_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError) as exc:
            ChatwootHandoff()
        assert "CHATWOOT_API_TOKEN" in str(exc.value)

    def test_chatwoot_repr_masks_the_token(self):
        from ancora.integrations import ChatwootHandoff

        client = ChatwootHandoff(
            base_url="https://app.example.com",
            account_id="1",
            api_token="supersecrettoken123",
        )
        assert "supersecrettoken123" not in repr(client)
        assert "n123" in repr(client)


class TestChatwootNote:
    def test_note_carries_the_draft_and_the_reason(self, kb):
        from ancora.integrations import ChatwootHandoff

        answer = kb.ask("qual o prazo?", provider=ScriptedProvider(["O prazo e de 3 dias uteis."]))
        note = ChatwootHandoff._build_note(
            InboundMessage(text="qual o prazo?", sender="5551@s.whatsapp.net"), answer
        )
        assert "3 dias" in note
        assert "unsafe_numeric" in note
        assert "qual o prazo?" in note
