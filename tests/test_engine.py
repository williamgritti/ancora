import json

import pytest

from ancora import Ancora, EchoProvider, Policy, ScriptedProvider
from ancora.types import Action, Verdict

KB = {
    "faq": (
        "O prazo de entrega padrao e de 5 dias uteis para todo o Brasil. "
        "O frete custa R$ 24,90 e e gratis acima de R$ 199,00. "
        "Trocas podem ser solicitadas em ate 7 dias corridos."
    )
}


@pytest.fixture
def kb():
    return Ancora.from_texts(KB)


class TestHappyPath:
    def test_echo_provider_is_grounded_by_construction(self, kb):
        answer = kb.ask("quanto custa o frete?", provider=EchoProvider())
        assert answer.action is Action.ANSWER
        assert answer.verdict is Verdict.GROUNDED
        assert answer.citations

    def test_correct_answer_passes(self, kb):
        provider = ScriptedProvider(["O prazo de entrega e de 5 dias uteis."])
        answer = kb.ask("qual o prazo?", provider=provider)
        assert answer.is_answerable
        assert answer.confidence > 0

    def test_sources_are_reported(self, kb):
        answer = kb.ask("qual o prazo?", provider=EchoProvider())
        assert answer.sources() == ["faq"]


class TestRejection:
    def test_wrong_number_escalates(self, kb):
        provider = ScriptedProvider(["O prazo de entrega e de 3 dias uteis."])
        answer = kb.ask("qual o prazo?", provider=provider)
        assert answer.action is Action.ESCALATE
        assert answer.verdict is Verdict.UNSAFE_NUMERIC

    def test_draft_preserved_for_the_human(self, kb):
        draft = "O prazo de entrega e de 3 dias uteis."
        answer = kb.ask("qual o prazo?", provider=ScriptedProvider([draft]))
        assert answer.draft == draft
        assert answer.text != draft

    def test_fabricated_policy_escalates(self, kb):
        provider = ScriptedProvider(["Oferecemos garantia vitalicia em todos os produtos."])
        answer = kb.ask("qual o prazo?", provider=provider)
        assert not answer.is_answerable

    def test_model_declining_is_honoured(self, kb):
        answer = kb.ask("qual o prazo?", provider=ScriptedProvider(["INSUFICIENTE"]))
        assert answer.verdict is Verdict.NO_EVIDENCE
        assert answer.action is Action.ESCALATE

    def test_unretrievable_question_escalates(self, kb):
        answer = kb.ask("qual a capital da Mongolia?", provider=EchoProvider())
        assert not answer.is_answerable
        assert answer.verdict is Verdict.NO_EVIDENCE

    def test_refuse_mode_instead_of_escalate(self):
        kb = Ancora.from_texts(KB, policy=Policy(escalate_to_human=False))
        answer = kb.ask("qual o prazo?", provider=ScriptedProvider(["O prazo e de 3 dias."]))
        assert answer.action is Action.REFUSE
        assert answer.text == kb.policy.refusal_text

    def test_rejected_claims_listed_in_diagnostics(self, kb):
        answer = kb.ask("qual o prazo?", provider=ScriptedProvider(["O prazo e de 3 dias uteis."]))
        assert answer.diagnostics.get("rejected_claims")


class TestConstructors:
    def test_from_directory(self, tmp_path):
        (tmp_path / "faq.md").write_text(KB["faq"], encoding="utf-8")
        (tmp_path / "ignore.png").write_bytes(b"\x89PNG")
        kb = Ancora.from_directory(tmp_path)
        assert kb.ask("qual o prazo?", provider=EchoProvider()).is_answerable

    def test_from_directory_rejects_empty(self, tmp_path):
        with pytest.raises(ValueError):
            Ancora.from_directory(tmp_path)

    def test_from_directory_rejects_file(self, tmp_path):
        f = tmp_path / "x.md"
        f.write_text("hi", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            Ancora.from_directory(f)

    def test_custom_chunks_accepted(self):
        from ancora.chunking import chunk_documents
        from ancora.types import Document

        chunks = chunk_documents([Document(id="d", text=KB["faq"])])
        assert Ancora(chunks).ask("frete?", provider=EchoProvider()).is_answerable


class TestAudit:
    def test_to_dict_is_json_serialisable(self, kb):
        answer = kb.ask("qual o prazo?", provider=EchoProvider())
        assert json.loads(json.dumps(answer.to_dict(), ensure_ascii=False))

    def test_audit_log_is_one_line(self, kb):
        answer = kb.ask("qual o prazo?", provider=EchoProvider())
        assert "\n" not in kb.audit_log(answer)

    def test_diagnostics_present(self, kb):
        d = kb.ask("qual o prazo?", provider=EchoProvider()).diagnostics
        assert "elapsed_ms" in d and "index_size" in d


class TestValidation:
    def test_empty_question_rejected(self, kb):
        with pytest.raises(ValueError):
            kb.ask("", provider=EchoProvider())

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            Ancora.from_texts(KB, top_k=0)

    def test_invalid_policy(self):
        with pytest.raises(ValueError):
            Policy(max_unsupported_claims=-1)


class TestEchoProviderHeadings:
    def test_leading_markdown_heading_is_dropped(self):
        """A heading is a topic label, not an answer."""
        from ancora import EchoProvider

        kb = Ancora.from_texts({"faq": "# Entregas e Frete\n\nO frete custa R$ 24,90."})
        answer = kb.ask("quanto custa o frete?", provider=EchoProvider())
        assert not answer.text.lstrip().startswith("#")
        assert "24,90" in answer.text

    def test_heading_only_chunk_falls_back_to_the_heading(self):
        from ancora import EchoProvider

        assert EchoProvider()._strip_leading_headings("# Só um título") == "# Só um título"
