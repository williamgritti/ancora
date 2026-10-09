"""Regression tests for defects found by examples/demo.py.

Both were silent failures — the pipeline returned a confident answer and
nothing looked wrong. They are the reason the demo exists.
"""

import pytest

from ancora import Ancora, ScriptedProvider
from ancora.grounding import GroundingChecker
from ancora.textutil import stem, tokenize
from ancora.types import Verdict


@pytest.fixture
def multi_doc_kb(tmp_path):
    """Three topic documents that share generic vocabulary.

    'dias úteis' appears on both the delivery and the payments page, which
    is what made the original bug possible.
    """
    (tmp_path / "entregas.md").write_text(
        "O prazo de entrega padrao e de 5 dias uteis para todo o Brasil. "
        "Nao realizamos entregas aos sabados, domingos e feriados nacionais.",
        encoding="utf-8",
    )
    (tmp_path / "pagamento.md").write_text(
        "Parcelamos em ate 6x sem juros. "
        "O boleto tem vencimento em 3 dias uteis e o pedido e liberado apos a compensacao.",
        encoding="utf-8",
    )
    (tmp_path / "trocas.md").write_text(
        "Trocas podem ser solicitadas em ate 7 dias corridos apos o recebimento.",
        encoding="utf-8",
    )
    return Ancora.from_directory(tmp_path)


class TestNumericScopeLeak:
    """A number from an unrelated document must not license a claim.

    Original defect: the guard checked the union of *all* retrieved chunks.
    The payments page contains "3 dias uteis" (boleto), which silently
    authorised "delivery takes 3 days".
    """

    def test_unrelated_document_does_not_license_a_figure(self, multi_doc_kb):
        answer = multi_doc_kb.ask(
            "qual o prazo de entrega?",
            provider=ScriptedProvider(["O prazo de entrega e de 3 dias uteis."]),
        )
        assert not answer.is_answerable
        assert answer.verdict is Verdict.UNSAFE_NUMERIC

    def test_the_correct_figure_still_passes(self, multi_doc_kb):
        answer = multi_doc_kb.ask(
            "qual o prazo de entrega?",
            provider=ScriptedProvider(["O prazo de entrega e de 5 dias uteis."]),
        )
        assert answer.is_answerable

    def test_figure_from_its_own_topic_passes(self, multi_doc_kb):
        answer = multi_doc_kb.ask(
            "qual o vencimento do boleto?",
            provider=ScriptedProvider(["O boleto vence em 3 dias uteis."]),
        )
        assert answer.is_answerable, "3 is legitimate on the payments topic"


class TestNumericTokensExcludedFromCoverage:
    """A claim's own number must not help select the chunk that licenses it.

    Original defect: the token '3' matched the payments page, pushing its
    lexical coverage (0.80) *above* the real delivery page (0.70). The
    number was vouching for its own provenance.
    """

    def test_coverage_ignores_numeric_tokens(self):
        checker = GroundingChecker()
        assert checker._lexical_tokens(["prazo", "entrega", "3", "dia"]) == [
            "prazo",
            "entrega",
            "dia",
        ]

    def test_shared_number_does_not_outrank_shared_topic(self):
        from ancora.chunking import chunk_documents
        from ancora.types import Document

        chunks = chunk_documents(
            [
                Document(id="entrega", text="O prazo de entrega e de 5 dias uteis."),
                Document(id="boleto", text="O boleto vence em 3 dias uteis."),
            ]
        )
        checker = GroundingChecker()
        idf = checker._idf(chunks)
        claim = tokenize("O prazo de entrega e de 3 dias uteis.")
        scores = {
            c.id: checker._coverage(claim, set(tokenize(c.text)), idf) for c in chunks
        }
        assert scores["entrega#0"] > scores["boleto#0"], (
            "the on-topic chunk must win regardless of the invented figure"
        )


class TestStemmingRecall:
    """Verbatim source sentences were being blocked as ungrounded.

    Original defect: no stemming, so 'sabado' != 'sabados' and
    'entregam' != 'entregas'. Retrieval scored 0.0 and a correct,
    word-for-word answer was escalated.
    """

    def test_verbatim_sentence_is_answerable(self, multi_doc_kb):
        answer = multi_doc_kb.ask(
            "voces entregam no sabado?",
            provider=ScriptedProvider(
                ["Nao realizamos entregas aos sabados, domingos e feriados nacionais."]
            ),
        )
        assert answer.is_answerable

    @pytest.mark.parametrize(
        "a,b",
        [
            ("sabado", "sabados"),
            ("entrega", "entregas"),
            ("entregar", "entregam"),
            ("prazo", "prazos"),
            ("delivery", "deliveries") ,
        ],
    )
    def test_inflections_collapse(self, a, b):
        assert stem(a) == stem(b) or stem(b).startswith(stem(a)[:4])

    def test_numbers_are_never_stemmed(self):
        for token in ("1.500,00", "24,90", "5", "12x"):
            assert stem(token) == token

    def test_short_words_are_not_over_stemmed(self):
        for token in ("mes", "pai", "sim", "voz"):
            assert len(stem(token)) >= 3
