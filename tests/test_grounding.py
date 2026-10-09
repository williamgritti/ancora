"""Tests for the grounding gate — the heart of the library."""

import pytest

from ancora.chunking import chunk_documents
from ancora.grounding import GroundingChecker, is_checkable_claim
from ancora.types import Document

KB_TEXT = (
    "O prazo de entrega padrao e de 5 dias uteis para todo o Brasil. "
    "Para a regiao Norte o prazo e de 12 dias uteis. "
    "O frete custa R$ 24,90 e e gratis acima de R$ 199,00. "
    "Trocas podem ser solicitadas em ate 7 dias corridos apos o recebimento."
)


@pytest.fixture
def evidence():
    return chunk_documents(
        [Document(id="faq", text=KB_TEXT, metadata={"source": "FAQ"})],
        target_chars=500,
    )


@pytest.fixture
def checker():
    return GroundingChecker()


class TestNumericGuard:
    """The commercially important behaviour."""

    def test_correct_number_passes(self, checker, evidence):
        check = checker.check_claim("O prazo de entrega e de 5 dias uteis.", evidence)
        assert check.supported
        assert not check.unsupported_numbers

    def test_wrong_number_is_rejected_despite_high_lexical_support(self, checker, evidence):
        """The motivating case: 80% lexical overlap, wrong figure, must fail."""
        check = checker.check_claim("O prazo de entrega e de 3 dias uteis.", evidence)
        assert not check.supported
        assert "3" in check.unsupported_numbers
        # High lexical similarity is precisely why a similarity checker fails here.
        assert check.support_score > 0.6

    def test_invented_price_rejected(self, checker, evidence):
        check = checker.check_claim("O frete custa R$ 19,90.", evidence)
        assert not check.supported
        assert "19.9" in check.unsupported_numbers

    def test_multiple_invented_numbers_all_reported(self, checker, evidence):
        check = checker.check_claim(
            "O frete custa R$ 19,90 e parcelamos em 10x.", evidence
        )
        assert set(check.unsupported_numbers) == {"19.9", "10"}

    def test_number_present_elsewhere_in_kb_is_not_flagged(self, checker, evidence):
        """12 appears in the KB (Norte). Only the invented figure is flagged."""
        check = checker.check_claim(
            "O frete custa R$ 19,90 e o prazo Norte e de 12 dias.", evidence
        )
        assert set(check.unsupported_numbers) == {"19.9"}

    def test_number_from_a_different_chunk_still_counts(self, checker):
        """A correct figure quoted from a neighbouring chunk is not punished."""
        chunks = chunk_documents(
            [Document(id="d", text=KB_TEXT, metadata={})], target_chars=80
        )
        check = checker.check_claim("O prazo para o Norte e de 12 dias uteis.", chunks)
        assert not check.unsupported_numbers

    def test_guard_can_be_disabled(self, evidence):
        lenient = GroundingChecker(require_numeric_support=False)
        check = lenient.check_claim("O prazo de entrega e de 3 dias uteis.", evidence)
        assert check.unsupported_numbers == ()
        assert check.supported  # now passes on lexical support alone

    def test_currency_format_does_not_matter(self, checker):
        chunks = chunk_documents(
            [Document(id="d", text="The fee is $1,500.00 per year.", metadata={})]
        )
        check = checker.check_claim("A taxa e de R$ 1.500,00 por ano.", chunks)
        assert not check.unsupported_numbers


class TestLexicalSupport:
    def test_fabricated_policy_rejected(self, checker, evidence):
        check = checker.check_claim(
            "Oferecemos garantia estendida vitalicia para eletronicos.", evidence
        )
        assert not check.supported
        assert check.support_score < 0.5

    def test_paraphrase_accepted(self, checker, evidence):
        check = checker.check_claim("Trocas sao solicitadas em ate 7 dias corridos.", evidence)
        assert check.supported

    def test_hedged_claim_uses_lower_bar(self, evidence):
        """A hedged sentence asserts less, so it clears a lower bar.

        Both claims sit in the 0.50-0.57 coverage band; only the thresholds
        differ, which isolates the hedge mechanism itself.
        """
        strict = GroundingChecker(support_threshold=0.6, hedged_threshold=0.45)
        plain = strict.check_claim(
            "A entrega ocorre em dias uteis conforme a regiao contratada.", evidence
        )
        hedged = strict.check_claim(
            "Geralmente a entrega ocorre em dias uteis conforme a regiao contratada.",
            evidence,
        )
        assert not plain.supported, "plain claim should miss the strict bar"
        assert hedged.supported, "hedged claim should clear the relaxed bar"

    def test_supporting_chunk_ids_recorded(self, checker, evidence):
        check = checker.check_claim("O frete custa R$ 24,90.", evidence)
        assert check.supporting_chunk_ids


class TestClaimFiltering:
    @pytest.mark.parametrize(
        "text",
        [
            "Ola!", "Bom dia", "Obrigado!", "Posso ajudar com mais alguma coisa?",
            "Qual o seu CEP?", "Claro.", "Hello", "Thanks!",
        ],
    )
    def test_non_claims_skipped(self, text):
        assert not is_checkable_claim(text)

    @pytest.mark.parametrize(
        "text",
        ["O prazo e de 5 dias uteis.", "O frete custa R$ 24,90.", "Trocas em 7 dias."],
    )
    def test_real_claims_checked(self, text):
        assert is_checkable_claim(text)

    def test_empty_is_not_a_claim(self):
        assert not is_checkable_claim("")
        assert not is_checkable_claim("   ")


class TestCheckAnswer:
    def test_greeting_plus_fact_checks_only_the_fact(self, checker, evidence):
        checks = checker.check_answer(
            "Ola! O prazo de entrega e de 5 dias uteis. Posso ajudar em algo mais?",
            evidence,
        )
        assert len(checks) == 1
        assert checks[0].supported

    def test_mixed_answer_flags_only_the_bad_sentence(self, checker, evidence):
        checks = checker.check_answer(
            "O frete custa R$ 24,90. O prazo e de 3 dias uteis.", evidence
        )
        assert len(checks) == 2
        assert checks[0].supported
        assert not checks[1].supported

    def test_empty_answer(self, checker, evidence):
        assert checker.check_answer("", evidence) == ()

    def test_no_evidence_rejects_everything(self, checker):
        checks = checker.check_answer("O prazo e de 5 dias uteis.", [])
        assert all(not c.supported for c in checks)


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_invalid_thresholds_rejected(self, bad):
        with pytest.raises(ValueError):
            GroundingChecker(support_threshold=bad)
