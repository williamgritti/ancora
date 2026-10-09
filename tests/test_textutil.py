import pytest

from ancora.textutil import (
    canonical_number,
    extract_numbers,
    normalize,
    split_sentences,
    tokenize,
)


class TestNormalize:
    def test_strips_accents_and_case(self):
        assert normalize("PRAZO de Entrega É Ótimo") == "prazo de entrega e otimo"

    def test_collapses_whitespace(self):
        assert normalize("  a\n\tb   c  ") == "a b c"

    def test_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("São Paulo", "sao paulo"),
            ("Città", "citta"),
            ("naïve café", "naive cafe"),
            ("ÀÉÎÕÜ", "aeiou"),
        ],
    )
    def test_multilingual(self, text, expected):
        assert normalize(text) == expected


class TestTokenize:
    def test_drops_stopwords_by_default(self):
        assert tokenize("o prazo de entrega") == ["prazo", "entrega"]

    def test_keeps_stopwords_when_asked(self):
        assert "de" in tokenize("o prazo de entrega", drop_stopwords=False)

    def test_keeps_formatted_numbers_intact(self):
        # The critical case: a price must not shatter into 1/500/00
        assert "1.500,00" in tokenize("custa R$ 1.500,00")

    def test_empty(self):
        assert tokenize("") == []


class TestCanonicalNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1.500,00", "1500"),   # pt-BR / it-IT
            ("1,500.00", "1500"),   # en-US
            ("1500", "1500"),
            ("1500.0", "1500"),
            ("1.5", "1.5"),
            ("3,5", "3.5"),
            ("R$ 24,90", "24.9"),
            ("$1,234,567.89", "1234567.89"),
            ("0.75", "0.75"),
            ("1.000.000", "1000000"),
        ],
    )
    def test_canonicalisation(self, raw, expected):
        assert canonical_number(raw) == expected

    def test_cross_locale_equality(self):
        """The property the numeric guard depends on."""
        assert canonical_number("R$ 1.500,00") == canonical_number("$1,500.00")
        assert canonical_number("1.500,00") == canonical_number("1500")

    @pytest.mark.parametrize("raw", ["", "abc", None, "R$"])
    def test_non_numbers(self, raw):
        assert canonical_number(raw) is None


class TestExtractNumbers:
    def test_currency_and_percent(self):
        assert extract_numbers("Custa R$ 1.500,00 com 30% de desconto") == ["1500", "30%"]

    def test_english_equivalent_matches(self):
        assert extract_numbers("It costs $1,500.00 with 30 percent off") == ["1500", "30%"]

    def test_percent_distinct_from_bare_number(self):
        assert extract_numbers("30%") != extract_numbers("30")

    def test_order_preserved(self):
        assert extract_numbers("5 dias, 7 dias, 12 dias") == ["5", "7", "12"]

    def test_none_present(self):
        assert extract_numbers("sem numeros aqui") == []
        assert extract_numbers("") == []


class TestSplitSentences:
    def test_basic(self):
        out = split_sentences("Prazo e 5 dias. Frete e gratis. Trocas em 7 dias.")
        assert len(out) == 3

    def test_does_not_split_decimals(self):
        out = split_sentences("Custa R$ 1.500,00 no total.")
        assert len(out) == 1

    def test_does_not_split_abbreviations(self):
        out = split_sentences("Conforme o art. 5 da lei aplicavel.")
        assert len(out) == 1

    def test_bullets_become_sentences(self):
        out = split_sentences("Regras:\n- prazo de 5 dias\n- frete gratis")
        assert len(out) == 3

    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []
