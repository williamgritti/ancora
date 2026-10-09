"""Tokenisation, normalisation and number extraction.

Deliberately dependency-free and deterministic: the same input always
produces the same tokens, which is what makes the grounding gate testable
without a model in the loop.

Handles Portuguese, Spanish, Italian and English out of the box, which
covers the accented-Latin cases where naive ``str.split()`` silently
breaks retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

__all__ = [
    "normalize",
    "tokenize",
    "split_sentences",
    "extract_numbers",
    "canonical_number",
    "stem",
    "STOPWORDS",
]

# Multilingual stopword set. Kept small on purpose: over-aggressive stopword
# removal destroys short factual claims like "o prazo e 5 dias".
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have he her his in is it its of on
    or that the their they this to was were will with you your we our i not
    o os as um uma uns umas de do da dos das em no na nos nas por para com
    que e ou se ao aos que qual quais como quando onde ser estar tem ter foi
    era sao sua seu suas seus mais mas ja nao sim pelo pela pelos pelas
    el la los las un una unos unas del al es son por con para
    il lo gli le dei delle degli nel nella con per che sono e' della
    """.split()
)

_WORD_RE = re.compile(r"[0-9a-z]+(?:[.,][0-9]+)*", re.IGNORECASE)

# Sentence boundaries: a terminator followed by whitespace and a capital or
# digit. Guards against splitting on decimals ("R$ 1.500,00") and on common
# abbreviations.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀ-ÖØ-Þ0-9])")

_ABBREVIATIONS = {
    "art", "arts", "inc", "ltda", "sr", "sra", "dr", "dra", "prof", "n",
    "no", "num", "pag", "fl", "fls", "cf", "ex", "etc", "vs", "p", "pp",
}

# Numbers with optional thousands separators, decimals, percent or currency.
# Matches: 1.500,00  1,500.00  R$ 250  30%  5  0.75  12h
_NUMBER_RE = re.compile(
    r"""
    (?<![\w])                       # not mid-word
    (?:R\$\s*|US\$\s*|\$\s*|€\s*)?  # optional currency prefix
    (\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?   # 1.500.000,25  (pt-BR / it)
     |\d{1,3}(?:,\d{3})+(?:\.\d+)?     # 1,500,000.25  (en)
     |\d+(?:[.,]\d+)?)                 # 5  0.75  3,5
    \s*(%|por\s*cento|percent)?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace.

    Accent folding means "prazo" and "práZo" collide, which is what we want
    for matching, while the original text is always preserved elsewhere for
    display and citation offsets.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.lower()).strip()


# Suffix rules for light stemming, longest first. The goal is *consistency*
# between query and document, not linguistic correctness: as long as both
# sides collapse the same way, retrieval matches. Deliberately small, because
# over-stemming destroys precision faster than under-stemming costs recall.
_SUFFIX_RULES: tuple[tuple[str, int], ...] = (
    ("amento", 6), ("imento", 6),   # pt/it nominalisations
    ("acoes", 5), ("aveis", 5),
    ("ando", 5), ("endo", 5), ("indo", 5),   # pt gerunds
    ("ing", 5), ("ed", 5),                    # en
    ("oes", 4), ("aes", 4), ("eis", 4),
    ("ar", 5), ("er", 5), ("ir", 5),          # pt/es infinitives
    ("am", 5), ("em", 5),                     # pt 3rd person plural
    ("es", 4), ("as", 4), ("os", 4),
    ("s", 4),                                  # generic plural
)

_MIN_STEM = 3


def stem(token: str) -> str:
    """Strip one common inflectional suffix, guarding a minimum stem length.

    Tokens containing digits are returned untouched: stemming a price would
    corrupt the numeric guard.
    """
    if not token or any(ch.isdigit() for ch in token):
        return token
    for suffix, min_len in _SUFFIX_RULES:
        if len(token) >= min_len and token.endswith(suffix):
            candidate = token[: -len(suffix)]
            if len(candidate) >= _MIN_STEM:
                return candidate
    return token


def tokenize(text: str, *, drop_stopwords: bool = True, stemming: bool = True) -> list[str]:
    """Split into normalised word tokens.

    Numbers keep their internal separators so ``1.500,00`` survives as one
    token rather than shattering into ``1``/``500``/``00``, and are never
    stemmed.

    Light stemming is on by default so that "sabado"/"sabados" and
    "entrega"/"entregas"/"entregam" match, which is the difference between
    finding the right chunk and silently returning nothing.
    """
    tokens = _WORD_RE.findall(normalize(text))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def _looks_like_abbreviation(fragment: str) -> bool:
    tail = fragment.rstrip()
    if not tail.endswith("."):
        return False
    last = re.split(r"[\s(]", tail[:-1])[-1]
    return normalize(last) in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping abbreviations and decimals intact."""
    if not text or not text.strip():
        return []

    raw = _SENT_SPLIT_RE.split(text.strip())
    merged: list[str] = []
    for part in raw:
        part = part.strip()
        if not part:
            continue
        if merged and _looks_like_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)

    # Newline-separated lines (bullet lists) are their own sentences even
    # without terminal punctuation.
    out: list[str] = []
    for sentence in merged:
        for line in sentence.split("\n"):
            line = line.strip(" \t-•*")
            if line:
                out.append(line)
    return out


def canonical_number(raw: str) -> str | None:
    """Reduce a number token to a comparable canonical form.

    ``R$ 1.500,00``, ``1500``, ``1,500.00`` and ``1500.0`` all collapse to
    ``1500``. Returns ``None`` when the token cannot be parsed as a number.

    The heuristic for separators: if both ``.`` and ``,`` appear, the *last*
    one is the decimal separator. If only one appears, it is a decimal
    separator when it is followed by exactly one or two digits and the
    integer part is short, otherwise a thousands separator.
    """
    if raw is None:
        return None
    text = re.sub(r"[^\d.,]", "", str(raw)).strip()
    if not text or not any(ch.isdigit() for ch in text):
        return None

    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        decimal_sep = "." if text.rfind(".") > text.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma:
        head, _, tail = text.rpartition(",")
        text = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else text.replace(",", "")
    elif has_dot:
        head, _, tail = text.rpartition(".")
        # "1.500" is one thousand five hundred; "1.5" is one point five.
        text = text.replace(".", "") if len(tail) == 3 and head else f"{head}.{tail}"

    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    normalized = value.normalize()
    # Avoid scientific notation for large round numbers (1E+3 -> 1000).
    as_str = format(normalized, "f")
    return as_str.rstrip("0").rstrip(".") if "." in as_str else as_str


def extract_numbers(text: str) -> list[str]:
    """Return canonical forms of every number in ``text``, order preserved.

    Percentages are suffixed with ``%`` so that "30" and "30%" are not
    treated as the same claim.
    """
    found: list[str] = []
    for match in _NUMBER_RE.finditer(text or ""):
        canonical = canonical_number(match.group(1))
        if canonical is None:
            continue
        if match.group(2):
            canonical += "%"
        found.append(canonical)
    return found
