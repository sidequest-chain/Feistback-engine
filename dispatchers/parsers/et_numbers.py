"""
Estonian Natural Language & Number Extraction Primitives
========================================================
Provides Estonian number word mapping, quantifier filtration, synthesis query
classification, and numerical operand extraction for Feistback dispatchers.
"""

from __future__ import annotations

import re
from typing import Dict, List


# Mapping of Estonian number words and common inflected forms to integers (0-10)
ESTONIAN_WORDS_TO_INT: Dict[str, int] = {
    "null": 0,
    "üks": 1, "ühe": 1,
    "kaks": 2, "kahe": 2,
    "kolm": 3, "kolme": 3,
    "neli": 4, "nelja": 4,
    "viis": 5, "viie": 5,
    "kuus": 6, "kuue": 6,
    "seitse": 7, "seitsme": 7,
    "kaheksa": 8,
    "üheksa": 9,
    "kümme": 10, "kümne": 10,
}

# Regular expressions and keywords identifying pure code synthesis requests (no VM execution needed)
SYNTHESIS_PATTERNS: List[str] = [
    r"\bloo\b.*\bskript\b",
    r"\btee\b.*\bskript\b",
    r"\bkirjuta\b.*\bkood\b",
    r"\bkirjuta\b.*\bskript\b",
    r"\bnäita\b.*\bkoodi?\b",
    r"\bloo\b.*\bfunktsioon\b",
    r"\btee\b.*\bfunktsioon\b",
    r"\bkirjuta\b.*\bfunktsioon\b",
    r"\bkuva\b.*\bkoodi?\b",
    r"\bgenerate\b.*\bcode\b",
    r"\bwrite\b.*\bscript\b",
    r"\bshow\b.*\bcode\b",
    r"\bcreate\b.*\bfunction\b",
    r"\bpythoni\s+skript\b",
    r"\bpythoni\s+kood\b",
    r"\bkoodi\s+süntees\b",
    r"\bsünteesi\b",
]

_WORDS_REGEX = "|".join(re.escape(w) for w in sorted(ESTONIAN_WORDS_TO_INT.keys(), key=len, reverse=True))
_NUMBER_PATTERN = re.compile(rf"(?:0x[0-9a-fA-F]+|-?\d+|\b(?:{_WORDS_REGEX})\b)", re.IGNORECASE)
_QUANTIFIER_PATTERN = re.compile(
    rf"\b(?:{_WORDS_REGEX}|\d+)\s+(?:arvu|arvud|arvude|arvudel|numbrit|numbrid|numbrite|numbritel|tükki|väärtust)\b",
    re.IGNORECASE,
)


def is_synthesis_query(text: str) -> bool:
    """Determines whether a query requests pure code/script synthesis without immediate execution."""
    clean = text.lower().strip()
    return any(re.search(pat, clean) for pat in SYNTHESIS_PATTERNS)


def extract_numbers(text: str) -> List[int]:
    """
    Extracts integer values (including negative numbers, hex format, and
    Estonian number words) from a raw query string in strict left-to-right order.
    Filters out numeral quantifiers modifying count nouns (e.g. 'kolm arvu', 'kaks numbrit').
    """
    cleaned_text = _QUANTIFIER_PATTERN.sub(" ", text)
    matches = _NUMBER_PATTERN.findall(cleaned_text)
    numbers: List[int] = []
    for m in matches:
        m_lower = m.lower()
        if m_lower in ESTONIAN_WORDS_TO_INT:
            numbers.append(ESTONIAN_WORDS_TO_INT[m_lower])
        elif m.startswith(("0x", "0X")):
            numbers.append(int(m, 16))
        else:
            numbers.append(int(m, 10))
    return numbers
