"""
Estonian Natural Language & Number Extraction Primitives
========================================================
Provides Estonian number word mapping, quantifier filtration, synthesis query
classification, and numerical operand extraction for Feistback dispatchers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Union


# Mapping of Estonian number words and common inflected forms to integers (0-10)
ESTONIAN_WORDS_TO_INT: Dict[str, int] = {
    "null": 0, "nulliga": 0,
    "üks": 1, "ühe": 1, "ühega": 1, "ühele": 1, "ühest": 1,
    "kaks": 2, "kahe": 2, "kahega": 2, "kahele": 2, "kahest": 2,
    "kolm": 3, "kolme": 3, "kolmega": 3, "kolmele": 3, "kolmest": 3,
    "neli": 4, "nelja": 4, "neljaga": 4, "neljale": 4, "neljast": 4,
    "viis": 5, "viie": 5, "viiega": 5, "viiele": 5, "viiest": 5,
    "kuus": 6, "kuue": 6, "kuuega": 6, "kuuele": 6, "kuuest": 6,
    "seitse": 7, "seitsme": 7, "seitsmega": 7, "seitsmele": 7, "seitsmest": 7,
    "kaheksa": 8, "kaheksaga": 8, "kaheksale": 8, "kaheksast": 8,
    "üheksa": 9, "üheksaga": 9, "üheksale": 9, "üheksast": 9,
    "kümme": 10, "kümne": 10, "kümnega": 10, "kümnele": 10, "kümnest": 10,
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

_QUANTIFIER_PATTERN = re.compile(
    rf"\b(?:{_WORDS_REGEX}|\d+)\s+(?:arvu|arvud|arvude|arvudel|numbrit|numbrid|numbrite|numbritel|väärtust)\b",
    re.IGNORECASE,
)

# Verbal float expressions (e.g. "kolm koma viis", "kaks punkt seitse", "miinus 0 koma 5")
_PART_PAT = rf"(?:\d+|(?:{_WORDS_REGEX}))"
_VERBAL_FLOAT = rf"(?:miinus\s+|-)?{_PART_PAT}\s+(?:koma|punkt)\s+{_PART_PAT}"

# Decimal numeric floats with dot or comma without space (e.g. "3.14", "-0.5", "12,5")
_DIGIT_FLOAT = r"-?\d+[.,]\d+"

# Hexadecimal integers (e.g. 0xFF)
_HEX_INT = r"0x[0-9a-fA-F]+"

# Decimal integers (e.g. 42, -7)
_DEC_INT = r"-?\d+"

# Single Estonian number word
_WORD_INT = rf"\b(?:{_WORDS_REGEX})\b"

_FULL_NUMBER_PATTERN = re.compile(
    rf"({_VERBAL_FLOAT})|({_DIGIT_FLOAT})|({_HEX_INT})|({_DEC_INT})|({_WORD_INT})",
    re.IGNORECASE,
)


def _parse_verbal_token_int(tok: str) -> int:
    """Parses a single number word or digit string into an integer."""
    clean = tok.strip().lower()
    if clean in ESTONIAN_WORDS_TO_INT:
        return ESTONIAN_WORDS_TO_INT[clean]
    return int(clean)


def is_synthesis_query(text: str) -> bool:
    """Determines whether a query requests pure code/script synthesis without immediate execution."""
    clean = text.lower().strip()
    return any(re.search(pat, clean) for pat in SYNTHESIS_PATTERNS)


def extract_numbers(text: str) -> List[Union[int, float]]:
    """
    Extracts numerical values (integers and floats, including Estonian verbal floats,
    commas/dots, negative numbers, hex format, and number words) in strict left-to-right order.
    Returns integer values as int and floating point values as float.
    Filters out numeral quantifiers modifying meta-count nouns (e.g. 'kolm arvu', 'kaks numbrit').
    """
    cleaned_text = _QUANTIFIER_PATTERN.sub(" ", text)
    numbers: List[Union[int, float]] = []

    for match in _FULL_NUMBER_PATTERN.finditer(cleaned_text):
        v_float, d_float, h_int, d_int, w_int = match.groups()

        if v_float:
            # Verbal float: e.g. "kolm koma viis", "miinus kaks koma seitse"
            is_neg = False
            s = v_float.lower().strip()
            if s.startswith("miinus"):
                is_neg = True
                s = s[6:].strip()
            elif s.startswith("-"):
                is_neg = True
                s = s[1:].strip()

            parts = re.split(r"\s+(?:koma|punkt)\s+", s)
            int_part = _parse_verbal_token_int(parts[0])
            frac_raw = parts[1].strip()
            if frac_raw in ESTONIAN_WORDS_TO_INT:
                frac_str = str(ESTONIAN_WORDS_TO_INT[frac_raw])
            else:
                frac_str = frac_raw

            val = float(f"{int_part}.{frac_str}")
            numbers.append(-val if is_neg else val)

        elif d_float:
            # Numeric float: e.g. "3.14", "-0.5", "12,5"
            numbers.append(float(d_float.replace(",", ".")))

        elif h_int:
            # Hexadecimal integer: e.g. 0xFF
            numbers.append(int(h_int, 16))

        elif d_int:
            # Decimal integer: e.g. 42, -7
            numbers.append(int(d_int, 10))

        elif w_int:
            # Estonian number word: e.g. "kaks", "kahega"
            numbers.append(ESTONIAN_WORDS_TO_INT[w_int.lower()])

    return numbers
