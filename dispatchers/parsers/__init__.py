"""
Natural language and numerical parser modules for Feistback dispatchers.
"""

from dispatchers.parsers.et_numbers import (
    ESTONIAN_WORDS_TO_INT,
    SYNTHESIS_PATTERNS,
    is_synthesis_query,
    extract_numbers,
)

__all__ = [
    "ESTONIAN_WORDS_TO_INT",
    "SYNTHESIS_PATTERNS",
    "is_synthesis_query",
    "extract_numbers",
]
