"""
Legacy Dispatcher Module (Deprecated)
=====================================
Maintained for backward compatibility. Please import from dispatchers instead.
"""

from dispatchers.base import (
    BaseDispatcher,
    ESTONIAN_WORDS_TO_INT,
    SYNTHESIS_PATTERNS,
    is_synthesis_query,
    extract_numbers,
)
from dispatchers.fast_path import ConfigDispatcher
from dispatchers.council import CouncilDispatcher

__all__ = [
    "BaseDispatcher",
    "ConfigDispatcher",
    "CouncilDispatcher",
    "ESTONIAN_WORDS_TO_INT",
    "SYNTHESIS_PATTERNS",
    "is_synthesis_query",
    "extract_numbers",
]
