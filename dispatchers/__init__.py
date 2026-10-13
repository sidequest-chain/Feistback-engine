"""
Intent Dispatchers and Dialogue Routing Layer
=============================================
Modular dispatchers routing natural language to deterministic neuro-symbolic contracts.
"""

from dispatchers.base import (
    BaseDispatcher,
    ESTONIAN_WORDS_TO_INT,
    SYNTHESIS_PATTERNS,
    is_synthesis_query,
    extract_numbers,
)
from dispatchers.cache_manager import LearnedIntentCache
from dispatchers.fast_path import ConfigDispatcher
from dispatchers.council import CouncilDispatcher

__all__ = [
    "BaseDispatcher",
    "LearnedIntentCache",
    "ConfigDispatcher",
    "CouncilDispatcher",
    "ESTONIAN_WORDS_TO_INT",
    "SYNTHESIS_PATTERNS",
    "is_synthesis_query",
    "extract_numbers",
]
