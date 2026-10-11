"""
Legacy Runner Package (Deprecated)
==================================
Maintained for backward compatibility. Please import from runtime or dispatchers instead.
"""

import warnings
warnings.warn(
    "Importing from 'runner' is deprecated and will be removed in a future release. "
    "Please update imports to 'runtime' or 'dispatchers'.",
    DeprecationWarning,
    stacklevel=2,
)

from runtime.engine import FeistbackEngine, EngineResult, TestOutcome
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
    "FeistbackEngine",
    "EngineResult",
    "TestOutcome",
    "BaseDispatcher",
    "ConfigDispatcher",
    "CouncilDispatcher",
    "ESTONIAN_WORDS_TO_INT",
    "SYNTHESIS_PATTERNS",
    "is_synthesis_query",
    "extract_numbers",
]
