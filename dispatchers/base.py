"""
Base Dispatcher Contract & Abstract Interface
==============================================
Defines the BaseDispatcher abstract contract for the Feistback dispatching layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Backward compatibility re-export
from dispatchers.parsers.et_numbers import (
    extract_numbers,
    is_synthesis_query,
    ESTONIAN_WORDS_TO_INT,
    SYNTHESIS_PATTERNS,
)


class BaseDispatcher(ABC):
    """
    Abstract Base Class defining the contract for intent dispatchers.
    """

    @abstractmethod
    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Parses a user query, identifies intent and arguments, and executes
        the corresponding neuro-symbolic generation and VM verification.

        Args:
            user_query: Free-form natural language or command string.
            pending_intent: Optional pending intent awaiting missing arguments.
            pending_args: Optional list of previously collected arguments.

        Returns:
            Dict[str, Any]: Execution status, intent metadata, generated code,
                            and deterministic MiniStackVM result.
        """
        pass


__all__ = [
    "BaseDispatcher",
    "extract_numbers",
    "is_synthesis_query",
]
