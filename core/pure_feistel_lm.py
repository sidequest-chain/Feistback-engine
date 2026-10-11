"""
Legacy Pure Feistel LM Module (Deprecated)
==========================================
Maintained for backward compatibility. Please import from runtime.models.pure_feistel_lm instead.
"""

from runtime.models.pure_feistel_lm import (
    PureFeistelLM,
    PureFeistelBackbone,
    FeistelBlock,
    GatedLinearRecurrence,
)

__all__ = [
    "PureFeistelLM",
    "PureFeistelBackbone",
    "FeistelBlock",
    "GatedLinearRecurrence",
]
