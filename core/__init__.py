"""
Legacy Core Package (Deprecated)
================================
Maintained for backward compatibility. Please import from runtime.models instead.
"""

import warnings
warnings.warn(
    "Importing from 'core' is deprecated and will be removed in a future release. "
    "Please update imports to 'runtime.models'.",
    DeprecationWarning,
    stacklevel=2,
)

from runtime.models.pure_feistel_lm import (
    PureFeistelLM,
    PureFeistelBackbone,
    FeistelBlock,
    GatedLinearRecurrence,
)
from runtime.models.associative_memory import AssociativeMemoryBlock
from runtime.models.holographic_mixer import functional_holographic_mixer, compute_param_len

__all__ = [
    "PureFeistelLM",
    "PureFeistelBackbone",
    "FeistelBlock",
    "GatedLinearRecurrence",
    "AssociativeMemoryBlock",
    "functional_holographic_mixer",
    "compute_param_len",
]
