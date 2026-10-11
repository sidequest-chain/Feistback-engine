"""
Neural Model Backbone Implementations
====================================
PureFeistelLM, AssociativeMemoryBlock (RAF), and Holographic Mixers with zero KV-cache.
"""

from runtime.models.pure_feistel_lm import PureFeistelLM, PureFeistelBackbone, FeistelBlock, GatedLinearRecurrence
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
