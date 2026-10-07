"""
Core Package
============
Pure neural architecture: PureFeistelLM with Reversible Associative Feistel (RAF).
Includes O(1) reversible Feistel layers, associative matrix memory, and holographic FFT mixer.
"""

from core.pure_feistel_lm import PureFeistelLM
from core.associative_memory import AssociativeMemoryBlock
from core.reversible_feistel import ReversibleFeistelNetwork, ReversibleFeistelBlock
from core.holographic_mixer import HolographicMixer, functional_holographic_mixer
from core.qwen_grafted import GraftedQwenCoder

# Optional backward compatibility with archived legacy model
try:
    from archive.legacy_core.model import FeistelHolographicLM, ReversibleHolographicTransformer
except ImportError:
    pass

__all__ = [
    "PureFeistelLM",
    "AssociativeMemoryBlock",
    "ReversibleFeistelNetwork",
    "ReversibleFeistelBlock",
    "HolographicMixer",
    "functional_holographic_mixer",
    "GraftedQwenCoder",
]

