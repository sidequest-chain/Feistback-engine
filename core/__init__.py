"""
Core Package
============
Pure neural architecture: Feistel Holographic Language Model.
Includes O(1) reversible Feistel layers and holographic FFT mixer.
"""

from core.model import FeistelHolographicLM, ReversibleHolographicTransformer
from core.reversible_feistel import ReversibleFeistelNetwork, ReversibleFeistelBlock
from core.holographic_mixer import HolographicMixer, functional_holographic_mixer

__all__ = [
    "FeistelHolographicLM",
    "ReversibleHolographicTransformer",
    "ReversibleFeistelNetwork",
    "ReversibleFeistelBlock",
    "HolographicMixer",
    "functional_holographic_mixer",
]
