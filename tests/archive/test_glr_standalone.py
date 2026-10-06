"""
Test script for verifying PureFeistelLM with GatedLinearRecurrence
"""
import torch
from core.pure_feistel_lm import PureFeistelLM, GatedLinearRecurrence, SpectralMixer, FeistelBlock

def test_glr():
    glr = GatedLinearRecurrence(dim=448, state_dim=128)
    params = sum(p.numel() for p in glr.parameters())
    print(f"GLR params per layer: {params}")
    assert params == 57472, f"Expected 57472, got {params}"

if __name__ == "__main__":
    test_glr()
