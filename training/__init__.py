"""
Training package for FeistBack Engine distillation and fine-tuning.
"""

from __future__ import annotations

from training.distill import FeistelDistillationTrainer
from training.dataset import CodeDistillationDataset, generate_default_code_dataset

__all__ = [
    "FeistelDistillationTrainer",
    "CodeDistillationDataset",
    "generate_default_code_dataset",
]
