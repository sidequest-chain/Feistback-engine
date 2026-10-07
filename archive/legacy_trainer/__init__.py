"""
Trainer Package
===============
Training scripts and dynamic dataset synthesis.
"""

from trainer.dynamic_dataset import generate_dynamic_dataset, generate_dynamic_sample
from trainer.train_hybrid import train_dynamic_hybrid

__all__ = [
    "generate_dynamic_dataset",
    "generate_dynamic_sample",
    "train_dynamic_hybrid",
]
