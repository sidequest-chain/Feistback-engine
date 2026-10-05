"""
Runner Package
==============
Orchestration layer linking Feistel neural reasoner and MiniStackVM.
"""

from runner.tokenizer import HybridTokenizer
from runner.args_parser import extract_numeric_args
from runner.pipeline import HybridPipeline
from .adaptive_loop import AdaptiveFeistelRunner
from .async_pipeline import AsyncHybridPipeline
from .agent_loop import AutonomousFeistelAgent
from .log_task_runner import LogTaskRunner

__all__ = [
    "HybridTokenizer",
    "extract_numeric_args",
    "HybridPipeline",
    "AdaptiveFeistelRunner",
    "AsyncHybridPipeline",
    "AutonomousFeistelAgent",
    "LogTaskRunner",
]


