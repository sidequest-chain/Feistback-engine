"""
Runner Package
==============
Orchestration and intent dispatching layer linking PureFeistelLM and MiniStackVM.
"""

from runner.engine import FeistbackEngine, EngineResult, TestOutcome
from runner.dispatcher import BaseDispatcher, ConfigDispatcher

# Optional backward compatibility with archived legacy components
try:
    from archive.legacy_runner.tokenizer import HybridTokenizer
    from archive.legacy_runner.args_parser import extract_numeric_args
    from archive.legacy_runner.pipeline import HybridPipeline
    from archive.legacy_runner.adaptive_loop import AdaptiveFeistelRunner
    from archive.legacy_runner.async_pipeline import AsyncHybridPipeline
    from archive.legacy_runner.agent_loop import AutonomousFeistelAgent
    from archive.legacy_runner.log_task_runner import LogTaskRunner
except ImportError:
    pass

__all__ = [
    "FeistbackEngine",
    "EngineResult",
    "TestOutcome",
    "BaseDispatcher",
    "ConfigDispatcher",
]
