"""
Benchmarks Package
==================
Evaluation and verification pipelines.
"""

from .tracker import ExperimentTracker
from .eval_suite import ModelEvaluationSuite

# Optional backward compatibility with archived legacy benchmarks
try:
    from archive.legacy_benchmarks.benchmark_infinite_context import run_infinite_context_benchmark
    from archive.legacy_benchmarks.benchmark_standard_transformer import run_standard_transformer_benchmark
    from archive.legacy_benchmarks.benchmark_attractor_stability import run_attractor_stability_benchmark
    from archive.legacy_benchmarks.benchmark_adaptive_loop import run_adaptive_loop_benchmark
    from archive.legacy_benchmarks.benchmark_mini_vm_ext import run_mini_vm_ext_benchmark
    from archive.legacy_benchmarks.benchmark_grafting import run_grafting_benchmark
    from archive.legacy_benchmarks.benchmark_stochastic_resonance import run_stochastic_resonance_benchmark
    from archive.legacy_benchmarks.benchmark_async_pipeline import run_async_pipeline_benchmark
    from archive.legacy_benchmarks.benchmark_agent_loop import run_agent_loop_benchmark
    from archive.legacy_benchmarks.benchmark_log_task import run_log_task_benchmark
    from archive.legacy_benchmarks.benchmark_qwen_grafting import run_qwen_grafting_benchmark
    from archive.legacy_benchmarks.verify_generalization import run_verification, load_pipeline
except ImportError:
    pass

__all__ = [
    "ExperimentTracker",
    "ModelEvaluationSuite",
]
