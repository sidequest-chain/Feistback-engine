"""
Benchmark EXP-007: Heterogeneous Asynchronous CPU-VM and GPU Overlapping Pipeline.
Measures execution speedup and GPU idle reduction when overlapping
GPU Feistel neural deliberation with CPU MiniStackVM symbolic execution.
"""

from __future__ import annotations

import time
from typing import Any, Dict
import torch

from benchmarks.tracker import ExperimentTracker
from mini_vm.ops import OpCode
from runner.async_pipeline import AsyncHybridPipeline


def run_async_pipeline_benchmark(
    cycles: int = 100,
    d_model: int = 128,
    seq_len: int = 512,
    batch_size: int = 16,
    deliberation_steps: int = 6,
    warmup_cycles: int = 5,
) -> Dict[str, Any]:
    """
    Simulates 100 multi-step agent reasoning/action cycles comparing
    sequential execution against async overlapped CPU-GPU pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running EXP-007 Async Pipeline Benchmark on device: {device} ({cycles} cycles)...")

    torch.manual_seed(42)
    z = torch.randn(batch_size, seq_len, d_model, device=device)

    # Prepare representative agent MiniStackVM program:
    # JSON extraction + arithmetic branch loop (combining EXP-004 opcodes)
    log_json = '{"status": 200, "threshold": 25, "active": 1}'
    bytecode = [
        OpCode.PUSH_CONST, log_json,
        OpCode.PUSH_CONST, "threshold",
        OpCode.LOAD_JSON_KEY,
        # Loop 25 times decrementing
        OpCode.PUSH_CONST, 1,
        OpCode.SUB,
        OpCode.DUP,
        OpCode.PUSH_CONST, 0,
        OpCode.CMP_GT,
        OpCode.BRANCH_IF, 5,
        OpCode.RET,
    ]

    pipeline = AsyncHybridPipeline(
        runner=None,
        device=device,
    )
    # Re-initialize runner with proper max_seq_len
    from runner.adaptive_loop import AdaptiveFeistelRunner
    pipeline.runner = AdaptiveFeistelRunner(d_model=d_model, max_seq_len=seq_len).to(pipeline.device)
    pipeline.runner.eval()

    # Representative agent CPU task: running batch of VM evaluations (60 VM program executions)
    def agent_cpu_task() -> int:
        accum = 0
        for _ in range(60):
            accum += pipeline.vm.execute(bytecode)
        return accum

    # Warmup all execution paths
    print("Warming up pipeline...")
    for _ in range(warmup_cycles):
        _ = pipeline.runner.forward_static(z, steps=deliberation_steps)
        _ = agent_cpu_task()
        _ = pipeline.run_sequential_cycle(z, agent_cpu_task, steps=deliberation_steps)
        _ = pipeline.run_async_cycle(z, agent_cpu_task, steps=deliberation_steps)
    if device == "cuda":
        torch.cuda.synchronize()

    with ExperimentTracker(
        "EXP-007-ASYNC-PIPELINE",
        notes="Heterogeneous CPU MiniStackVM and GPU Feistel overlapping pipeline",
    ) as tracker:
        # 1. Profile isolated component times to calculate GPU idle baseline
        if device == "cuda":
            torch.cuda.synchronize()
        gpu_start = time.perf_counter()
        for _ in range(cycles):
            _ = pipeline.runner.forward_static(z, steps=deliberation_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        isolated_gpu_ms = (time.perf_counter() - gpu_start) * 1000.0

        cpu_start = time.perf_counter()
        for _ in range(cycles):
            _ = agent_cpu_task()
        isolated_cpu_ms = (time.perf_counter() - cpu_start) * 1000.0

        # 2. Benchmark Sequential Mode
        if device == "cuda":
            torch.cuda.synchronize()
        seq_start = time.perf_counter()
        for _ in range(cycles):
            _ = pipeline.run_sequential_cycle(z, agent_cpu_task, steps=deliberation_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        sequential_total_ms = (time.perf_counter() - seq_start) * 1000.0

        # 3. Benchmark Async Overlapped Mode
        if device == "cuda":
            torch.cuda.synchronize()
        async_start = time.perf_counter()
        for _ in range(cycles):
            _ = pipeline.run_async_cycle(z, agent_cpu_task, steps=deliberation_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        async_total_ms = (time.perf_counter() - async_start) * 1000.0

        # In sequential execution, the GPU is idle during CPU work and stalls:
        gpu_idle_seq_ms = max(0.0, sequential_total_ms - isolated_gpu_ms)
        # Residual GPU idle time during async overlapping:
        gpu_idle_async_ms = max(0.0, async_total_ms - isolated_gpu_ms)
        gpu_idle_reduction_pct = (
            max(0.0, min(100.0, ((gpu_idle_seq_ms - gpu_idle_async_ms) / gpu_idle_seq_ms) * 100.0))
            if gpu_idle_seq_ms > 0
            else 0.0
        )

        speedup_x = sequential_total_ms / async_total_ms if async_total_ms > 0 else 1.0

        tracker.log_metric("sequential_total_ms", round(sequential_total_ms, 2))
        tracker.log_metric("async_total_ms", round(async_total_ms, 2))
        tracker.log_metric("speedup_x", round(speedup_x, 2))
        tracker.log_metric("gpu_idle_reduction_pct", round(gpu_idle_reduction_pct, 1))
        tracker.set_status("SUCCESS")

        # Formatted Output
        seq_per_cycle_ms = sequential_total_ms / cycles
        async_per_cycle_ms = async_total_ms / cycles

        print("\n" + "=" * 84)
        print("EXP-007: HETEROGENEOUS ASYNC CPU-VM AND GPU PIPELINE BENCHMARK")
        print("=" * 84)
        print(f"{'Režiim':<28} | {'Kokkuaeg (ms)':<15} | {'Tsükli aeg (ms)':<16} | {'Kiiruse suhe':<14}")
        print("-" * 84)
        print(f"{'Sequential (Järjestikune)':<28} | {sequential_total_ms:<15.2f} | {seq_per_cycle_ms:<16.3f} | {'1.00x (baas)':<14}")
        print(f"{'Async Overlapped (Põimitud)':<28} | {async_total_ms:<15.2f} | {async_per_cycle_ms:<16.3f} | {f'{speedup_x:.2f}x kiirem':<14}")
        print("=" * 84)
        print(f"GPU jõudeoleku vähenemine (Idle Reduction): {gpu_idle_reduction_pct:.1f}%")
        print(f"GPU isoleeritud aeg ({cycles} tsüklit):        {isolated_gpu_ms:.2f} ms")
        print(f"CPU VM isoleeritud aeg ({cycles} tsüklit):     {isolated_cpu_ms:.2f} ms\n")

    pipeline.close()

    return {
        "sequential_total_ms": sequential_total_ms,
        "async_total_ms": async_total_ms,
        "speedup_x": speedup_x,
        "gpu_idle_reduction_pct": gpu_idle_reduction_pct,
    }


if __name__ == "__main__":
    run_async_pipeline_benchmark()
