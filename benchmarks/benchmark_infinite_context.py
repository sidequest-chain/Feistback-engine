"""
EXP-001: Infinite Context VRAM Profiling Benchmark
==================================================
Measures FeistelHolographicLM peak VRAM allocations and forward pass execution
timing across sequence lengths scaling exponentially from 1,024 to 65,536 tokens.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import torch

from benchmarks.tracker import ExperimentTracker
from core.model import FeistelHolographicLM


def benchmark_single_length(
    model: FeistelHolographicLM,
    device: torch.device,
    seq_len: int,
    vocab_size: int = 124,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> Tuple[float, float, str]:
    """
    Executes warmup and timing passes for a given sequence length.

    Args:
        model: FeistelHolographicLM instance.
        device: PyTorch device ('cuda' or 'cpu').
        seq_len: Target sequence length (L).
        vocab_size: Vocabulary size for synthetic token sampling.
        num_warmup: Warmup iterations before profiling.
        num_repeats: Iterations for calculating mean step duration.

    Returns:
        Tuple of (peak_vram_mb, avg_time_ms, status_str).
    """
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Generate synthetic input sequence
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)

        # Warmup passes
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Reset peak memory stats before measurement phase
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            # Profile forward pass timing
            start_time = time.perf_counter()
            for _ in range(num_repeats):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_total = time.perf_counter() - start_time

        avg_time_ms = (elapsed_total / num_repeats) * 1000.0

        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_vram_mb = 0.0

        return peak_vram_mb, avg_time_ms, "SUCCESS"

    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
        if "out of memory" in str(err).lower() or isinstance(err, torch.cuda.OutOfMemoryError):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return 0.0, 0.0, "OOM"
        raise err


def run_infinite_context_benchmark(
    vocab_size: int = 124,
    d_model: int = 128,
    num_layers: int = 16,
    seq_lengths: List[int] | None = None,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> List[Dict[str, Any]]:
    """
    Runs the complete EXP-001 sequence length sweep and logs telemetry.

    Args:
        vocab_size: Vocabulary size (default: 124).
        d_model: Hidden dimension (default: 128).
        num_layers: Number of reversible Feistel layers (default: 16).
        seq_lengths: Sequence lengths to evaluate (default: 1k to 65k).
        num_warmup: Warmup iterations per length.
        num_repeats: Measurement iterations per length.

    Returns:
        List of result dictionaries.
    """
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    max_len = max(seq_lengths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing FeistelHolographicLM on {device} (max_seq_len={max_len}, d_model={d_model}, layers={num_layers})...")
    model = FeistelHolographicLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max_len,
    ).to(device)
    model.eval()

    results: List[Dict[str, Any]] = []

    with ExperimentTracker(
        "EXP-001-VRAM-SWEEP", notes="Context sweep 1k-65k tokens"
    ) as tracker:
        print(f"\nStarting sweep across sequence lengths: {seq_lengths}\n")

        for seq_len in seq_lengths:
            print(f"Profiling seq_len={seq_len}...", end=" ", flush=True)
            vram_mb, time_ms, status = benchmark_single_length(
                model=model,
                device=device,
                seq_len=seq_len,
                vocab_size=vocab_size,
                num_warmup=num_warmup,
                num_repeats=num_repeats,
            )
            print(f"VRAM: {vram_mb:.2f} MB | Time: {time_ms:.2f} ms | Status: {status}")

            results.append({
                "seq_len": seq_len,
                "vram_mb": vram_mb,
                "time_ms": time_ms,
                "status": status,
            })

            # Record metrics per sequence length
            tracker.log_metric(f"vram_mb_{seq_len}", round(vram_mb, 2))
            tracker.log_metric(f"time_ms_{seq_len}", round(time_ms, 2))
            tracker.log_metric(f"status_{seq_len}", status)

        # Log aggregated summary metrics
        successful = [r for r in results if r["status"] == "SUCCESS"]
        if successful:
            tracker.log_metric("vram_1k_mb", round(successful[0]["vram_mb"], 2))
            tracker.log_metric("vram_65k_mb", round(successful[-1]["vram_mb"], 2))
            tracker.log_metric("max_vram_mb", round(max(r["vram_mb"] for r in successful), 2))
            tracker.log_metric("time_65k_ms", round(successful[-1]["time_ms"], 2))

        # Check overall sweep status
        oom_count = sum(1 for r in results if r["status"] == "OOM")
        if oom_count > 0:
            tracker.set_status(f"PARTIAL_OOM ({oom_count}/{len(results)} OOM)")

    # Print formatted ASCII / Markdown comparison table
    print("\n" + "=" * 70)
    print(f"{'Seq Len':<10} | {'FeistBack VRAM (MB)':<22} | {'FeistBack Time (ms)':<22} | {'Status':<10}")
    print("-" * 70)
    for row in results:
        vram_str = f"{row['vram_mb']:.2f}" if row["status"] != "OOM" else "OOM"
        time_str = f"{row['time_ms']:.2f}" if row["status"] != "OOM" else "N/A"
        print(f"{row['seq_len']:<10} | {vram_str:<22} | {time_str:<22} | {row['status']:<10}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_infinite_context_benchmark()
