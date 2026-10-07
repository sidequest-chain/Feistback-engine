"""
EXP-001 Baseline: Standard Transformer OOM Ceiling Benchmark
============================================================
Profiles standard PyTorch nn.TransformerEncoder (16 layers, d_model=128, nhead=4)
across sequence lengths 1k to 65k to determine exact OOM boundaries and VRAM scaling.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from benchmarks.tracker import ExperimentTracker


class BaselineTransformerLM(nn.Module):
    """
    Reference Transformer language model using standard PyTorch nn.TransformerEncoder.
    Constructed with the same depth (16 layers) and hidden width (d_model=128)
    as FeistelHolographicLM for fair baseline comparison.
    """

    def __init__(
        self,
        vocab_size: int = 124,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 16,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.token_embed(input_ids)
        h = self.transformer(h)
        return self.lm_head(h)


def benchmark_single_length(
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    vocab_size: int = 124,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> Tuple[float, float, str]:
    """
    Executes warmup and timing passes for a single sequence length.

    Returns:
        (peak_vram_mb, avg_time_ms, status)
    """
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        x = torch.randint(0, vocab_size, (1, seq_len), device=device)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

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


def run_standard_transformer_benchmark(
    vocab_size: int = 124,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 16,
    seq_lengths: List[int] | None = None,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> List[Dict[str, Any]]:
    """
    Runs the baseline standard Transformer benchmark across sequence lengths.
    Aborts further sweep testing immediately after encountering the first OOM.
    """
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Baseline Standard Transformer on {device} (d_model={d_model}, nhead={nhead}, layers={num_layers})...")

    model = BaselineTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    first_oom_len: int | None = None

    with ExperimentTracker(
        "EXP-001-BASELINE-TRANSFORMER", notes="Standard Transformer OOM ceiling test"
    ) as tracker:
        print(f"\nStarting baseline sweep across sequence lengths: {seq_lengths}\n")

        for seq_len in seq_lengths:
            print(f"Profiling Standard Transformer seq_len={seq_len}...", end=" ", flush=True)
            vram_mb, time_ms, status = benchmark_single_length(
                model=model,
                device=device,
                seq_len=seq_len,
                vocab_size=vocab_size,
                num_warmup=num_warmup,
                num_repeats=num_repeats,
            )

            if status == "OOM":
                print(f"STATUS: OOM (Out Of Memory encountered!)")
                results.append({
                    "seq_len": seq_len,
                    "vram_mb": 0.0,
                    "time_ms": 0.0,
                    "status": "OOM",
                })
                tracker.log_metric(f"status_{seq_len}", "OOM")
                first_oom_len = seq_len
                print(f"--> Aborting further sweep at seq_len={seq_len} to avoid excessive driver stress.")
                break

            print(f"VRAM: {vram_mb:.2f} MB | Time: {time_ms:.2f} ms | Status: {status}")
            results.append({
                "seq_len": seq_len,
                "vram_mb": vram_mb,
                "time_ms": time_ms,
                "status": status,
            })
            tracker.log_metric(f"vram_mb_{seq_len}", round(vram_mb, 2))
            tracker.log_metric(f"time_ms_{seq_len}", round(time_ms, 2))
            tracker.log_metric(f"status_{seq_len}", status)

        # Log aggregated baseline metrics
        successful = [r for r in results if r["status"] == "SUCCESS"]
        if successful:
            tracker.log_metric("baseline_max_success_len", successful[-1]["seq_len"])
            tracker.log_metric("baseline_peak_vram_mb", round(max(r["vram_mb"] for r in successful), 2))
            tracker.log_metric("baseline_last_time_ms", round(successful[-1]["time_ms"], 2))

        if first_oom_len is not None:
            tracker.log_metric("oom_ceiling_len", first_oom_len)
            tracker.set_status(f"OOM_AT_{first_oom_len}")
        else:
            tracker.set_status("SUCCESS_ALL")

    # Print ASCII / Markdown table
    print("\n" + "=" * 72)
    print(f"{'Seq Len':<10} | {'Transformer VRAM (MB)':<24} | {'Transformer Time (ms)':<22} | {'Status':<10}")
    print("-" * 72)
    for row in results:
        vram_str = f"{row['vram_mb']:.2f}" if row["status"] != "OOM" else "OOM"
        time_str = f"{row['time_ms']:.2f}" if row["status"] != "OOM" else "N/A"
        print(f"{row['seq_len']:<10} | {vram_str:<24} | {time_str:<22} | {row['status']:<10}")
    print("=" * 72 + "\n")

    return results


if __name__ == "__main__":
    run_standard_transformer_benchmark()
