"""
Benchmark EXP-005: Model Brain Grafting (Attention-to-Feistel Grafting).
Compares pure 16-layer Transformer baseline against a hybrid model where
the final 4 layers are replaced by ReversibleFeistelNetwork (12 Transformer + 4 Feistel).
Measures forward-pass latency, peak VRAM, KV-cache savings, and latent cosine similarity.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.tracker import ExperimentTracker
from core.reversible_feistel import ReversibleFeistelNetwork


class BaselineTransformerModel(nn.Module):
    """
    Model A: Pure 16-layer Transformer Encoder.
    16 full multi-head attention layers with standard O(N^2) KV footprint.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 16) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class GraftedTransformerFeistelModel(nn.Module):
    """
    Model B: Hybrid Grafted Model.
    First 12 layers are TransformerEncoder, final 4 layers are ReversibleFeistelNetwork.
    Eliminates KV-cache storage for the final 4 layers (25% theoretical KV reduction).
    """

    def __init__(
        self,
        base_encoder: nn.TransformerEncoder,
        num_transformer_layers: int = 12,
        num_feistel_layers: int = 4,
        d_model: int = 128,
        max_seq_len: int = 16384,
        layer_scale: float = 0.05,
    ) -> None:
        super().__init__()
        # Initialize 12-layer TransformerEncoder with identical configuration
        nhead = base_encoder.layers[0].self_attn.num_heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Copy initialized weights from the first 12 layers of Model A
        for i in range(num_transformer_layers):
            self.transformer.layers[i].load_state_dict(base_encoder.layers[i].state_dict())

        # Graft 4-layer ReversibleFeistelNetwork for the final deliberation depth
        self.feistel = ReversibleFeistelNetwork(
            num_layers=num_feistel_layers,
            d_model=d_model,
            max_seq_len=max_seq_len,
            layer_scale=layer_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transformer(x)
        h = self.feistel(h)
        return h


def benchmark_model_forward(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> Tuple[float, float]:
    """
    Profiles average forward pass execution time (ms) and peak VRAM allocated (MB).
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
        elapsed_sec = time.perf_counter() - start_time

    avg_time_ms = (elapsed_sec / num_repeats) * 1000.0

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_vram_mb = 0.0

    return avg_time_ms, peak_vram_mb


def run_grafting_benchmark(
    d_model: int = 128,
    nhead: int = 4,
    total_layers: int = 16,
    grafted_feistel_layers: int = 4,
    seq_lengths: List[int] | None = None,
    num_warmup: int = 3,
    num_repeats: int = 10,
) -> Dict[str, Any]:
    """
    Executes the EXP-005 Attention-to-Feistel Grafting benchmark across sequence lengths.
    """
    if seq_lengths is None:
        seq_lengths = [1024, 4096, 16384]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXP-005 Grafting Benchmark on device: {device}")

    torch.manual_seed(42)
    max_len = max(seq_lengths)
    num_transformer_layers = total_layers - grafted_feistel_layers

    # 1. Instantiate Model A (Pure Baseline 16 layers)
    model_a = BaselineTransformerModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=total_layers,
    ).to(device)
    model_a.eval()

    # 2. Instantiate Model B (Grafted 12 Transformer + 4 Feistel)
    model_b = GraftedTransformerFeistelModel(
        base_encoder=model_a.transformer,
        num_transformer_layers=num_transformer_layers,
        num_feistel_layers=grafted_feistel_layers,
        d_model=d_model,
        max_seq_len=max_len,
        layer_scale=0.05,
    ).to(device)
    model_b.eval()

    results: List[Dict[str, Any]] = []
    cos_similarities: List[float] = []

    with ExperimentTracker(
        "EXP-005-GRAFTING",
        notes="Attention-to-Feistel 4-layer grafting KV/latency test",
    ) as tracker:
        for seq_len in seq_lengths:
            print(f"Evaluating sequence length {seq_len}...")
            x = torch.randn(1, seq_len, d_model, device=device)

            # Profile Model A
            base_time_ms, base_vram_mb = benchmark_model_forward(
                model_a, x, device, num_warmup=num_warmup, num_repeats=num_repeats
            )

            # Profile Model B
            graft_time_ms, graft_vram_mb = benchmark_model_forward(
                model_b, x, device, num_warmup=num_warmup, num_repeats=num_repeats
            )

            # Latent Representation Cosine Similarity
            with torch.no_grad():
                out_a = model_a(x)
                out_b = model_b(x)
                cos_sim = F.cosine_similarity(out_a, out_b, dim=-1).mean().item()
            cos_similarities.append(cos_sim)

            speedup = base_time_ms / graft_time_ms if graft_time_ms > 0 else 1.0

            results.append({
                "seq_len": seq_len,
                "base_time_ms": base_time_ms,
                "graft_time_ms": graft_time_ms,
                "speedup": speedup,
                "base_vram_mb": base_vram_mb,
                "graft_vram_mb": graft_vram_mb,
                "cos_sim": cos_sim,
            })

        # Key metrics for 16k sequence length
        res_16k = next(r for r in results if r["seq_len"] == 16384)
        mean_cos_sim = sum(cos_similarities) / len(cos_similarities)
        kv_cache_reduction_pct = (grafted_feistel_layers / total_layers) * 100.0
        vram_saving_16k_mb = max(0.0, res_16k["base_vram_mb"] - res_16k["graft_vram_mb"])

        tracker.log_metric("baseline_time_16k_ms", round(res_16k["base_time_ms"], 3))
        tracker.log_metric("grafted_time_16k_ms", round(res_16k["graft_time_ms"], 3))
        tracker.log_metric("speedup_16k", round(res_16k["speedup"], 2))
        tracker.log_metric("baseline_vram_16k_mb", round(res_16k["base_vram_mb"], 2))
        tracker.log_metric("grafted_vram_16k_mb", round(res_16k["graft_vram_mb"], 2))
        tracker.log_metric("vram_saving_mb", round(vram_saving_16k_mb, 2))
        tracker.log_metric("kv_cache_reduction_pct", kv_cache_reduction_pct)
        tracker.log_metric("mean_cosine_similarity", round(mean_cos_sim, 4))
        tracker.set_status("SUCCESS")

        # Print Comparison Table
        print("\n" + "=" * 96)
        print("EXP-005: ATTENTION-TO-FEISTEL GRAFTING BENCHMARK RESULTS")
        print("=" * 96)
        print(f"{'Seq Len':<10} | {'Baseline Time (ms)':<20} | {'Grafted Time (ms)':<20} | {'Speedup':<10} | {'Baseline VRAM':<15} | {'Grafted VRAM':<15} | {'Cos Sim':<10}")
        print("-" * 96)
        for r in results:
            speedup_str = f"{r['speedup']:.2f}x"
            base_vram_str = f"{r['base_vram_mb']:.2f} MB"
            graft_vram_str = f"{r['graft_vram_mb']:.2f} MB"
            print(
                f"{r['seq_len']:<10} | "
                f"{r['base_time_ms']:<20.3f} | "
                f"{r['graft_time_ms']:<20.3f} | "
                f"{speedup_str:<10} | "
                f"{base_vram_str:<15} | "
                f"{graft_vram_str:<15} | "
                f"{r['cos_sim']:<10.4f}"
            )
        print("=" * 96)
        print(f"KV-Cache Theoretical Reduction: {kv_cache_reduction_pct:.1f}% (4 layers without KV memory footprint)")
        print(f"Mean Latent Cosine Similarity:  {mean_cos_sim:.4f}")
        print(f"16k Tokens VRAM Saving:        {vram_saving_16k_mb:.2f} MB\n")

        return {
            "results": results,
            "mean_cos_sim": mean_cos_sim,
            "kv_cache_reduction_pct": kv_cache_reduction_pct,
        }


if __name__ == "__main__":
    run_grafting_benchmark()
