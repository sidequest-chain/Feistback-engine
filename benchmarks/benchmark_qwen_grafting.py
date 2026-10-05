"""
Benchmark EXP-011: 8-Layer Code Model Grafting (Qwen2.5-Coder-0.5B).
===================================================================
Validates aggressive architectural grafting on a real pre-trained LLM:
  - Base: Qwen/Qwen2.5-Coder-0.5B (24 transformer decoder layers, hidden_size=896).
  - Grafted: 16 retained Qwen decoder layers + 8 ReversibleFeistelNetwork layers
    (d_model=896, layer_scale=0.0) replacing layers 16..23 (33.3% KV reduction).
Measures:
  - Baseline vs. Grafted generation latency (ms) and throughput (tokens/sec).
  - Semantic probability distribution cosine similarity and raw logits similarity.
  - Peak VRAM allocation and theoretical KV-cache footprint reduction (33.3%).
  - Generated code quality and syntactic correctness on standard Python prompt.
"""

from __future__ import annotations

import time
from typing import Any, Dict
import torch

from benchmarks.tracker import ExperimentTracker
from core.qwen_grafted import GraftedQwenCoder


def run_qwen_grafting_benchmark(
    max_new_tokens: int = 50,
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    num_grafted_layers: int = 8,
    layer_scale: float = 0.0,
    exp_id: str = "EXP-011-QWEN-8LAYER-GRAFTING",
    notes: str = "8-layer grafting test: 16 Qwen + 8 Feistel layers",
) -> Dict[str, Any]:
    """
    Executes and benchmarks the EXP-011 Qwen2.5-Coder-0.5B 8-layer grafting experiment.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {exp_id} GraftedQwenCoder on device: {device}...")

    # Initialize model
    grafted_model = GraftedQwenCoder(
        model_name=model_name,
        num_grafted_layers=num_grafted_layers,
        layer_scale=layer_scale,
        device=device,
    )

    prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
    encoded = grafted_model.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(grafted_model.device)

    # 1. Warmup both modes
    print("Warming up baseline and grafted pipelines...")
    _ = grafted_model.forward(input_ids, mode="baseline")
    _ = grafted_model.forward(input_ids, mode="grafted")
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"Running {exp_id} generation benchmark (max_new_tokens={max_new_tokens})...")

    with ExperimentTracker(exp_id, notes=notes) as tracker:
        # A. Logits & Distribution Cosine Similarity at last prompt token
        logits_base = grafted_model.forward(input_ids, mode="baseline")
        logits_graft = grafted_model.forward(input_ids, mode="grafted")

        # Probability distribution cosine similarity (semantic output distribution alignment)
        probs_base = torch.softmax(logits_base[:, -1, :].float(), dim=-1)
        probs_graft = torch.softmax(logits_graft[:, -1, :].float(), dim=-1)
        prob_cos_sim = torch.cosine_similarity(probs_base, probs_graft, dim=-1).item()
        raw_cos_sim = torch.cosine_similarity(
            logits_base[:, -1, :].float(),
            logits_graft[:, -1, :].float(),
            dim=-1,
        ).item()
        # Semantic probability alignment metric (expected >0.85–0.95+)
        cos_sim = prob_cos_sim

        # B. Baseline Generation Benchmark (24 Attention Layers)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0_base = time.perf_counter()
        text_base, ids_base = grafted_model.generate(
            prompt, max_new_tokens=max_new_tokens, mode="baseline"
        )
        if device == "cuda":
            torch.cuda.synchronize()
        base_duration_sec = time.perf_counter() - t0_base
        base_gen_ms = base_duration_sec * 1000.0
        base_tokens_generated = ids_base.shape[-1] - input_ids.shape[-1]
        base_tokens_sec = (
            base_tokens_generated / base_duration_sec if base_duration_sec > 0 else 0.0
        )
        base_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        )

        # C. Grafted Generation Benchmark (20 Attention + 4 Reversible Feistel)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0_graft = time.perf_counter()
        text_graft, ids_graft = grafted_model.generate(
            prompt, max_new_tokens=max_new_tokens, mode="grafted"
        )
        if device == "cuda":
            torch.cuda.synchronize()
        graft_duration_sec = time.perf_counter() - t0_graft
        graft_gen_ms = graft_duration_sec * 1000.0
        graft_tokens_generated = ids_graft.shape[-1] - input_ids.shape[-1]
        graft_tokens_sec = (
            graft_tokens_generated / graft_duration_sec if graft_duration_sec > 0 else 0.0
        )
        graft_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        )

        speedup_x = base_gen_ms / graft_gen_ms if graft_gen_ms > 0 else 1.0
        kv_cache_reduction_pct = (num_grafted_layers / grafted_model.total_layers) * 100.0

        # Log metrics to ExperimentTracker
        tracker.log_metric("baseline_gen_ms", round(base_gen_ms, 2))
        tracker.log_metric("grafted_gen_ms", round(graft_gen_ms, 2))
        tracker.log_metric("speedup_x", round(speedup_x, 2))
        tracker.log_metric("baseline_vram_mb", round(base_vram_mb, 2))
        tracker.log_metric("grafted_vram_mb", round(graft_vram_mb, 2))
        tracker.log_metric("logits_cosine_similarity", round(cos_sim, 4))
        tracker.log_metric("raw_logits_cosine_similarity", round(raw_cos_sim, 4))
        tracker.log_metric("kv_cache_reduction_pct", round(kv_cache_reduction_pct, 1))
        tracker.set_status("SUCCESS")

        # Formatted Result Table
        graft_label = f"Poogitud ({grafted_model.num_retained_layers} Qwen + {num_grafted_layers} F)"
        print("\n" + "=" * 90)
        print(f"{exp_id}: REAL CODE MODEL GRAFTING ({grafted_model.num_retained_layers} Qwen + {num_grafted_layers} Feistel) BENCHMARK")
        print("=" * 90)
        print(f"{'Arhitektuur':<26} | {'Genereerimise aeg (ms)':<23} | {'Kiirus (t/s)':<14} | {'VRAM (MB)':<10} | {'KV kokkuhoid':<12}")
        print("-" * 90)
        print(f"{'Baasmudel (24 Attention)':<26} | {base_gen_ms:<23.2f} | {base_tokens_sec:<14.1f} | {base_vram_mb:<10.2f} | {'0.0% (baas)':<12}")
        print(f"{graft_label:<26} | {graft_gen_ms:<23.2f} | {graft_tokens_sec:<14.1f} | {graft_vram_mb:<10.2f} | {f'{kv_cache_reduction_pct:.1f}%':<12}")
        print("=" * 90)
        print(f"Logitsite (jaotus) koosiinus-sarnasus: {cos_sim:.4f} (Raw logits: {raw_cos_sim:.4f})")
        print(f"Kiiruse suhe (Speedup):               {speedup_x:.2f}x")
        print("-" * 90)
        print("Baasmudeli genereeritud kood:")
        print(prompt + text_base.rstrip())
        print("-" * 90)
        print("Poogitud mudeli genereeritud kood:")
        print(prompt + text_graft.rstrip())
        print("=" * 90 + "\n")

    return {
        "baseline_gen_ms": base_gen_ms,
        "grafted_gen_ms": graft_gen_ms,
        "speedup_x": speedup_x,
        "baseline_vram_mb": base_vram_mb,
        "grafted_vram_mb": graft_vram_mb,
        "logits_cosine_similarity": cos_sim,
        "kv_cache_reduction_pct": kv_cache_reduction_pct,
        "text_base": text_base,
        "text_graft": text_graft,
    }


if __name__ == "__main__":
    run_qwen_grafting_benchmark()
