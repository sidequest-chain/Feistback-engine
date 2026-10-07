"""
EXP-002: Poincaré Attractor Stability & Fixed-Point Convergence Benchmark
=========================================================================
Measures the phase-space stability and Lyapunov bounded dynamics of a recursive
reversible Feistel round (z_{t+1} = Block(z_t)) across iterations 1 to 16.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch

from benchmarks.tracker import ExperimentTracker
from core.reversible_feistel import ReversibleFeistelNetwork


def run_attractor_stability_benchmark(
    batch_size: int = 16,
    seq_len: int = 64,
    d_model: int = 128,
    num_steps: int = 16,
    epsilon: float = 0.01,
    layer_scale: float = 0.05,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Executes the EXP-002 Poincaré attractor stability test.

    Iterates hidden state z through the same reversible Feistel block:
        z_{t+1} = Block(z_t)
    Tracking:
        1. Mean Euclidean state delta: delta_t = ||z_{t+1} - z_t||_2
        2. Mean state vector norm: norm_t = ||z_t||_2

    Args:
        batch_size: Number of parallel sequence samples (default: 16).
        seq_len: Sequence length L (default: 64).
        d_model: Hidden state dimension D (default: 128).
        num_steps: Recurrence iterations (default: 16).
        epsilon: Stability / convergence threshold for delta_t (default: 0.01).
        layer_scale: Feistel residual branch scaling factor (default: 0.05).
        seed: Random generator seed for reproducibility.

    Returns:
        List of dictionaries with per-step measurements.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Feistel Reversible Block on {device} (d_model={d_model}, seq_len={seq_len})...")

    # A canonical full Feistel round (2 half-layers so both branches are updated)
    block = ReversibleFeistelNetwork(
        num_layers=2,
        d_model=d_model,
        max_seq_len=seq_len,
        layer_scale=layer_scale,
    ).to(device)
    block.eval()

    # Initial state vector z_0
    z = torch.randn(batch_size, seq_len, d_model, device=device)

    results: List[Dict[str, Any]] = []
    first_converged_step: Optional[int] = None

    with ExperimentTracker(
        "EXP-002-ATTRACTOR-STABILITY", notes="Poincaré fixed-point convergence test"
    ) as tracker:
        print(f"\nRunning {num_steps} recurrent iterations with epsilon={epsilon}...\n")

        with torch.no_grad():
            for t in range(1, num_steps + 1):
                norm_t = torch.norm(z, dim=-1).mean().item()
                z_next = block(z)
                delta_t = torch.norm(z_next - z, dim=-1).mean().item()

                is_converged = delta_t < epsilon
                status = "CONVERGED" if is_converged else "ACTIVE"

                if is_converged and first_converged_step is None:
                    first_converged_step = t

                results.append({
                    "step": t,
                    "delta": delta_t,
                    "norm": norm_t,
                    "status": status,
                })

                z = z_next

        # Compute summary metrics across orbit
        final_delta = results[-1]["delta"]
        max_norm = max(r["norm"] for r in results)
        min_norm = min(r["norm"] for r in results)
        norm_drift = max_norm - min_norm

        tracker.log_metric("delta_1", round(results[0]["delta"], 6))
        tracker.log_metric("delta_16", round(final_delta, 6))
        tracker.log_metric("max_norm", round(max_norm, 4))
        tracker.log_metric("min_norm", round(min_norm, 4))
        tracker.log_metric("norm_drift", round(norm_drift, 6))
        tracker.log_metric("convergence_step", first_converged_step if first_converged_step is not None else -1)
        tracker.log_metric("epsilon", epsilon)

        if first_converged_step is not None:
            tracker.set_status(f"CONVERGED (step {first_converged_step})")
        else:
            tracker.set_status("STABLE")

    # Print ASCII / Markdown table
    print("\n" + "=" * 76)
    print(f"{'Step':<8} | {'Delta (||z_{t+1} - z_t||)':<27} | {'State Norm ||z_t||':<20} | {'Status':<12}")
    print("-" * 76)
    for row in results:
        print(f"{row['step']:<8} | {row['delta']:<27.6f} | {row['norm']:<20.4f} | {row['status']:<12}")
    print("=" * 76 + "\n")

    return results


if __name__ == "__main__":
    run_attractor_stability_benchmark()
