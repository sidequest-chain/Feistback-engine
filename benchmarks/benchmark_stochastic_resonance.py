"""
Benchmark EXP-006: Stochastic Resonance in 2-Bit Quantization.
Demonstrates the non-equilibrium thermodynamic principle of stochastic resonance:
Injecting controlled stochastic noise before non-linear 2-bit quantization (ternary/quaternary)
linearizes quantization boundaries, allowing sub-threshold signals to be recovered with
higher fidelity and improved SNR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F

from benchmarks.tracker import ExperimentTracker
from core.reversible_feistel import ReversibleFeistelNetwork


def quantize_2bit(x: torch.Tensor) -> torch.Tensor:
    """
    Uniform symmetric 2-bit (4-level) quantizer.
    Quantizes continuous input into discrete levels {-1.5, -0.5, 0.5, 1.5}
    with decision thresholds at {-1.0, 0.0, 1.0}.
    """
    q = torch.empty_like(x)
    q[x < -1.0] = -1.5
    q[(x >= -1.0) & (x < 0.0)] = -0.5
    q[(x >= 0.0) & (x < 1.0)] = 0.5
    q[x >= 1.0] = 1.5
    return q


def run_stochastic_resonance_benchmark(
    batch_size: int = 16,
    seq_len: int = 128,
    d_model: int = 128,
    num_layers: int = 2,
    sigmas: List[float] | None = None,
    num_samples: int = 8,
) -> Dict[str, Any]:
    """
    Executes the EXP-006 Stochastic Resonance benchmark across noise standard deviations.

    Args:
        batch_size: Batch dimension (default: 16).
        seq_len: Sequence length (default: 128).
        d_model: Model hidden dimension (default: 128).
        num_layers: ReversibleFeistelNetwork depth (default: 2).
        sigmas: Standard deviations of injected noise to sweep.
        num_samples: Number of stochastic noise realizations for ensemble dithering.

    Returns:
        Dictionary containing sweep metrics and optimal resonance parameters.
    """
    if sigmas is None:
        sigmas = [0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EXP-006 Stochastic Resonance Benchmark on device: {device}")

    torch.manual_seed(42)

    # 1. Generate real hidden state distribution z ~ N(0, 1)
    z = torch.randn(batch_size, seq_len, d_model, device=device)

    # 2. Instantiate ReversibleFeistelNetwork and compute ground-truth target signal y_true
    block = ReversibleFeistelNetwork(
        num_layers=num_layers,
        d_model=d_model,
        max_seq_len=seq_len,
        layer_scale=0.05,
    ).to(device)
    block.eval()

    with torch.no_grad():
        y_true = block(z)
        signal_power = torch.mean(y_true ** 2).item()

    sweep_results: List[Dict[str, Any]] = []

    with ExperimentTracker(
        "EXP-006-STOCHASTIC-RESONANCE",
        notes="2-bit quantization recovery via stochastic resonance",
    ) as tracker:
        for sigma in sigmas:
            with torch.no_grad():
                if sigma == 0.0:
                    z_quant = quantize_2bit(z)
                    y_pred = block(z_quant)
                else:
                    # Multi-sample stochastic resonance (dithered ensemble)
                    y_pred = (
                        sum(
                            block(quantize_2bit(z + torch.randn_like(z) * sigma))
                            for _ in range(num_samples)
                        )
                        / num_samples
                    )

                mse = F.mse_loss(y_pred, y_true).item()
                cos_sim = F.cosine_similarity(y_pred, y_true, dim=-1).mean().item()
                snr_db = (
                    (10.0 * torch.log10(torch.tensor(signal_power / mse))).item()
                    if mse > 0
                    else float("inf")
                )

            sweep_results.append({
                "sigma": sigma,
                "mse": mse,
                "cos_sim": cos_sim,
                "snr_db": snr_db,
            })

        # Identify optimal noise level where CosSim reaches maximum (and MSE minimum)
        best_entry = max(sweep_results, key=lambda x: x["cos_sim"])
        baseline_entry = next(r for r in sweep_results if r["sigma"] == 0.0)

        baseline_cossim = baseline_entry["cos_sim"]
        optimal_sigma = best_entry["sigma"]
        optimal_cossim = best_entry["cos_sim"]
        resonance_gain_pct = (
            ((optimal_cossim - baseline_cossim) / baseline_cossim) * 100.0
            if baseline_cossim > 0
            else 0.0
        )

        # Log metrics to tracker
        tracker.log_metric("baseline_cossim_no_noise", round(baseline_cossim, 4))
        tracker.log_metric("optimal_sigma", optimal_sigma)
        tracker.log_metric("optimal_cossim", round(optimal_cossim, 4))
        tracker.log_metric("resonance_gain_pct", round(resonance_gain_pct, 2))
        tracker.set_status("SUCCESS")

        # Determine resonance status labels
        for r in sweep_results:
            sig = r["sigma"]
            if sig == 0.0:
                r["status"] = "Baseline (No Noise)"
            elif sig == optimal_sigma:
                r["status"] = "OPTIMAL RESONANCE (Peak)"
            elif sig < optimal_sigma:
                r["status"] = "Resonance Rising (+)"
            else:
                r["status"] = "Supra-optimal (Noise Dominated)"

        # Print ASCII/Markdown Comparison Table
        print("\n" + "=" * 80)
        print("EXP-006: STOCHASTIC RESONANCE IN 2-BIT QUANTIZATION BENCHMARK")
        print("=" * 80)
        print(f"{'Sigma (Müra)':<14} | {'MSE Viga':<12} | {'Cos Sim':<10} | {'SNR (dB)':<10} | {'Resonantsi olek':<28}")
        print("-" * 80)
        for r in sweep_results:
            print(
                f"{r['sigma']:<14.3f} | "
                f"{r['mse']:<12.5f} | "
                f"{r['cos_sim']:<10.5f} | "
                f"{r['snr_db']:<10.2f} | "
                f"{r['status']:<28}"
            )
        print("=" * 80)
        print(f"Müravaba 2-bit baasjoon:   CosSim = {baseline_cossim:.4f}")
        print(f"Optimaalne müra lävi:      sigma* = {optimal_sigma:.3f}")
        print(f"Optimaalne rekonstruktsioon: CosSim = {optimal_cossim:.4f} (+{resonance_gain_pct:.2f}% gain)")
        print(
            f"Rekonstruktsiooni SNR tõus: {best_entry['snr_db'] - baseline_entry['snr_db']:+.2f} dB "
            f"({baseline_entry['snr_db']:.2f} dB -> {best_entry['snr_db']:.2f} dB)\n"
        )

        return {
            "sweep_results": sweep_results,
            "baseline_cossim": baseline_cossim,
            "optimal_sigma": optimal_sigma,
            "optimal_cossim": optimal_cossim,
            "resonance_gain_pct": resonance_gain_pct,
        }


if __name__ == "__main__":
    run_stochastic_resonance_benchmark()
