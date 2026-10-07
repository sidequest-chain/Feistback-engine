"""
Benchmark EXP-003: Adaptive Depth and Early-Exit Latency.
Compares static unroll against adaptive early-exit across recurrent steps.
"""

import time
import torch
from benchmarks.tracker import ExperimentTracker
from runner.adaptive_loop import AdaptiveFeistelRunner


def run_adaptive_loop_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running EXP-003 Benchmark on device: {device}")

    d_model = 128
    seq_len = 64
    batch_size = 16
    max_steps = 8
    epsilon = 0.01
    iterations = 50

    runner = AdaptiveFeistelRunner(
        d_model=d_model,
        max_seq_len=seq_len,
        default_max_steps=max_steps,
        default_epsilon=epsilon,
    ).to(device)
    runner.eval()

    torch.manual_seed(42)
    inputs = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = runner.forward_static(inputs, steps=max_steps)
            _ = runner.forward_adaptive(inputs, max_steps=max_steps, epsilon=epsilon)
    if device == "cuda":
        torch.cuda.synchronize()

    with ExperimentTracker(
        "EXP-003-ADAPTIVE-LOOP",
        notes=f"Adaptive Early-Exit (max_steps={max_steps}, eps={epsilon}) vs Static {max_steps} unroll",
    ) as tracker:
        # 1. Benchmark Static Unroll
        static_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = runner.forward_static(inputs, steps=max_steps)
                if device == "cuda":
                    torch.cuda.synchronize()
        static_total_sec = time.perf_counter() - static_start
        static_time_ms = (static_total_sec / iterations) * 1000

        # 2. Benchmark Adaptive Early-Exit
        adaptive_start = time.perf_counter()
        steps_recorded = []
        with torch.no_grad():
            for _ in range(iterations):
                _, steps_taken, _ = runner.forward_adaptive(
                    inputs, max_steps=max_steps, epsilon=epsilon
                )
                steps_recorded.append(steps_taken)
                if device == "cuda":
                    torch.cuda.synchronize()
        adaptive_total_sec = time.perf_counter() - adaptive_start
        adaptive_time_ms = (adaptive_total_sec / iterations) * 1000

        avg_steps = sum(steps_recorded) / len(steps_recorded)
        speedup = static_time_ms / adaptive_time_ms if adaptive_time_ms > 0 else 1.0

        tracker.log_metric("max_steps", max_steps)
        tracker.log_metric("epsilon", epsilon)
        tracker.log_metric("avg_steps", round(avg_steps, 2))
        tracker.log_metric("static_ms", round(static_time_ms, 3))
        tracker.log_metric("adaptive_ms", round(adaptive_time_ms, 3))
        tracker.log_metric("speedup_x", round(speedup, 2))
        tracker.set_status("SUCCESS")

        print("\n" + "=" * 76)
        print("EXP-003: ADAPTIVE DEPTH & EARLY-EXIT BENCHMARK TULEMUSED")
        print("=" * 76)
        print(f"{'Režiim':<24} | {'Sammude arv':<14} | {'Aeg (ms)':<14} | {'Kiiruse suhe':<14}")
        print("-" * 76)
        print(f"{'Staatiline tsükkel':<24} | {max_steps:<14} | {static_time_ms:<14.3f} | {'1.00x (baas)':<14}")
        print(f"{'Adaptiivne Early-Exit':<24} | {avg_steps:<14.1f} | {adaptive_time_ms:<14.3f} | {f'{speedup:.2f}x kiirem':<14}")
        print("=" * 76)
        print(f"Keskmine arvutussammude kokkuhoid: {(1.0 - (avg_steps / max_steps)) * 100:.1f}%\n")


if __name__ == "__main__":
    run_adaptive_loop_benchmark()
