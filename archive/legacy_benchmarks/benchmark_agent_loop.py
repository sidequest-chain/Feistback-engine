"""
Benchmark EXP-008: Autonomous Feistel Agent Reasoning Loop.
===========================================================
Validates the full perception-deliberation-action cycle by integrating:
  - EXP-001/002: Reversible Feistel network with zero KV-cache growth.
  - EXP-003: Adaptive latent deliberation runner with Poincaré attractor early-exit.
  - EXP-004: MiniStackVM with extended symbolic operations (JSON, RegEx, Branching).
  - EXP-007: Heterogeneous async CPU-GPU execution pipeline.
Simulates 20 multi-stage autonomous agent tasks (log parsing, threshold evaluation,
branch dispatch, and latent state updates).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple
import torch

from benchmarks.tracker import ExperimentTracker
from mini_vm.ops import OpCode
from runner.agent_loop import AutonomousFeistelAgent


def generate_agent_log_task(step: int) -> Tuple[List[Any], str]:
    """
    Generates realistic multi-stage log analysis bytecode and semantic operation label.
    Simulates:
      - JSON extraction (parsing metadata/latency)
      - SLA Threshold check (numerical comparison)
      - Control-flow branching (alarm dispatch)
      - RegEx pattern parsing (client IP extraction)
    """
    stage = (step - 1) % 4
    if stage == 0:
        json_log = f'{{"log_id": {step}, "status": {500 if step % 3 == 0 else 200}, "latency": {120 + step * 4}, "sla": 150}}'
        bytecode = [
            OpCode.PUSH_CONST, json_log,
            OpCode.PUSH_CONST, "latency",
            OpCode.LOAD_JSON_KEY,
            OpCode.RET,
        ]
        return bytecode, "JSON_EXTRACT"
    elif stage == 1:
        latency = 120 + (step - 1) * 4
        sla = 150
        bytecode = [
            OpCode.PUSH_CONST, latency,
            OpCode.PUSH_CONST, sla,
            OpCode.CMP_GT,
            OpCode.RET,
        ]
        return bytecode, "THRESHOLD_CMP"
    elif stage == 2:
        cond = 1 if (120 + (step - 2) * 4) > 150 else 0
        bytecode = [
            OpCode.PUSH_CONST, cond,
            OpCode.BRANCH_IF, 5,
            OpCode.PUSH_CONST, 0,
            OpCode.RET,
            OpCode.PUSH_CONST, 999,
            OpCode.RET,
        ]
        return bytecode, "BRANCH_DECISION"
    else:
        log_line = f"2026-10-05 [WARN] client_ip=10.0.0.{step} metric=CPU_LOAD"
        bytecode = [
            OpCode.PUSH_CONST, r"client_ip=([0-9.]+)",
            OpCode.PUSH_CONST, log_line,
            OpCode.REGEX_MATCH,
            OpCode.RET,
        ]
        return bytecode, "REGEX_MATCH"


def run_agent_loop_benchmark(
    num_steps: int = 20,
    d_model: int = 128,
    seq_len: int = 64,
    max_deliberation_steps: int = 8,
    epsilon: float = 0.01,
) -> Dict[str, Any]:
    """
    Executes and benchmarks 20 multi-stage autonomous agent cycles.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing AutonomousFeistelAgent on device: {device}...")

    agent = AutonomousFeistelAgent(
        d_model=d_model,
        seq_len=seq_len,
        max_deliberation_steps=max_deliberation_steps,
        epsilon=epsilon,
        device=device,
    )

    # Initial token prompt representing task query
    initial_tokens = torch.randint(0, 100, (1, seq_len), device=agent.device)

    # Warmup agent pipeline (3 steps)
    print("Warming up agent perception, deliberation, and async pipeline...")
    _ = agent.run_agent_task(
        initial_tokens=initial_tokens,
        num_agent_steps=3,
        bytecode_fn=generate_agent_log_task,
    )
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"Running EXP-008 Autonomous Agent Loop Benchmark ({num_steps} steps)...")

    with ExperimentTracker(
        "EXP-008-AGENT-LOOP",
        notes="Autonomous Agent perception-deliberation-action cycle benchmark",
    ) as tracker:
        result = agent.run_agent_task(
            initial_tokens=initial_tokens,
            num_agent_steps=num_steps,
            bytecode_fn=generate_agent_log_task,
        )

        history = result["step_history"]
        mean_step_time_ms = sum(h["step_time_ms"] for h in history) / len(history)
        mean_delib_depth = sum(h["deliberation_steps"] for h in history) / len(history)
        mean_vm_us = sum(h["vm_time_us"] for h in history) / len(history)

        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_vram_mb = 0.0

        # Log primary benchmark metrics
        tracker.log_metric("mean_step_time_ms", round(mean_step_time_ms, 3))
        tracker.log_metric("mean_deliberation_depth", round(mean_delib_depth, 2))
        tracker.log_metric("vm_latency_us", round(mean_vm_us, 2))
        tracker.log_metric("peak_vram_mb", round(peak_vram_mb, 2))
        tracker.set_status("SUCCESS")

        # Print formatted Markdown / ASCII Table
        print("\n" + "=" * 80)
        print("EXP-008: AUTONOMOUS FEISTEL AGENT REASONING LOOP BENCHMARK")
        print("=" * 80)
        print(f"{'Agendi samm':<12} | {'Latentne sügavus':<17} | {'VM operatsioon':<16} | {'Tsükli kestus (ms)':<19} | {'Olek':<10}")
        print("-" * 80)
        for row in history:
            print(
                f"{row['step']:<12} | "
                f"{row['deliberation_steps']:<17} | "
                f"{row['vm_op']:<16} | "
                f"{row['step_time_ms']:<19.3f} | "
                f"{row['status']:<10}"
            )
        print("=" * 80)
        print(f"Keskmine agenditsükli sammu kestus: {mean_step_time_ms:.3f} ms/samm")
        print(f"Keskmine latentse kaalutluse sügavus: {mean_delib_depth:.2f} steps/token")
        print(f"Deterministliku VM-i täitmise latentsus: {mean_vm_us:.2f} µs")
        print(f"VRAM-i tippkulu: {peak_vram_mb:.2f} MB\n")

    agent.close()

    return {
        "mean_step_time_ms": mean_step_time_ms,
        "mean_deliberation_depth": mean_delib_depth,
        "vm_latency_us": mean_vm_us,
        "peak_vram_mb": peak_vram_mb,
        "history": history,
    }


if __name__ == "__main__":
    run_agent_loop_benchmark()
