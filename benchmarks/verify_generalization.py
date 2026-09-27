"""
Verify Dynamic Generalization with LOAD_ARG Registers
=====================================================
Evaluates the trained FeistelHybrid pipeline on out-of-distribution numbers:
  1. Uses HybridPipeline to parse arguments and execute in MiniStackVM.
  2. Tests 50 held-out tasks with 4-digit numbers (1000 - 9999).
  3. Verifies 100% mathematical accuracy.
"""

import os
import sys
import time
import random
from typing import List, Dict, Any, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
except ImportError:
    print("❌ PyTorch is not installed in the current Python environment.")
    print("Please install PyTorch using: pip install torch")
    sys.exit(1)

from core import FeistelHolographicLM
from mini_vm import MiniStackVM
from runner import HybridTokenizer, HybridPipeline
from trainer.dynamic_dataset import PROMPT_TEMPLATES, OP_TYPES, calculate_op


def load_pipeline(device: str = "cuda") -> Optional[HybridPipeline]:
    """Loads model from checkpoints/feistel_dynamic_hybrid.pt and creates HybridPipeline."""
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "feistel_dynamic_hybrid.pt")
    if not os.path.exists(ckpt_path):
        print(f"⚠️  Checkpoint not found at: {ckpt_path}")
        print("    Train a checkpoint first using: python -m trainer.train_hybrid")
        return None

    tokenizer = HybridTokenizer(num_registers=4)
    vm = MiniStackVM()

    checkpoint = torch.load(ckpt_path, map_location=device)
    max_seq_len = checkpoint.get("max_seq_len", 112)

    model = FeistelHolographicLM(
        vocab_size=tokenizer.vocab_size,
        d_model=checkpoint.get("d_model", 128),
        num_layers=checkpoint.get("num_layers", 16),
        max_seq_len=max_seq_len,
        activation="silu",
        layer_scale=1.0
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from: {ckpt_path} (epoch {checkpoint.get('epochs', '?')})", flush=True)

    return HybridPipeline(
        model=model,
        tokenizer=tokenizer,
        vm=vm,
        device=device,
        max_seq_len=max_seq_len
    )


def generate_unseen_large_tasks(num_tasks: int = 50, seed: int = 2026) -> List[Dict[str, Any]]:
    """
    Generates 50 held-out tasks strictly with large 4-digit numbers (1000 - 9999).
    Completely unseen during training (which only saw [1, 500]).
    """
    rng = random.Random(seed)
    tasks = []

    for _ in range(num_tasks):
        op = rng.choice(OP_TYPES)
        a = rng.randint(1000, 9999)
        b = rng.randint(1000, 9999)

        if op == "sub" and b > a:
            a, b = b, a

        prompt_template = rng.choice(PROMPT_TEMPLATES[op])
        prompt = prompt_template.format(a=a, b=b)
        expected_val = calculate_op(op, a, b)

        tasks.append({
            "op": op,
            "a": a,
            "b": b,
            "prompt": prompt,
            "expected_val": expected_val
        })

    return tasks


def run_verification(device: str = "cuda") -> None:
    """Executes verification test suite on unseen 4-digit numeric tasks."""
    print("=" * 95)
    print("  OUT-OF-DISTRIBUTION 4-DIGIT ARITHMETIC GENERALIZATION BENCHMARK")
    print("=" * 95)

    forced = os.environ.get("DEVICE")
    if forced:
        device = forced
    elif device == "cuda":
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            try:
                t = torch.zeros(1, device="cuda")
                del t
            except Exception:
                print("  ℹ️ CUDA available but GPU kernels not supported by installed PyTorch wheel. Falling back to CPU.")
                device = "cpu"

    pipeline = load_pipeline(device=device)
    if pipeline is None:
        print("=" * 95)
        return

    tasks = generate_unseen_large_tasks(num_tasks=50, seed=2026)
    print(f"\nEvaluating {len(tasks)} unseen 4-digit tasks on {device.upper()}...")
    print("-" * 95)
    print(f"{'#':<4} | {'Status':<8} | {'Prompt':<36} | {'Registers':<14} | {'VM Res':<12} | {'Expected':<12}")
    print("-" * 95)

    results = []

    for i, task in enumerate(tasks, start=1):
        res = pipeline.run_task(task["prompt"])
        is_pass = (res["vm_result"] == task["expected_val"])

        results.append({
            "task_idx": i,
            "op": task["op"],
            "prompt": task["prompt"],
            "expected": task["expected_val"],
            "vm_result": res["vm_result"],
            "is_pass": is_pass,
            "total_time_ms": res["total_time_ms"],
            "vm_time_ms": res["vm_time_ms"]
        })

        status = "✅ PASS" if is_pass else "❌ FAIL"
        prompt_short = (task["prompt"][:33] + "...") if len(task["prompt"]) > 36 else task["prompt"]
        args_str = str(res["args"])
        print(f"{i:<4} | {status:<8} | {prompt_short:<36} | {args_str:<14} | {str(res['vm_result']):<12} | {str(task['expected_val']):<12}")

    passed_count = sum(1 for r in results if r["is_pass"])
    accuracy = (passed_count / len(results)) * 100.0
    avg_latency = sum(r["total_time_ms"] for r in results) / len(results)
    avg_vm_latency = sum(r["vm_time_ms"] for r in results) / len(results)

    print("=" * 95)
    print("                               FINAL SUMMARY")
    print("=" * 95)
    print(f"  Total tasks:               {len(results)}")
    print(f"  Passed (PASS):             {passed_count} / {len(results)}")
    print(f"  Generalization accuracy:   {accuracy:.1f}%")
    print(f"  Average model latency:     {avg_latency:.2f} ms")
    print(f"  Average VM execution:      {avg_vm_latency * 1000.0:.2f} µs")
    print("=" * 95)

    if accuracy == 100.0:
        print("\n🎉 SUCCESS: Feistback engine achieved 100.0% accuracy on unseen 4-digit numbers!")
    else:
        print(f"\n⚠️ Accuracy was {accuracy:.1f}%.")


if __name__ == "__main__":
    run_verification()
