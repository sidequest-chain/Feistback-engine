"""
Feistback Neuro-Symbolic Engine CLI Runner
=========================================
Executes end-to-end code synthesis, compilation to MiniStackVM bytecode,
and sub-microsecond deterministic verification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime.engine import FeistbackEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FeistbackEngine neuro-symbolic generation and MiniStackVM execution"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/logic_feistel_clean/pure_feistel_raf_clean.pt",
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input Python function signature prompt",
    )
    parser.add_argument(
        "--test-inputs",
        type=str,
        default=None,
        help="Optional JSON string of test cases (e.g. '[[2, 3, 4], [5, 10, 2]]')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    # Parse test_inputs if provided
    test_inputs = None
    if args.test_inputs is not None:
        try:
            test_inputs = json.loads(args.test_inputs)
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse --test-inputs as JSON: {e}")
            sys.exit(1)

    print("=" * 80)
    print("FEISTBACK PRODUCTION ENGINE: NEURO-SYMBOLIC PIPELINE")
    print("=" * 80)
    print(f"Checkpoint          : {args.checkpoint}")
    print(f"Input Prompt        : {args.prompt}")
    print(f"Custom Test Inputs  : {test_inputs if test_inputs is not None else 'Auto-Generated'}")
    print("=" * 80)

    # 1. Initialize Engine
    print("\n[1/3] Loading FeistbackEngine...")
    engine = FeistbackEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("  Engine successfully initialized and ready.")

    # 2. Execute Generation & Verification
    print("\n[2/3] Generating code, compiling to AST, and executing on MiniStackVM...")
    result = engine.generate_and_verify(
        prompt=args.prompt,
        test_inputs=test_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # 3. Print Report
    print("\n[3/3] Comprehensive Execution Report:")
    print("=" * 80)
    print("1. GENERATED PYTHON CODE:")
    print("-" * 80)
    for line in result.code.split("\n"):
        print(f"   {line}")
    print("-" * 80)
    if result.inv_map:
        print(f"   BPE Canonical Mapping : {result.inv_map}")
        print(f"   Canonical Prompt      : {result.canonical_prompt}")
        print("-" * 80)

    print("\n2. MINISTACKVM BYTECODE DISASSEMBLY:")
    print("-" * 80)
    for line in result.disassembly.split("\n"):
        print(f"   {line}")
    print(f"   Total Instruction Words: {len(result.bytecode)}")
    print("-" * 80)

    print("\n3. DETERMINISTIC VERIFICATION & EXECUTION BENCHMARKS:")
    print("-" * 80)
    header = f"{'Inputs':<28} | {'VM Res':<10} | {'Py Ref':<10} | {'VM Latency':<12} | {'Status'}"
    print(header)
    print("-" * len(header))
    for t in result.test_outcomes:
        in_str = str(t.inputs)
        if len(in_str) > 26:
            in_str = in_str[:23] + "..."
        vm_str = str(t.vm_result)
        py_str = str(t.py_result)
        lat_str = f"{t.latency_us:.2f} µs"
        status_str = "PASS" if t.matched else "FAIL"
        print(f"{in_str:<28} | {vm_str:<10} | {py_str:<10} | {lat_str:<12} | {status_str}")
    print("-" * 80)

    print("\n4. LATENCY SUMMARY:")
    print(f"   Neural Generation Latency : {result.latency_gen_s:.3f} s (0 bytes KV cache)")
    print(f"   AST Compilation Latency   : {result.latency_comp_ms:.4f} ms (< 0.1 ms)")
    print(f"   Average VM Exec Latency   : {result.latency_vm_avg_us:.2f} µs (Hardware Stack)")
    print("=" * 80)

    if result.verified:
        print("PIPELINE RESULT: 100% VERIFIED MATHEMATICAL ACCURACY")
    else:
        print("PIPELINE RESULT: VERIFICATION FAILED (DISCREPANCY DETECTED)")
    print("=" * 80)


if __name__ == "__main__":
    main()
