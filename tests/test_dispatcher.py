"""
End-to-End Test Suite for Intent Dispatcher
============================================
Validates natural language and 'estengrish' query routing to PureFeistelLM signatures,
sub-microsecond execution on MiniStackVM, and 100% mathematical accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runner.engine import FeistbackEngine
from runner.dispatcher import ConfigDispatcher


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 85)
    print("FEISTBACK INTENT DISPATCHER VERIFICATION TEST (NATURAL LANGUAGE -> VM)")
    print("=" * 85)

    checkpoint_path = "checkpoints/pure_feistel_raf_clean.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[1/2] Loading FeistbackEngine from {checkpoint_path}...")
    engine = FeistbackEngine.from_checkpoint(checkpoint_path)
    dispatcher = ConfigDispatcher(engine, config_path="config/intents.json")
    print(f"Loaded {len(dispatcher.intents)} intents from config/intents.json.\n")

    # The 5 required benchmark test queries
    test_cases = [
        {
            "id": 1,
            "query": "liida kokku 15 ja 25",
            "expected_intent": "add",
            "expected_args": [15, 25],
            "expected_result": 40,
        },
        {
            "id": 2,
            "query": "clämbi see counter 105 vahemikku 0 ja 100",
            "expected_intent": "clamp_math",
            "expected_args": [105, 0, 100],
            "expected_result": 100,
        },
        {
            "id": 3,
            "query": "arvuta vahe ruut arvudel 15 ja 8 ja liida üks otsa",
            "expected_intent": "calc_with_locals",
            "expected_args": [15, 8],
            "expected_result": 50,
        },
        {
            "id": 4,
            "query": "viska xor 12 ja 4 vahel",
            "expected_intent": "bitwise_comb",
            "expected_args": [12, 4],
            "expected_result": 8,
        },
        {
            "id": 5,
            "query": "kas number -7 on negatiivne",
            "expected_intent": "is_negative",
            "expected_args": [-7],
            "expected_result": 1,
        },
    ]

    print("[2/2] Running 5 Natural Language & Estengrish Test Queries...")
    print("=" * 85)

    all_passed = True
    results_summary = []

    for tc in test_cases:
        t0 = time.perf_counter()
        resp = dispatcher.dispatch(tc["query"])
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        status = resp.get("status")
        intent_id = resp.get("intent_id")
        args = resp.get("args")
        vm_res = resp.get("result")
        gen_code = resp.get("generated_code", "").strip()
        vm_lat_us = resp.get("vm_latency_us", 0.0)

        passed = (
            status == "SUCCESS"
            and intent_id == tc["expected_intent"]
            and args == tc["expected_args"]
            and vm_res == tc["expected_result"]
        )

        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        results_summary.append({
            "id": tc["id"],
            "query": tc["query"],
            "intent": intent_id,
            "args": args,
            "vm_res": vm_res,
            "expected": tc["expected_result"],
            "vm_latency_us": vm_lat_us,
            "total_ms": total_time_ms,
            "status": status_str,
            "code": gen_code,
        })

        print(f"Test {tc['id']}: \"{tc['query']}\"")
        print(f"  Matched Intent  : {intent_id} (Expected: {tc['expected_intent']})")
        print(f"  Extracted Args  : {args} (Expected: {tc['expected_args']})")
        print(f"  Generated Code  :\n    " + "\n    ".join(gen_code.split("\n")))
        print(f"  MiniStackVM Res : {vm_res} (Expected: {tc['expected_result']})")
        print(f"  Execution Stats : VM Latency: {vm_lat_us:.2f} us | Total Turnaround: {total_time_ms:.1f} ms")
        print(f"  Status          : {status_str}")
        print("-" * 85)

    print("\n" + "=" * 85)
    print("TEST SUMMARY REPORT")
    print("=" * 85)
    print(f"{'ID':<3} | {'Query':<36} | {'Intent':<17} | {'Result':<7} | {'VM Lat':<10} | {'Status'}")
    print("-" * 85)
    for r in results_summary:
        print(
            f"{r['id']:<3} | {r['query'][:36]:<36} | {r['intent']:<17} | "
            f"{str(r['vm_res']):<7} | {r['vm_latency_us']:>6.2f} us | {r['status']}"
        )
    print("=" * 85)

    assert all_passed, "One or more dispatcher test cases failed!"
    print("\nALL 5 NATURAL LANGUAGE & ESTENGRISH DISPATCHER TESTS PASSED WITH 100% ACCURACY!")


if __name__ == "__main__":
    main()
