"""
Benchmark EXP-004: MiniStackVM Extended Instruction Set.
Evaluates microsecond-level latency and throughput for JSON extraction,
regular expression matching, and conditional branch loops in pure bytecode.
"""

import time
from typing import Any, Dict, List
from benchmarks.tracker import ExperimentTracker
from mini_vm.ops import OpCode
from mini_vm.vm import MiniStackVM


def run_mini_vm_ext_benchmark() -> Dict[str, Any]:
    print("Running EXP-004 Benchmark: MiniStackVM Extended Instruction Set...")

    vm = MiniStackVM(max_steps=200)
    iterations = 10_000

    # -------------------------------------------------------------
    # Bytecode definitions for Tests A, B, and C
    # -------------------------------------------------------------
    # Test A: JSON Extraction
    log_json = '{"status": 200, "service": "auth", "ip": "192.168.1.1"}'
    bytecode_a: List[Any] = [
        OpCode.PUSH_CONST, log_json,
        OpCode.PUSH_CONST, "status",
        OpCode.LOAD_JSON_KEY,
        OpCode.RET,
    ]

    # Test B: RegEx Pattern Matching
    ip_text = "192.168.1.1"
    pattern = r"^192\.168\.\d+\.\d+$"
    bytecode_b: List[Any] = [
        OpCode.PUSH_CONST, ip_text,
        OpCode.PUSH_CONST, pattern,
        OpCode.REGEX_MATCH,
        OpCode.RET,
    ]

    # Test C: Conditional Branch Loop (10 iterations decrementing from 10 to 0)
    bytecode_c: List[Any] = [
        OpCode.PUSH_CONST, 10,
        OpCode.PUSH_CONST, 1,
        OpCode.SUB,
        OpCode.DUP,
        OpCode.PUSH_CONST, 0,
        OpCode.CMP_GT,
        OpCode.BRANCH_IF, 2,
        OpCode.RET,
    ]

    # Verification / Correctness Check
    res_a = vm.execute(bytecode_a)
    assert res_a == 200, f"Test A assertion failed: expected 200, got {res_a}"

    res_b = vm.execute(bytecode_b)
    assert res_b == 1, f"Test B assertion failed: expected 1, got {res_b}"

    res_c = vm.execute(bytecode_c)
    assert res_c == 0, f"Test C assertion failed: expected 0, got {res_c}"

    # Warmup
    for _ in range(100):
        _ = vm.execute(bytecode_a)
        _ = vm.execute(bytecode_b)
        _ = vm.execute(bytecode_c)

    with ExperimentTracker(
        "EXP-004-MINI-VM-EXT",
        notes="JSON parsing, Regex match, and Branching benchmark",
    ) as tracker:
        # Measure Test A (JSON Extraction)
        start_a = time.perf_counter()
        for _ in range(iterations):
            _ = vm.execute(bytecode_a)
        elapsed_a = time.perf_counter() - start_a
        json_latency_us = (elapsed_a / iterations) * 1_000_000

        # Measure Test B (RegEx Matching)
        start_b = time.perf_counter()
        for _ in range(iterations):
            _ = vm.execute(bytecode_b)
        elapsed_b = time.perf_counter() - start_b
        regex_latency_us = (elapsed_b / iterations) * 1_000_000

        # Measure Test C (Conditional Branch Loop)
        start_c = time.perf_counter()
        for _ in range(iterations):
            _ = vm.execute(bytecode_c)
        elapsed_c = time.perf_counter() - start_c
        branch_loop_latency_us = (elapsed_c / iterations) * 1_000_000

        total_time_sec = elapsed_a + elapsed_b + elapsed_c
        total_ops = 3 * iterations
        throughput_ops_sec = total_ops / total_time_sec if total_time_sec > 0 else 0.0

        tracker.log_metric("json_latency_us", round(json_latency_us, 3))
        tracker.log_metric("regex_latency_us", round(regex_latency_us, 3))
        tracker.log_metric("branch_loop_latency_us", round(branch_loop_latency_us, 3))
        tracker.log_metric("throughput_ops_sec", round(throughput_ops_sec, 1))
        tracker.set_status("SUCCESS")

        print("\n" + "=" * 78)
        print("EXP-004: MINISTACKVM EXTENDED INSTRUCTION SET BENCHMARK RESULTS")
        print("=" * 78)
        print(f"{'Test':<30} | {'Iteratsioone':<14} | {'Kestus (μs)':<14} | {'Läbilaskevõime':<14}")
        print("-" * 78)
        print(f"{'Test A (JSON Extraction)':<30} | {iterations:<14} | {json_latency_us:<14.3f} | {f'{1_000_000 / json_latency_us:.0f} ops/s':<14}")
        print(f"{'Test B (RegEx Match)':<30} | {iterations:<14} | {regex_latency_us:<14.3f} | {f'{1_000_000 / regex_latency_us:.0f} ops/s':<14}")
        print(f"{'Test C (Branch Loop 10 iters)':<30} | {iterations:<14} | {branch_loop_latency_us:<14.3f} | {f'{1_000_000 / branch_loop_latency_us:.0f} ops/s':<14}")
        print("=" * 78)
        print(f"Koondläbilaskevõime: {throughput_ops_sec:,.0f} ops/sec\n")

        return {
            "json_latency_us": json_latency_us,
            "regex_latency_us": regex_latency_us,
            "branch_loop_latency_us": branch_loop_latency_us,
            "throughput_ops_sec": throughput_ops_sec,
        }


if __name__ == "__main__":
    run_mini_vm_ext_benchmark()
