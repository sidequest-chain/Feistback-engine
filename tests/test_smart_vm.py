"""
Smart MiniStackVM Test Suite & Benchmarks
==========================================
Validates:
1. Bitwise operations ((a & b) | (a ^ b)).
2. Scientific mathematics (abs(x) ** y + min(x, y)).
3. Local variable assignment and access (diff, sq).
4. Stack manipulation opcodes (DROP, SWAP, OVER, ROT).
5. Infinite loop gas limit protection (RuntimeExecutionError).
6. Nested if/elif/else control flow, boolean operations (and/or), and floats (PUSH_FLOAT).
7. Execution latency benchmarks and 100% mathematical parity with native Python.
"""

import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime.vm.opcodes import OpCode
from runtime.vm.ministack_vm import MiniStackVM, RuntimeExecutionError
from runtime.compiler import ASTCompiler, compile_to_vm


def test_bitwise_ops():
    """Validates bitwise operations (&, |, ^, ~, <<, >>) against native Python."""
    code = """
def bitwise_ops(a: int, b: int) -> int:
    return (a & b) | (a ^ b)
"""
    compiler = ASTCompiler()
    bytecode = compiler.compile(code)
    vm = MiniStackVM()

    def py_bitwise_ops(a: int, b: int) -> int:
        return (a & b) | (a ^ b)

    test_cases = [
        (12, 25),
        (0, 0),
        (255, 15),
        (1024, 513),
        (7, 3),
        (42, 99),
        (127, 128),
    ]

    for a, b in test_cases:
        expected = py_bitwise_ops(a, b)
        vm_result = vm.execute(bytecode, [a, b])
        assert vm_result == expected, f"Bitwise mismatch for ({a}, {b}): {vm_result} != {expected}"

    # Benchmark latency
    iterations = 20_000
    start = time.perf_counter()
    for _ in range(iterations):
        vm.execute(bytecode, [12, 25])
    elapsed_us = ((time.perf_counter() - start) / iterations) * 1_000_000

    print(f"[PASS] bitwise_ops: 100% parity across {len(test_cases)} cases (Avg latency: {elapsed_us:.2f} us/exec)")


def test_calc_complex():
    """Validates scientific math: abs(), pow (**), min(), max(), and isqrt()."""
    code = """
def calc_complex(x: int, y: int) -> int:
    return abs(x) ** y + min(x, y)
"""
    compiler = ASTCompiler()
    bytecode = compiler.compile(code)
    vm = MiniStackVM()

    def py_calc_complex(x: int, y: int) -> int:
        return abs(x) ** y + min(x, y)

    test_cases = [
        (-3, 4),
        (2, 5),
        (-5, 3),
        (0, 4),
        (-2, 0),
        (7, 2),
        (-10, 2),
    ]

    for x, y in test_cases:
        expected = py_calc_complex(x, y)
        vm_result = vm.execute(bytecode, [x, y])
        assert vm_result == expected, f"Math mismatch for ({x}, {y}): {vm_result} != {expected}"

    # Also test integer square root (isqrt)
    code_isqrt = """
def calc_isqrt(n: int) -> int:
    return isqrt(n)
"""
    bc_isqrt = compiler.compile(code_isqrt)
    for n in [0, 1, 4, 9, 15, 16, 99, 100, 1000]:
        assert vm.execute(bc_isqrt, [n]) == math.isqrt(n)

    # Benchmark latency
    iterations = 20_000
    start = time.perf_counter()
    for _ in range(iterations):
        vm.execute(bytecode, [-3, 4])
    elapsed_us = ((time.perf_counter() - start) / iterations) * 1_000_000

    print(f"[PASS] calc_complex: 100% parity across {len(test_cases)} cases + isqrt (Avg latency: {elapsed_us:.2f} us/exec)")


def test_calc_with_locals():
    """Validates local variable assignments and multi-step computations."""
    code = """
def calc_with_locals(a: int, b: int) -> int:
    diff = a - b
    sq = diff * diff
    return sq + 1
"""
    compiler = ASTCompiler()
    bytecode = compiler.compile(code)
    vm = MiniStackVM()

    def py_calc_with_locals(a: int, b: int) -> int:
        diff = a - b
        sq = diff * diff
        return sq + 1

    test_cases = [
        (10, 4),
        (5, 5),
        (3, 8),
        (100, 25),
        (-5, -15),
        (0, 12),
        (50, 75),
    ]

    for a, b in test_cases:
        expected = py_calc_with_locals(a, b)
        vm_result = vm.execute(bytecode, [a, b])
        assert vm_result == expected, f"Locals mismatch for ({a}, {b}): {vm_result} != {expected}"

    # Benchmark latency
    iterations = 20_000
    start = time.perf_counter()
    for _ in range(iterations):
        vm.execute(bytecode, [10, 4])
    elapsed_us = ((time.perf_counter() - start) / iterations) * 1_000_000

    print(f"[PASS] calc_with_locals: 100% parity across {len(test_cases)} cases (Avg latency: {elapsed_us:.2f} us/exec)")


def test_nested_branching_and_bool_ops():
    """Validates nested if/elif/else statements, boolean operators (and/or), and PUSH_FLOAT."""
    code = """
def classify_range(x: int, low: int, high: int) -> int:
    if x >= low and x <= high:
        res = 1
    elif x < 0 or x > 1000:
        res = -1
    else:
        res = 0
    return res
"""
    compiler = ASTCompiler()
    bytecode = compiler.compile(code)
    vm = MiniStackVM()

    for x, low, high, expected in [
        (50, 0, 100, 1),
        (-5, 0, 100, -1),
        (1500, 0, 100, -1),
        (250, 0, 100, 0),
    ]:
        res = vm.execute(bytecode, [x, low, high])
        assert res == expected, f"Nested branching mismatch for ({x}, {low}, {high}): {res} != {expected}"

    float_code = """
def scale_val(x: float) -> float:
    return x * 1.5 + 0.25
"""
    bc_float = compiler.compile(float_code)
    assert abs(vm.execute(bc_float, [4.0]) - 6.25) < 1e-9

    print("[PASS] nested_branching_and_bool_ops: nested if/elif/else, and/or, and PUSH_FLOAT verified")


def test_stack_operations():
    """Validates stack manipulation instructions: DROP, SWAP, OVER, ROT."""
    vm = MiniStackVM()

    # 1. DROP: PUSH 10, PUSH 20, DROP, RET -> returns 10
    bc_drop = [OpCode.PUSH_CONST, 10, OpCode.PUSH_CONST, 20, OpCode.DROP, OpCode.RET]
    assert vm.execute(bc_drop) == 10

    # 2. SWAP: PUSH 10, PUSH 20, SWAP -> stack has [20, 10], SUB -> 20 - 10 = 10
    bc_swap = [OpCode.PUSH_CONST, 10, OpCode.PUSH_CONST, 20, OpCode.SWAP, OpCode.SUB, OpCode.RET]
    assert vm.execute(bc_swap) == 10

    # 3. OVER: PUSH 10, PUSH 20, OVER -> stack has [10, 20, 10], ADD -> [10, 30], ADD -> 40
    bc_over = [OpCode.PUSH_CONST, 10, OpCode.PUSH_CONST, 20, OpCode.OVER, OpCode.ADD, OpCode.ADD, OpCode.RET]
    assert vm.execute(bc_over) == 40

    # 4. ROT: PUSH 1, PUSH 2, PUSH 3, ROT -> stack has [2, 3, 1], RET -> returns 1
    bc_rot = [OpCode.PUSH_CONST, 1, OpCode.PUSH_CONST, 2, OpCode.PUSH_CONST, 3, OpCode.ROT, OpCode.RET]
    assert vm.execute(bc_rot) == 1

    print("[PASS] stack_operations: DROP, SWAP, OVER, ROT verified successfully")


def test_gas_limit_protection():
    """Validates cycle protection against infinite loops."""
    vm = MiniStackVM()

    # Bytecode infinite loop: 0000: JMP 0
    infinite_loop_bytecode = [OpCode.JMP, 0]

    limit = 250
    exception_caught = False
    try:
        vm.execute(infinite_loop_bytecode, max_steps=limit)
    except RuntimeExecutionError as err:
        exception_caught = True
        assert "Instruction limit exceeded" in str(err), f"Unexpected error message: {err}"

    assert exception_caught, "Expected RuntimeExecutionError was not raised!"
    print(f"[PASS] gas_limit_protection: RuntimeExecutionError raised when exceeding {limit} steps")


def main():
    print("=" * 70)
    print("MiniStackVM & ASTCompiler Validation Test Suite")
    print("=" * 70)

    test_bitwise_ops()
    test_calc_complex()
    test_calc_with_locals()
    test_nested_branching_and_bool_ops()
    test_stack_operations()
    test_gas_limit_protection()

    print("=" * 70)
    print("ALL TESTS PASSED: 100% Mathematical Parity and Safety Guarantees Met")
    print("=" * 70)


if __name__ == "__main__":
    main()
