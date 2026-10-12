"""
Synthetic Bytecode Dataset Generator for MiniStackVM (RFC-002 Tier 3)
=====================================================================
Generates verified synthetic instruction-contract training pairs for training
the specialized logic core to translate Structured Execution Contracts
(intent signatures) into deterministic MiniStackVM assembly mnemonics.

Dataset format per line:
{
    "prompt": "CONTRACT: <intent_signature> -> BYTECODE:\n",
    "completion": "<mnemonics_string>\n",
    "intent": "<intent_name>",
    "test_cases": [[[arg1, arg2, ...], expected_result], ...]
}

Every sample is compiled with ASTCompiler, converted to mnemonics, re-assembled
with MiniStackVM.assemble_mnemonics, and 100% verified against native Python ground truth.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.compiler import ASTCompiler
from runtime.vm.ministack_vm import MiniStackVM, ARG_OPCODES, OPCODE_NAMES, OpCode


OPCODE_TO_MNEMONIC: Dict[int, str] = {
    OpCode.PUSH_CONST: "LOAD_CONST",
    OpCode.PUSH_FLOAT: "LOAD_CONST",
    OpCode.JMP: "JUMP",
    OpCode.JMP_IF_FALSE: "JUMP_IF_FALSE",
}


def bytecode_to_mnemonics(bytecode: List[Any]) -> str:
    """
    Translates raw compiler bytecode into space-separated mnemonic text.
    Uses canonical mnemonic names (LOAD_CONST, JUMP, JUMP_IF_FALSE).
    """
    tokens: List[str] = []
    ip = 0
    n = len(bytecode)
    while ip < n:
        op = bytecode[ip]
        ip += 1
        name = OPCODE_TO_MNEMONIC.get(op, OPCODE_NAMES.get(op, f"UNK_{op}"))
        tokens.append(name)
        if op in ARG_OPCODES:
            if ip < n:
                arg_val = bytecode[ip]
                ip += 1
                tokens.append(str(arg_val))
    return " ".join(tokens)


# Variable name pools for orthogonal desynchronization
SINGLE_VARS = ["a", "x", "n", "val", "num", "v", "counter_val", "item", "arg0"]
BINARY_VARS = [
    ("a", "b"),
    ("x", "y"),
    ("u", "v"),
    ("num_a", "num_b"),
    ("val1", "val2"),
    ("p", "q"),
    ("left", "right"),
    ("first", "second"),
]
TRIPLE_VARS = [
    ("a", "b", "c"),
    ("x", "y", "z"),
    ("u", "v", "w"),
    ("val1", "val2", "val3"),
    ("first", "second", "third"),
]
QUAD_VARS = [
    ("a", "b", "c", "d"),
    ("x", "y", "z", "w"),
    ("v1", "v2", "v3", "v4"),
    ("n1", "n2", "n3", "n4"),
]


class TemplateSpec:
    """Specification for a parameterized algorithmic template."""

    def __init__(
        self,
        intent: str,
        arity: int,
        ret_type: str,
        code_fn: Callable[..., str],
        eval_fn: Callable[..., Any],
        arg_gen_fn: Callable[[], List[Any]],
        arg_types: Optional[List[str]] = None,
    ) -> None:
        self.intent = intent
        self.arity = arity
        self.ret_type = ret_type
        self.code_fn = code_fn
        self.eval_fn = eval_fn
        self.arg_gen_fn = arg_gen_fn
        self.arg_types = arg_types or ["int"] * arity


def create_template_registry() -> List[TemplateSpec]:
    """Builds the comprehensive pool of mathematical and logical templates."""
    specs: List[TemplateSpec] = []

    # =========================================================================
    # 1. Base Arithmetic Operations
    # =========================================================================
    specs.append(TemplateSpec(
        intent="add",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def add({u}: int, {v}: int) -> int:\n    return {u} + {v}",
        eval_fn=lambda u, v: u + v,
        arg_gen_fn=lambda: [random.randint(-100, 150), random.randint(-100, 150)],
    ))

    specs.append(TemplateSpec(
        intent="sub",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def sub({u}: int, {v}: int) -> int:\n    return {u} - {v}",
        eval_fn=lambda u, v: u - v,
        arg_gen_fn=lambda: [random.randint(-100, 150), random.randint(-100, 150)],
    ))

    specs.append(TemplateSpec(
        intent="subtract",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def subtract({u}: int, {v}: int) -> int:\n    return {u} - {v}",
        eval_fn=lambda u, v: u - v,
        arg_gen_fn=lambda: [random.randint(-100, 150), random.randint(-100, 150)],
    ))

    specs.append(TemplateSpec(
        intent="mul",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def mul({u}: int, {v}: int) -> int:\n    return {u} * {v}",
        eval_fn=lambda u, v: u * v,
        arg_gen_fn=lambda: [random.randint(-50, 50), random.randint(-50, 50)],
    ))

    specs.append(TemplateSpec(
        intent="multiply",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def multiply({u}: int, {v}: int) -> int:\n    return {u} * {v}",
        eval_fn=lambda u, v: u * v,
        arg_gen_fn=lambda: [random.randint(-50, 50), random.randint(-50, 50)],
    ))

    specs.append(TemplateSpec(
        intent="div",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def div({u}: int, {v}: int) -> int:\n    return {u} // {v}",
        eval_fn=lambda u, v: u // v if v != 0 else 0,
        arg_gen_fn=lambda: [random.randint(-100, 200), random.choice([x for x in range(-20, 21) if x != 0])],
    ))

    specs.append(TemplateSpec(
        intent="integer_divide",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def integer_divide({u}: int, {v}: int) -> int:\n    return {u} // {v}",
        eval_fn=lambda u, v: u // v if v != 0 else 0,
        arg_gen_fn=lambda: [random.randint(-100, 200), random.choice([x for x in range(-20, 21) if x != 0])],
    ))

    specs.append(TemplateSpec(
        intent="mod",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def mod({u}: int, {v}: int) -> int:\n    return {u} % {v}",
        eval_fn=lambda u, v: u % v if v != 0 else 0,
        arg_gen_fn=lambda: [random.randint(0, 200), random.choice([x for x in range(1, 30)])],
    ))

    specs.append(TemplateSpec(
        intent="modulo",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def modulo({u}: int, {v}: int) -> int:\n    return {u} % {v}",
        eval_fn=lambda u, v: u % v if v != 0 else 0,
        arg_gen_fn=lambda: [random.randint(0, 200), random.choice([x for x in range(1, 30)])],
    ))

    specs.append(TemplateSpec(
        intent="pow",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def pow({u}: int, {v}: int) -> int:\n    return {u} ** {v}",
        eval_fn=lambda u, v: u ** v,
        arg_gen_fn=lambda: [random.randint(-5, 8), random.randint(0, 4)],
    ))

    specs.append(TemplateSpec(
        intent="power",
        arity=2,
        ret_type="int",
        code_fn=lambda u, v: f"def power({u}: int, {v}: int) -> int:\n    return {u} ** {v}",
        eval_fn=lambda u, v: u ** v,
        arg_gen_fn=lambda: [random.randint(-5, 8), random.randint(0, 4)],
    ))

    specs.append(TemplateSpec(
        intent="negate",
        arity=1,
        ret_type="int",
        code_fn=lambda v: f"def negate({v}: int) -> int:\n    return -{v}",
        eval_fn=lambda v: -v,
        arg_gen_fn=lambda: [random.randint(-150, 150)],
    ))

    specs.append(TemplateSpec(
        intent="abs_val",
        arity=1,
        ret_type="int",
        code_fn=lambda v: f"def abs_val({v}: int) -> int:\n    return abs({v})",
        eval_fn=lambda v: abs(v),
        arg_gen_fn=lambda: [random.randint(-200, 200)],
    ))

    # =========================================================================
    # 2. Kernel Fusion & Compound Operations
    # =========================================================================
    specs.append(TemplateSpec(
        intent="sum3_sub",
        arity=4,
        ret_type="int",
        code_fn=lambda a, b, c, d: f"def sum3_sub({a}: int, {b}: int, {c}: int, {d}: int) -> int:\n    return ({a} + {b} + {c}) - {d}",
        eval_fn=lambda a, b, c, d: (a + b + c) - d,
        arg_gen_fn=lambda: [random.randint(-50, 80) for _ in range(4)],
    ))

    specs.append(TemplateSpec(
        intent="add_mul",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def add_mul({a}: int, {b}: int, {c}: int) -> int:\n    return ({a} + {b}) * {c}",
        eval_fn=lambda a, b, c: (a + b) * c,
        arg_gen_fn=lambda: [random.randint(-20, 30), random.randint(-20, 30), random.randint(-10, 15)],
    ))

    specs.append(TemplateSpec(
        intent="mul_add",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def mul_add({a}: int, {b}: int, {c}: int) -> int:\n    return ({a} * {b}) + {c}",
        eval_fn=lambda a, b, c: (a * b) + c,
        arg_gen_fn=lambda: [random.randint(-20, 25), random.randint(-20, 25), random.randint(-50, 50)],
    ))

    specs.append(TemplateSpec(
        intent="sub_mul",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def sub_mul({a}: int, {b}: int, {c}: int) -> int:\n    return ({a} - {b}) * {c}",
        eval_fn=lambda a, b, c: (a - b) * c,
        arg_gen_fn=lambda: [random.randint(-30, 30), random.randint(-30, 30), random.randint(-10, 15)],
    ))

    specs.append(TemplateSpec(
        intent="sum3",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def sum3({a}: int, {b}: int, {c}: int) -> int:\n    return {a} + {b} + {c}",
        eval_fn=lambda a, b, c: a + b + c,
        arg_gen_fn=lambda: [random.randint(-80, 100) for _ in range(3)],
    ))

    specs.append(TemplateSpec(
        intent="mul3",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def mul3({a}: int, {b}: int, {c}: int) -> int:\n    return {a} * {b} * {c}",
        eval_fn=lambda a, b, c: a * b * c,
        arg_gen_fn=lambda: [random.randint(-10, 15) for _ in range(3)],
    ))

    specs.append(TemplateSpec(
        intent="sum4",
        arity=4,
        ret_type="int",
        code_fn=lambda a, b, c, d: f"def sum4({a}: int, {b}: int, {c}: int, {d}: int) -> int:\n    return {a} + {b} + {c} + {d}",
        eval_fn=lambda a, b, c, d: a + b + c + d,
        arg_gen_fn=lambda: [random.randint(-50, 80) for _ in range(4)],
    ))

    specs.append(TemplateSpec(
        intent="calc_with_locals",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def calc_with_locals({a}: int, {b}: int) -> int:\n    diff = {a} - {b}\n    term = diff * diff\n    return term + 1",
        eval_fn=lambda a, b: (a - b) * (a - b) + 1,
        arg_gen_fn=lambda: [random.randint(0, 50), random.randint(0, 50)],
    ))

    specs.append(TemplateSpec(
        intent="clamp_math",
        arity=3,
        ret_type="int",
        code_fn=lambda u, v, w: f"def clamp_math({u}: int, {v}: int, {w}: int) -> int:\n    return min(max({u}, {v}), {w})",
        eval_fn=lambda u, v, w: min(max(u, v), w),
        arg_gen_fn=lambda: [random.randint(-50, 150), 0, 100],
    ))

    specs.append(TemplateSpec(
        intent="clamp_with_locals",
        arity=3,
        ret_type="int",
        code_fn=lambda u, v, w: f"def clamp_with_locals({u}: int, {v}: int, {w}: int) -> int:\n    val = max({u}, {v})\n    res = min(val, {w})\n    return res",
        eval_fn=lambda u, v, w: min(max(u, v), w),
        arg_gen_fn=lambda: [random.randint(-50, 150), 0, 100],
    ))

    specs.append(TemplateSpec(
        intent="abs_diff_pow",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def abs_diff_pow({a}: int, {b}: int) -> int:\n    diff = abs({a} - {b})\n    res = diff ** 2\n    return res",
        eval_fn=lambda a, b: (abs(a - b)) ** 2,
        arg_gen_fn=lambda: [random.randint(0, 40), random.randint(0, 40)],
    ))

    specs.append(TemplateSpec(
        intent="scaled_diff",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def scaled_diff({a}: int, {b}: int) -> int:\n    diff = {a} - {b}\n    res = diff * 3\n    return res + {a}",
        eval_fn=lambda a, b: (a - b) * 3 + a,
        arg_gen_fn=lambda: [random.randint(-30, 50), random.randint(-30, 50)],
    ))

    specs.append(TemplateSpec(
        intent="poly_eval",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def poly_eval({a}: int, {b}: int, {c}: int) -> int:\n    term = {a} * {a}\n    res = {b} * term\n    return res + {c}",
        eval_fn=lambda a, b, c: b * (a * a) + c,
        arg_gen_fn=lambda: [random.randint(0, 12), random.randint(1, 5), random.randint(0, 30)],
    ))

    specs.append(TemplateSpec(
        intent="sum_three_scaled",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def sum_three_scaled({a}: int, {b}: int, {c}: int) -> int:\n    total = {a} + {b} + {c}\n    res = total * 2\n    return res",
        eval_fn=lambda a, b, c: (a + b + c) * 2,
        arg_gen_fn=lambda: [random.randint(-30, 50) for _ in range(3)],
    ))

    specs.append(TemplateSpec(
        intent="linear_blend",
        arity=3,
        ret_type="int",
        code_fn=lambda a, b, c: f"def linear_blend({a}: int, {b}: int, {c}: int) -> int:\n    d1 = {a} * {c}\n    d2 = {b} * (100 - {c})\n    return (d1 + d2) // 100",
        eval_fn=lambda a, b, c: (a * c + b * (100 - c)) // 100,
        arg_gen_fn=lambda: [random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)],
    ))

    specs.append(TemplateSpec(
        intent="norm_l1",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def norm_l1({a}: int, {b}: int) -> int:\n    d1 = abs({a})\n    d2 = abs({b})\n    return d1 + d2",
        eval_fn=lambda a, b: abs(a) + abs(b),
        arg_gen_fn=lambda: [random.randint(-100, 100), random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="average_squares",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def average_squares({a}: int, {b}: int) -> int:\n    d1 = {a} * {a}\n    d2 = {b} * {b}\n    return (d1 + d2) // 2",
        eval_fn=lambda a, b: (a * a + b * b) // 2,
        arg_gen_fn=lambda: [random.randint(0, 50), random.randint(0, 50)],
    ))

    specs.append(TemplateSpec(
        intent="hypot_sqr",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def hypot_sqr({a}: int, {b}: int) -> int:\n    return {a} * {a} + {b} * {b}",
        eval_fn=lambda a, b: a * a + b * b,
        arg_gen_fn=lambda: [random.randint(-25, 25), random.randint(-25, 25)],
    ))

    specs.append(TemplateSpec(
        intent="diff_sqr",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def diff_sqr({a}: int, {b}: int) -> int:\n    diff = {a} - {b}\n    return diff * diff",
        eval_fn=lambda a, b: (a - b) * (a - b),
        arg_gen_fn=lambda: [random.randint(-25, 25), random.randint(-25, 25)],
    ))

    specs.append(TemplateSpec(
        intent="cube_plus_square",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def cube_plus_square({a}: int) -> int:\n    term = {a} * {a}\n    res = term * {a}\n    return res + term",
        eval_fn=lambda a: a * a * a + a * a,
        arg_gen_fn=lambda: [random.randint(0, 12)],
    ))

    specs.append(TemplateSpec(
        intent="min_max_delta",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def min_max_delta({a}: int, {b}: int) -> int:\n    lo = min({a}, {b})\n    hi = max({a}, {b})\n    return hi - lo",
        eval_fn=lambda a, b: max(a, b) - min(a, b),
        arg_gen_fn=lambda: [random.randint(-50, 100), random.randint(-50, 100)],
    ))

    specs.append(TemplateSpec(
        intent="double_val",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def double_val({a}: int) -> int:\n    return {a} * 2",
        eval_fn=lambda a: a * 2,
        arg_gen_fn=lambda: [random.randint(-100, 150)],
    ))

    specs.append(TemplateSpec(
        intent="triple_val",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def triple_val({a}: int) -> int:\n    return {a} * 3",
        eval_fn=lambda a: a * 3,
        arg_gen_fn=lambda: [random.randint(-100, 150)],
    ))

    # =========================================================================
    # 3. Bitwise & Logical Comparison Operations
    # =========================================================================
    specs.append(TemplateSpec(
        intent="bitwise_comb",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def bitwise_comb({a}: int, {b}: int) -> int:\n    return ({a} & {b}) | ({a} ^ {b})",
        eval_fn=lambda a, b: (a & b) | (a ^ b),
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="bitwise_and",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def bitwise_and({a}: int, {b}: int) -> int:\n    return {a} & {b}",
        eval_fn=lambda a, b: a & b,
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="bitwise_or",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def bitwise_or({a}: int, {b}: int) -> int:\n    return {a} | {b}",
        eval_fn=lambda a, b: a | b,
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="bitwise_xor",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def bitwise_xor({a}: int, {b}: int) -> int:\n    return {a} ^ {b}",
        eval_fn=lambda a, b: a ^ b,
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="bitwise_not",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def bitwise_not({a}: int) -> int:\n    return ~{a}",
        eval_fn=lambda a: ~a,
        arg_gen_fn=lambda: [random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="shift_left",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def shift_left({a}: int, {b}: int) -> int:\n    return {a} << ({b} & 15)",
        eval_fn=lambda a, b: a << (b & 15),
        arg_gen_fn=lambda: [random.randint(0, 127), random.randint(0, 7)],
    ))

    specs.append(TemplateSpec(
        intent="shift_right",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def shift_right({a}: int, {b}: int) -> int:\n    return {a} >> ({b} & 15)",
        eval_fn=lambda a, b: a >> (b & 15),
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 7)],
    ))

    specs.append(TemplateSpec(
        intent="shift_and_mask",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def shift_and_mask({a}: int, {b}: int) -> int:\n    shift = {a} << 2\n    res = shift & {b}\n    return res",
        eval_fn=lambda a, b: (a << 2) & b,
        arg_gen_fn=lambda: [random.randint(0, 63), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="bitwise_combine_locals",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def bitwise_combine_locals({a}: int, {b}: int) -> int:\n    mask = {a} & {b}\n    res = mask ^ ({a} | {b})\n    return res",
        eval_fn=lambda a, b: (a & b) ^ (a | b),
        arg_gen_fn=lambda: [random.randint(0, 255), random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="clear_lowest_bit",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def clear_lowest_bit({a}: int) -> int:\n    return {a} & ({a} - 1)",
        eval_fn=lambda a: a & (a - 1),
        arg_gen_fn=lambda: [random.randint(1, 255)],
    ))

    specs.append(TemplateSpec(
        intent="mask_low_byte",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def mask_low_byte({a}: int) -> int:\n    return {a} & 255",
        eval_fn=lambda a: a & 255,
        arg_gen_fn=lambda: [random.randint(0, 1023)],
    ))

    specs.append(TemplateSpec(
        intent="mask_low_nibble",
        arity=1,
        ret_type="int",
        code_fn=lambda a: f"def mask_low_nibble({a}: int) -> int:\n    return {a} & 15",
        eval_fn=lambda a: a & 15,
        arg_gen_fn=lambda: [random.randint(0, 255)],
    ))

    specs.append(TemplateSpec(
        intent="is_negative",
        arity=1,
        ret_type="bool",
        code_fn=lambda a: f"def is_negative({a}: int) -> bool:\n    return {a} < 0",
        eval_fn=lambda a: 1 if a < 0 else 0,
        arg_gen_fn=lambda: [random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="is_positive",
        arity=1,
        ret_type="bool",
        code_fn=lambda a: f"def is_positive({a}: int) -> bool:\n    return {a} > 0",
        eval_fn=lambda a: 1 if a > 0 else 0,
        arg_gen_fn=lambda: [random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="is_zero",
        arity=1,
        ret_type="bool",
        code_fn=lambda a: f"def is_zero({a}: int) -> bool:\n    return {a} == 0",
        eval_fn=lambda a: 1 if a == 0 else 0,
        arg_gen_fn=lambda: [random.choice([0, random.randint(-50, 50)])],
    ))

    specs.append(TemplateSpec(
        intent="is_even",
        arity=1,
        ret_type="bool",
        code_fn=lambda a: f"def is_even({a}: int) -> bool:\n    return {a} % 2 == 0",
        eval_fn=lambda a: 1 if (a % 2 == 0) else 0,
        arg_gen_fn=lambda: [random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="is_odd",
        arity=1,
        ret_type="bool",
        code_fn=lambda a: f"def is_odd({a}: int) -> bool:\n    return {a} % 2 != 0",
        eval_fn=lambda a: 1 if (a % 2 != 0) else 0,
        arg_gen_fn=lambda: [random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="are_equal",
        arity=2,
        ret_type="bool",
        code_fn=lambda a, b: f"def are_equal({a}: int, {b}: int) -> bool:\n    return {a} == {b}",
        eval_fn=lambda a, b: 1 if a == b else 0,
        arg_gen_fn=lambda: [random.randint(1, 20), random.choice([random.randint(1, 20), random.randint(1, 5)])],
    ))

    specs.append(TemplateSpec(
        intent="is_greater",
        arity=2,
        ret_type="bool",
        code_fn=lambda a, b: f"def is_greater({a}: int, {b}: int) -> bool:\n    return {a} > {b}",
        eval_fn=lambda a, b: 1 if a > b else 0,
        arg_gen_fn=lambda: [random.randint(-50, 50), random.randint(-50, 50)],
    ))

    specs.append(TemplateSpec(
        intent="is_less",
        arity=2,
        ret_type="bool",
        code_fn=lambda a, b: f"def is_less({a}: int, {b}: int) -> bool:\n    return {a} < {b}",
        eval_fn=lambda a, b: 1 if a < b else 0,
        arg_gen_fn=lambda: [random.randint(-50, 50), random.randint(-50, 50)],
    ))

    specs.append(TemplateSpec(
        intent="max_two",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def max_two({a}: int, {b}: int) -> int:\n    return {a} if {a} > {b} else {b}",
        eval_fn=lambda a, b: max(a, b),
        arg_gen_fn=lambda: [random.randint(-100, 100), random.randint(-100, 100)],
    ))

    specs.append(TemplateSpec(
        intent="min_two",
        arity=2,
        ret_type="int",
        code_fn=lambda a, b: f"def min_two({a}: int, {b}: int) -> int:\n    return {a} if {a} < {b} else {b}",
        eval_fn=lambda a, b: min(a, b),
        arg_gen_fn=lambda: [random.randint(-100, 100), random.randint(-100, 100)],
    ))

    # =========================================================================
    # 4. Floating Point Operations & Scientific Kernels
    # =========================================================================
    specs.append(TemplateSpec(
        intent="fdiv",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def fdiv({u}: float, {v}: float) -> float:\n    return {u} / {v}",
        eval_fn=lambda u, v: u / v,
        arg_gen_fn=lambda: [
            random.choice([random.randint(-100, 100), round(random.uniform(-50.0, 50.0), 2)]),
            random.choice([x for x in range(-20, 21) if x != 0]),
        ],
        arg_types=["float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="percentage",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def percentage({u}: float, {v}: float) -> float:\n    return ({u} * {v}) / 100.0",
        eval_fn=lambda u, v: (u * v) / 100.0,
        arg_gen_fn=lambda: [round(random.uniform(1.0, 500.0), 2), round(random.uniform(1.0, 100.0), 2)],
        arg_types=["float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="weighted_avg",
        arity=4,
        ret_type="float",
        code_fn=lambda w1, x1, w2, x2: (
            f"def weighted_avg({w1}: float, {x1}: float, {w2}: float, {x2}: float) -> float:\n"
            f"    return ({w1} * {x1} + {w2} * {x2}) / ({w1} + {w2})"
        ),
        eval_fn=lambda w1, x1, w2, x2: (w1 * x1 + w2 * x2) / (w1 + w2),
        arg_gen_fn=lambda: [
            round(random.uniform(1.0, 10.0), 1),
            round(random.uniform(0.0, 100.0), 1),
            round(random.uniform(1.0, 10.0), 1),
            round(random.uniform(0.0, 100.0), 1),
        ],
        arg_types=["float", "float", "float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="lerp",
        arity=3,
        ret_type="float",
        code_fn=lambda a, b, t: f"def lerp({a}: float, {b}: float, {t}: float) -> float:\n    return {a} + {t} * ({b} - {a})",
        eval_fn=lambda a, b, t: a + t * (b - a),
        arg_gen_fn=lambda: [
            round(random.uniform(0.0, 100.0), 1),
            round(random.uniform(0.0, 100.0), 1),
            random.choice([0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.2]),
        ],
        arg_types=["float", "float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="compound_scale",
        arity=3,
        ret_type="float",
        code_fn=lambda a, rate, n: (
            f"def compound_scale({a}: float, {rate}: float, {n}: int) -> float:\n"
            f"    return {a} * (1.0 + {rate}) ** {n}"
        ),
        eval_fn=lambda a, rate, n: a * (1.0 + rate) ** n,
        arg_gen_fn=lambda: [
            round(random.uniform(10.0, 200.0), 1),
            random.choice([0.05, 0.1, 0.15, 0.2, 0.25]),
            random.randint(1, 4),
        ],
        arg_types=["float", "float", "int"],
    ))

    specs.append(TemplateSpec(
        intent="round_val",
        arity=1,
        ret_type="int",
        code_fn=lambda x: f"def round_val({x}: float) -> int:\n    return round({x})",
        eval_fn=lambda x: round(x),
        arg_gen_fn=lambda: [round(random.uniform(-50.0, 50.0), 2)],
        arg_types=["float"],
    ))

    specs.append(TemplateSpec(
        intent="fadd",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def fadd({u}: float, {v}: float) -> float:\n    return {u} + {v}",
        eval_fn=lambda u, v: u + v,
        arg_gen_fn=lambda: [round(random.uniform(-100.0, 100.0), 2), round(random.uniform(-100.0, 100.0), 2)],
        arg_types=["float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="fsub",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def fsub({u}: float, {v}: float) -> float:\n    return {u} - {v}",
        eval_fn=lambda u, v: u - v,
        arg_gen_fn=lambda: [round(random.uniform(-100.0, 100.0), 2), round(random.uniform(-100.0, 100.0), 2)],
        arg_types=["float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="fmul",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def fmul({u}: float, {v}: float) -> float:\n    return {u} * {v}",
        eval_fn=lambda u, v: u * v,
        arg_gen_fn=lambda: [round(random.uniform(-20.0, 20.0), 2), round(random.uniform(-20.0, 20.0), 2)],
        arg_types=["float", "float"],
    ))

    specs.append(TemplateSpec(
        intent="average_two_float",
        arity=2,
        ret_type="float",
        code_fn=lambda u, v: f"def average_two_float({u}: float, {v}: float) -> float:\n    return ({u} + {v}) / 2.0",
        eval_fn=lambda u, v: (u + v) / 2.0,
        arg_gen_fn=lambda: [round(random.uniform(-100.0, 100.0), 2), round(random.uniform(-100.0, 100.0), 2)],
        arg_types=["float", "float"],
    ))

    return specs


def select_variable_names(arity: int) -> Tuple[str, ...]:
    """Selects a balanced tuple of orthogonal parameter names based on arity."""
    if arity == 1:
        return (random.choice(SINGLE_VARS),)
    elif arity == 2:
        return random.choice(BINARY_VARS)
    elif arity == 3:
        return random.choice(TRIPLE_VARS)
    elif arity == 4:
        return random.choice(QUAD_VARS)
    else:
        return tuple(f"arg{i}" for i in range(arity))


def generate_dataset(
    num_samples: int = 10_000,
    output_path: str = "data/bytecode_train_10k.jsonl",
    seed: int = 42,
) -> None:
    """
    Generates synthetic bytecode dataset and writes to JSONL.
    Each sample is verified against native Python execution.
    """
    random.seed(seed)
    compiler = ASTCompiler()
    vm = MiniStackVM()
    templates = create_template_registry()

    resolved_out = Path(output_path)
    if not resolved_out.is_absolute():
        resolved_out = REPO_ROOT / output_path
    resolved_out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"GENERATING SYNTHETIC BYTECODE DATASET: {num_samples} SAMPLES")
    print("=" * 80)
    print(f"Output Target   : {resolved_out}")
    print(f"Template Count  : {len(templates)} unique algorithmic intents")
    print(f"Random Seed     : {seed}\n")

    compiled_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, bytes]] = {}
    records: List[Dict[str, Any]] = []

    for i in range(num_samples):
        spec = random.choice(templates)
        var_names = select_variable_names(spec.arity)
        cache_key = (spec.intent, var_names)

        if cache_key in compiled_cache:
            mnemonics_str, assembled_bc = compiled_cache[cache_key]
        else:
            code_str = spec.code_fn(*var_names)
            raw_bc = compiler.compile(code_str)
            mnemonics_str = bytecode_to_mnemonics(raw_bc)
            assembled_bc = vm.assemble_mnemonics(mnemonics_str)
            compiled_cache[cache_key] = (mnemonics_str, assembled_bc)

        # Generate 4 diverse test cases
        test_cases: List[Tuple[List[Any], Any]] = []
        for _ in range(4):
            args = spec.arg_gen_fn()
            expected_raw = spec.eval_fn(*args)
            expected = int(expected_raw) if isinstance(expected_raw, bool) else expected_raw

            # Immediate verification on MiniStackVM
            vm_res = vm.execute(assembled_bc, args)
            if isinstance(expected, float) or isinstance(vm_res, float):
                if not math.isclose(float(vm_res), float(expected), rel_tol=1e-4, abs_tol=1e-4):
                    raise RuntimeError(
                        f"Verification parity mismatch for '{spec.intent}' with args {args}: "
                        f"VM returned {vm_res}, expected {expected}"
                    )
            else:
                if vm_res != expected:
                    raise RuntimeError(
                        f"Verification parity mismatch for '{spec.intent}' with args {args}: "
                        f"VM returned {vm_res}, expected {expected}"
                    )

            test_cases.append((args, expected))

        # Build prompt signature
        param_str = ", ".join(f"{v}: {t}" for v, t in zip(var_names, spec.arg_types))
        signature = f"def {spec.intent}({param_str}) -> {spec.ret_type}:"

        record = {
            "prompt": f"CONTRACT: {signature} -> BYTECODE:\n",
            "completion": f"{mnemonics_str}\n",
            "intent": spec.intent,
            "test_cases": test_cases,
        }
        records.append(record)

        if (i + 1) % 2000 == 0 or (i + 1) == num_samples:
            print(f"  Processed {i + 1:>5}/{num_samples} samples (100% VM verified)...")

    # Write output to JSONL file
    with open(resolved_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[SUCCESS] Successfully generated and verified {len(records)} samples.")
    print(f"Saved dataset to: {resolved_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MiniStackVM Synthetic Bytecode Training Dataset")
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Number of synthetic samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/bytecode_train_10k.jsonl",
        help="Target output JSONL path (default: data/bytecode_train_10k.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    generate_dataset(num_samples=args.samples, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
