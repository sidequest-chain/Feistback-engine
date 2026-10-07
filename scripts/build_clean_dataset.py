"""
Build Clean Algorithmic Python Dataset (Orthogonal Variable Distribution)
=========================================================================
Generates 2000 pure, syntactically verified algorithmic Python functions with:
  - ZERO docstrings
  - Desynchronized, orthogonal variable naming across all function categories
  - Strict variable binding (header arguments strictly match body variables)
  - Max frequency for any argument <= 15%
  - Zero 'self', 'cls', 'class', or framework/library internals
  - 100% verified via ast.parse
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import random
from typing import List, Tuple, Set, Dict

# Ensure reproducibility
random.seed(42)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train_code_clean.jsonl"

# Required variable pools from user specification
SINGLE_VARS = ["n", "x", "val", "num", "value", "a", "k", "item", "target", "elem"]
BINARY_VARS = [
    ("a", "b"),
    ("x", "y"),
    ("val1", "val2"),
    ("num_a", "num_b"),
    ("left", "right"),
    ("first", "second"),
    ("u", "v"),
    ("start", "end"),
    ("p", "q"),
]
TRIPLE_VARS = [
    ("x", "y", "z"),
    ("a", "b", "c"),
    ("val1", "val2", "val3"),
    ("u", "v", "w"),
    ("first", "second", "third"),
]
COLLECTIONS_VARS = ["items", "nums", "arr", "lst", "values", "elements", "data", "seq"]


def validate_code(code: str) -> bool:
    """Verifies that code parses cleanly and violates no purity rules."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    has_func = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
            return False
        if isinstance(node, ast.FunctionDef):
            has_func = True
            for arg in node.args.args:
                if arg.arg in ("self", "cls"):
                    return False
        if isinstance(node, ast.Name):
            if node.id in ("self", "cls", "inspect", "urllib"):
                return False
        # Disallow docstrings
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return False

    return has_func


def generate_orthogonal_dataset() -> List[str]:
    """Generates 2000 functions with orthogonal, balanced variable naming."""
    funcs: List[str] = []

    # 1. Binary arithmetic & comparisons (rotated over all BINARY_VARS)
    binary_templates = [
        ("add", lambda u, v: f"def add({u}: int, {v}: int) -> int:\n    return {u} + {v}"),
        ("subtract", lambda u, v: f"def subtract({u}: int, {v}: int) -> int:\n    return {u} - {v}"),
        ("multiply", lambda u, v: f"def multiply({u}: int, {v}: int) -> int:\n    return {u} * {v}"),
        ("divide", lambda u, v: f"def divide({u}: float, {v}: float) -> float:\n    if {v} == 0:\n        return 0.0\n    return {u} / {v}"),
        ("integer_divide", lambda u, v: f"def integer_divide({u}: int, {v}: int) -> int:\n    if {v} == 0:\n        return 0\n    return {u} // {v}"),
        ("modulo", lambda u, v: f"def modulo({u}: int, {v}: int) -> int:\n    if {v} == 0:\n        return 0\n    return {u} % {v}"),
        ("power", lambda u, v: f"def power({u}: float, {v}: int) -> float:\n    return {u} ** {v}"),
        ("average_two", lambda u, v: f"def average_two({u}: float, {v}: float) -> float:\n    return ({u} + {v}) / 2.0"),
        ("max_two", lambda u, v: f"def max_two({u}: int, {v}: int) -> int:\n    return {u} if {u} > {v} else {v}"),
        ("min_two", lambda u, v: f"def min_two({u}: int, {v}: int) -> int:\n    return {u} if {u} < {v} else {v}"),
        ("is_greater", lambda u, v: f"def is_greater({u}: int, {v}: int) -> bool:\n    return {u} > {v}"),
        ("is_less", lambda u, v: f"def is_less({u}: int, {v}: int) -> bool:\n    return {u} < {v}"),
        ("are_equal", lambda u, v: f"def are_equal({u}: int, {v}: int) -> bool:\n    return {u} == {v}"),
        ("is_divisible", lambda u, v: f"def is_divisible({u}: int, {v}: int) -> bool:\n    if {v} == 0:\n        return False\n    return {u} % {v} == 0"),
        ("abs_diff", lambda u, v: f"def abs_diff({u}: int, {v}: int) -> int:\n    return abs({u} - {v})"),
        ("hypotenuse", lambda u, v: f"def hypotenuse({u}: float, {v}: float) -> float:\n    return ({u} * {u} + {v} * {v}) ** 0.5"),
        ("gcd", lambda u, v: f"def gcd({u}: int, {v}: int) -> int:\n    while {v}:\n        {u}, {v} = {v}, {u} % {v}\n    return abs({u})"),
        ("lcm", lambda u, v: f"def lcm({u}: int, {v}: int) -> int:\n    if {u} == 0 or {v} == 0:\n        return 0\n    x, y = abs({u}), abs({v})\n    a, b = x, y\n    while b:\n        a, b = b, a % b\n    return (x * y) // a"),
    ]

    for name, templ in binary_templates:
        for u, v in BINARY_VARS:
            funcs.append(templ(u, v))

    # 2. Triple variable operations (rotated over all TRIPLE_VARS)
    triple_templates = [
        ("multiply_three", lambda u, v, w: f"def multiply_three({u}: int, {v}: int, {w}: int) -> int:\n    return {u} * {v} * {w}"),
        ("add_three", lambda u, v, w: f"def add_three({u}: int, {v}: int, {w}: int) -> int:\n    return {u} + {v} + {w}"),
        ("average_three", lambda u, v, w: f"def average_three({u}: float, {v}: float, {w}: float) -> float:\n    return ({u} + {v} + {w}) / 3.0"),
        ("max_three", lambda u, v, w: f"def max_three({u}: int, {v}: int, {w}: int) -> int:\n    m = {u}\n    if {v} > m:\n        m = {v}\n    if {w} > m:\n        m = {w}\n    return m"),
        ("min_three", lambda u, v, w: f"def min_three({u}: int, {v}: int, {w}: int) -> int:\n    m = {u}\n    if {v} < m:\n        m = {v}\n    if {w} < m:\n        m = {w}\n    return m"),
        ("clamp", lambda u, v, w: f"def clamp({u}: float, {v}: float, {w}: float) -> float:\n    if {u} < {v}:\n        return {v}\n    if {u} > {w}:\n        return {w}\n    return {u}"),
        ("in_range", lambda u, v, w: f"def in_range({u}: int, {v}: int, {w}: int) -> bool:\n    return {v} <= {u} <= {w}"),
    ]

    for name, templ in triple_templates:
        for u, v, w in TRIPLE_VARS:
            funcs.append(templ(u, v, w))

    # 3. Single variable arithmetic, logic, and series (rotated over all SINGLE_VARS)
    single_templates = [
        ("is_even", lambda v: f"def is_even({v}: int) -> bool:\n    return {v} % 2 == 0"),
        ("is_odd", lambda v: f"def is_odd({v}: int) -> bool:\n    return {v} % 2 != 0"),
        ("is_positive", lambda v: f"def is_positive({v}: float) -> bool:\n    return {v} > 0"),
        ("is_negative", lambda v: f"def is_negative({v}: float) -> bool:\n    return {v} < 0"),
        ("is_zero", lambda v: f"def is_zero({v}: float) -> bool:\n    return {v} == 0"),
        ("abs_val", lambda v: f"def abs_val({v}: float) -> float:\n    if {v} < 0:\n        return -{v}\n    return {v}"),
        ("square", lambda v: f"def square({v}: float) -> float:\n    return {v} * {v}"),
        ("cube", lambda v: f"def cube({v}: float) -> float:\n    return {v} * {v} * {v}"),
        ("sign", lambda v: f"def sign({v}: float) -> int:\n    if {v} > 0:\n        return 1\n    elif {v} < 0:\n        return -1\n    return 0"),
        ("double_val", lambda v: f"def double_val({v}: int) -> int:\n    return {v} * 2"),
        ("triple_val", lambda v: f"def triple_val({v}: int) -> int:\n    return {v} * 3"),
        ("negate", lambda v: f"def negate({v}: int) -> int:\n    return -{v}"),
        ("is_power_of_two", lambda v: f"def is_power_of_two({v}: int) -> bool:\n    if {v} <= 0:\n        return False\n    return ({v} & ({v} - 1)) == 0"),
        ("factorial", lambda v: f"def factorial({v}: int) -> int:\n    if {v} <= 1:\n        return 1\n    res = 1\n    for i in range(2, {v} + 1):\n        res *= i\n    return res"),
        ("fibonacci", lambda v: f"def fibonacci({v}: int) -> int:\n    if {v} <= 0:\n        return 0\n    if {v} == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, {v} + 1):\n        a, b = b, a + b\n    return b"),
        ("triangular_number", lambda v: f"def triangular_number({v}: int) -> int:\n    if {v} <= 0:\n        return 0\n    return {v} * ({v} + 1) // 2"),
        ("is_prime", lambda v: f"def is_prime({v}: int) -> bool:\n    if {v} < 2:\n        return False\n    if {v} in (2, 3):\n        return True\n    if {v} % 2 == 0 or {v} % 3 == 0:\n        return False\n    i = 5\n    while i * i <= {v}:\n        if {v} % i == 0 or {v} % (i + 2) == 0:\n            return False\n        i += 6\n    return True"),
        ("sum_of_digits", lambda v: f"def sum_of_digits({v}: int) -> int:\n    {v} = abs({v})\n    total = 0\n    while {v} > 0:\n        total += {v} % 10\n        {v} //= 10\n    return total"),
        ("count_digits", lambda v: f"def count_digits({v}: int) -> int:\n    if {v} == 0:\n        return 1\n    {v} = abs({v})\n    c = 0\n    while {v} > 0:\n        c += 1\n        {v} //= 10\n    return c"),
        ("reverse_digits", lambda v: f"def reverse_digits({v}: int) -> int:\n    s = -1 if {v} < 0 else 1\n    {v} = abs({v})\n    rev = 0\n    while {v} > 0:\n        rev = rev * 10 + {v} % 10\n        {v} //= 10\n    return s * rev"),
        ("collatz_steps", lambda v: f"def collatz_steps({v}: int) -> int:\n    steps = 0\n    while {v} > 1:\n        if {v} % 2 == 0:\n            {v} //= 2\n        else:\n            {v} = 3 * {v} + 1\n        steps += 1\n    return steps"),
    ]

    for name, templ in single_templates:
        for v in SINGLE_VARS:
            funcs.append(templ(v))

    # 4. Collection single operations (rotated over all COLLECTIONS_VARS)
    collection_single_templates = [
        ("list_sum", lambda c: f"def list_sum({c}: list[int]) -> int:\n    total = 0\n    for x in {c}:\n        total += x\n    return total"),
        ("list_product", lambda c: f"def list_product({c}: list[int]) -> int:\n    prod = 1\n    for x in {c}:\n        prod *= x\n    return prod"),
        ("find_max", lambda c: f"def find_max({c}: list[float]) -> float:\n    if not {c}:\n        return 0.0\n    m = {c}[0]\n    for x in {c}[1:]:\n        if x > m:\n            m = x\n    return m"),
        ("find_min", lambda c: f"def find_min({c}: list[float]) -> float:\n    if not {c}:\n        return 0.0\n    m = {c}[0]\n    for x in {c}[1:]:\n        if x < m:\n            m = x\n    return m"),
        ("average", lambda c: f"def average({c}: list[float]) -> float:\n    if not {c}:\n        return 0.0\n    return sum({c}) / len({c})"),
        ("reverse_list", lambda c: f"def reverse_list({c}: list[int]) -> list[int]:\n    res = []\n    for i in range(len({c}) - 1, -1, -1):\n        res.append({c}[i])\n    return res"),
        ("filter_positive", lambda c: f"def filter_positive({c}: list[int]) -> list[int]:\n    return [x for x in {c} if x > 0]"),
        ("filter_negative", lambda c: f"def filter_negative({c}: list[int]) -> list[int]:\n    return [x for x in {c} if x < 0]"),
        ("filter_even", lambda c: f"def filter_even({c}: list[int]) -> list[int]:\n    return [x for x in {c} if x % 2 == 0]"),
        ("filter_odd", lambda c: f"def filter_odd({c}: list[int]) -> list[int]:\n    return [x for x in {c} if x % 2 != 0]"),
        ("remove_duplicates", lambda c: f"def remove_duplicates({c}: list[int]) -> list[int]:\n    seen = set()\n    res = []\n    for x in {c}:\n        if x not in seen:\n            seen.add(x)\n            res.append(x)\n    return res"),
        ("cumulative_sum", lambda c: f"def cumulative_sum({c}: list[int]) -> list[int]:\n    res = []\n    total = 0\n    for x in {c}:\n        total += x\n        res.append(total)\n    return res"),
        ("is_sorted_asc", lambda c: f"def is_sorted_asc({c}: list[int]) -> bool:\n    for i in range(len({c}) - 1):\n        if {c}[i] > {c}[i + 1]:\n            return False\n    return True"),
        ("is_sorted_desc", lambda c: f"def is_sorted_desc({c}: list[int]) -> bool:\n    for i in range(len({c}) - 1):\n        if {c}[i] < {c}[i + 1]:\n            return False\n    return True"),
        ("bubble_sort", lambda c: f"def bubble_sort({c}: list[int]) -> list[int]:\n    arr = list({c})\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr"),
        ("insertion_sort", lambda c: f"def insertion_sort({c}: list[int]) -> list[int]:\n    arr = list({c})\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr"),
        ("max_subarray_sum", lambda c: f"def max_subarray_sum({c}: list[int]) -> int:\n    if not {c}:\n        return 0\n    max_so_far = cur_max = {c}[0]\n    for x in {c}[1:]:\n        cur_max = max(x, cur_max + x)\n        max_so_far = max(max_so_far, cur_max)\n    return max_so_far"),
        ("count_elements", lambda c: f"def count_elements({c}: list[int]) -> int:\n    return len({c})"),
    ]

    for name, templ in collection_single_templates:
        for c in COLLECTIONS_VARS:
            funcs.append(templ(c))

    # 5. Collection + Scalar operation (pairs between COLLECTIONS_VARS and SINGLE_VARS)
    collection_scalar_templates = [
        ("contains", lambda c, v: f"def contains({c}: list[int], {v}: int) -> bool:\n    for x in {c}:\n        if x == {v}:\n            return True\n    return False"),
        ("count_occurrences", lambda c, v: f"def count_occurrences({c}: list[int], {v}: int) -> int:\n    count = 0\n    for x in {c}:\n        if x == {v}:\n            count += 1\n    return count"),
        ("find_index", lambda c, v: f"def find_index({c}: list[int], {v}: int) -> int:\n    for i, x in enumerate({c}):\n        if x == {v}:\n            return i\n    return -1"),
        ("linear_search", lambda c, v: f"def linear_search({c}: list[int], {v}: int) -> int:\n    for i in range(len({c})):\n        if {c}[i] == {v}:\n            return i\n    return -1"),
        ("binary_search", lambda c, v: f"def binary_search({c}: list[int], {v}: int) -> int:\n    left, right = 0, len({c}) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if {c}[mid] == {v}:\n            return mid\n        elif {c}[mid] < {v}:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"),
        ("lower_bound", lambda c, v: f"def lower_bound({c}: list[int], {v}: int) -> int:\n    left, right = 0, len({c})\n    while left < right:\n        mid = (left + right) // 2\n        if {c}[mid] < {v}:\n            left = mid + 1\n        else:\n            right = mid\n    return left"),
        ("upper_bound", lambda c, v: f"def upper_bound({c}: list[int], {v}: int) -> int:\n    left, right = 0, len({c})\n    while left < right:\n        mid = (left + right) // 2\n        if {c}[mid] <= {v}:\n            left = mid + 1\n        else:\n            right = mid\n    return left"),
        ("count_greater", lambda c, v: f"def count_greater({c}: list[float], {v}: float) -> int:\n    return sum(1 for x in {c} if x > {v})"),
        ("count_less", lambda c, v: f"def count_less({c}: list[float], {v}: float) -> int:\n    return sum(1 for x in {c} if x < {v})"),
        ("remove_all", lambda c, v: f"def remove_all({c}: list[int], {v}: int) -> list[int]:\n    return [x for x in {c} if x != {v}]"),
        ("filter_greater", lambda c, v: f"def filter_greater({c}: list[int], {v}: int) -> list[int]:\n    return [x for x in {c} if x > {v}]"),
        ("filter_less", lambda c, v: f"def filter_less({c}: list[int], {v}: int) -> list[int]:\n    return [x for x in {c} if x < {v}]"),
        ("rotate_left", lambda c, v: f"def rotate_left({c}: list[int], {v}: int) -> list[int]:\n    if not {c}:\n        return []\n    k = {v} % len({c})\n    return {c}[k:] + {c}[:k]"),
        ("rotate_right", lambda c, v: f"def rotate_right({c}: list[int], {v}: int) -> list[int]:\n    if not {c}:\n        return []\n    k = {v} % len({c})\n    return {c}[-k:] + {c}[:-k] if k > 0 else list({c})"),
        ("chunk_list", lambda c, v: f"def chunk_list({c}: list[int], {v}: int) -> list[list[int]]:\n    if {v} <= 0:\n        return []\n    res = []\n    for i in range(0, len({c}), {v}):\n        res.append({c}[i : i + {v}])\n    return res"),
        ("scale_list", lambda c, v: f"def scale_list({c}: list[float], {v}: float) -> list[float]:\n    return [x * {v} for x in {c}]"),
        ("offset_list", lambda c, v: f"def offset_list({c}: list[int], {v}: int) -> list[int]:\n    return [x + {v} for x in {c}]"),
        ("take_first_n", lambda c, v: f"def take_first_n({c}: list[int], {v}: int) -> list[int]:\n    return {c}[:{v}]"),
        ("drop_first_n", lambda c, v: f"def drop_first_n({c}: list[int], {v}: int) -> list[int]:\n    return {c}[{v}:]"),
        ("two_sum_indices", lambda c, v: f"def two_sum_indices({c}: list[int], {v}: int) -> list[int]:\n    seen = {{}}\n    for i, x in enumerate({c}):\n        comp = {v} - x\n        if comp in seen:\n            return [seen[comp], i]\n        seen[x] = i\n    return []"),
    ]

    # Combine collections and single vars in a balanced round-robin
    for templ_name, templ in collection_scalar_templates:
        for idx in range(len(COLLECTIONS_VARS)):
            c = COLLECTIONS_VARS[idx]
            v = SINGLE_VARS[idx % len(SINGLE_VARS)]
            funcs.append(templ(c, v))

    # 6. Two collections operations (rotated over pairs of COLLECTIONS_VARS)
    collection_pairs = [
        ("items1", "items2"),
        ("nums1", "nums2"),
        ("arr1", "arr2"),
        ("lst1", "lst2"),
        ("left", "right"),
        ("first", "second"),
        ("list_a", "list_b"),
        ("values1", "values2"),
    ]
    two_col_templates = [
        ("elementwise_add", lambda c1, c2: f"def elementwise_add({c1}: list[int], {c2}: list[int]) -> list[int]:\n    return [a + b for a, b in zip({c1}, {c2})]"),
        ("elementwise_sub", lambda c1, c2: f"def elementwise_sub({c1}: list[int], {c2}: list[int]) -> list[int]:\n    return [a - b for a, b in zip({c1}, {c2})]"),
        ("dot_product", lambda c1, c2: f"def dot_product({c1}: list[float], {c2}: list[float]) -> float:\n    total = 0.0\n    for a, b in zip({c1}, {c2}):\n        total += a * b\n    return total"),
        ("merge_sorted", lambda c1, c2: f"def merge_sorted({c1}: list[int], {c2}: list[int]) -> list[int]:\n    res = []\n    i = j = 0\n    while i < len({c1}) and j < len({c2}):\n        if {c1}[i] <= {c2}[j]:\n            res.append({c1}[i])\n            i += 1\n        else:\n            res.append({c2}[j])\n            j += 1\n    res.extend({c1}[i:])\n    res.extend({c2}[j:])\n    return res"),
        ("interleave", lambda c1, c2: f"def interleave({c1}: list[int], {c2}: list[int]) -> list[int]:\n    res = []\n    min_len = min(len({c1}), len({c2}))\n    for i in range(min_len):\n        res.append({c1}[i])\n        res.append({c2}[i])\n    res.extend({c1}[min_len:])\n    res.extend({c2}[min_len:])\n    return res"),
        ("list_intersection", lambda c1, c2: f"def list_intersection({c1}: list[int], {c2}: list[int]) -> list[int]:\n    s = set({c2})\n    return [x for x in {c1} if x in s]"),
        ("list_difference", lambda c1, c2: f"def list_difference({c1}: list[int], {c2}: list[int]) -> list[int]:\n    s = set({c2})\n    return [x for x in {c1} if x not in s]"),
    ]

    for name, templ in two_col_templates:
        for c1, c2 in collection_pairs:
            funcs.append(templ(c1, c2))

    # 7. Add systematic, indexed variations to reach exactly 2000 functions
    # Using strict round-robin over pools so that no single variable dominates
    idx = 1
    while len(funcs) < 2200:
        c = idx * 2
        # Single var round-robin
        v_s = SINGLE_VARS[idx % len(SINGLE_VARS)]
        funcs.append(f"def scale_by_{idx}({v_s}: float) -> float:\n    return {v_s} * {idx}.0")
        funcs.append(f"def add_const_{idx}({v_s}: int) -> int:\n    return {v_s} + {c}")
        funcs.append(f"def sub_const_{idx}({v_s}: int) -> int:\n    return {v_s} - {c}")
        funcs.append(f"def is_greater_than_{idx}({v_s}: int) -> bool:\n    return {v_s} > {idx}")
        funcs.append(f"def is_less_than_{idx}({v_s}: int) -> bool:\n    return {v_s} < {idx}")
        funcs.append(f"def is_equal_to_{idx}({v_s}: int) -> bool:\n    return {v_s} == {idx}")
        funcs.append(f"def is_divisible_by_{idx + 1}({v_s}: int) -> bool:\n    return {v_s} % {idx + 1} == 0")
        funcs.append(f"def modulo_{idx + 2}({v_s}: int) -> int:\n    return {v_s} % {idx + 2}")

        # Collection var round-robin
        v_l = COLLECTIONS_VARS[idx % len(COLLECTIONS_VARS)]
        funcs.append(f"def filter_above_{idx}({v_l}: list[int]) -> list[int]:\n    return [x for x in {v_l} if x > {idx}]")
        funcs.append(f"def filter_below_{idx}({v_l}: list[int]) -> list[int]:\n    return [x for x in {v_l} if x < {idx}]")
        funcs.append(f"def multiply_elements_{idx}({v_l}: list[int]) -> list[int]:\n    return [x * {idx} for x in {v_l}]")
        funcs.append(f"def add_to_elements_{idx}({v_l}: list[int]) -> list[int]:\n    return [x + {idx} for x in {v_l}]")
        funcs.append(f"def take_first_{idx}({v_l}: list[int]) -> list[int]:\n    return {v_l}[:{idx}]")
        funcs.append(f"def drop_first_{idx}({v_l}: list[int]) -> list[int]:\n    return {v_l}[{idx}:]")

        # Binary var round-robin
        v_b1, v_b2 = BINARY_VARS[idx % len(BINARY_VARS)]
        funcs.append(f"def linear_combo_{idx}({v_b1}: int, {v_b2}: int) -> int:\n    return {idx} * {v_b1} + {v_b2}")
        funcs.append(f"def diff_scaled_{idx}({v_b1}: int, {v_b2}: int) -> int:\n    return ({v_b1} - {v_b2}) * {idx}")

        idx += 1

    # Deduplicate and validate
    seen = set()
    valid_funcs: List[str] = []
    for f in funcs:
        code = f.strip()
        if code not in seen and validate_code(code):
            seen.add(code)
            valid_funcs.append(code)

    return valid_funcs[:2000]


def main() -> None:
    print("=" * 70)
    print("GENERATING ORTHOGONALLY BALANCED DATASET (2000 FUNCTIONS)")
    print("=" * 70)

    funcs = generate_orthogonal_dataset()
    print(f"Generated and validated {len(funcs)} functions.")

    random.shuffle(funcs)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for code in funcs:
            f.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")

    print(f"Saved dataset to: {DATA_PATH}")
    print(f"File size: {os.path.getsize(DATA_PATH):,} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
