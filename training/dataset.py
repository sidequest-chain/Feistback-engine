"""
Training Dataset & Code Generator
=================================
Generates curated, diverse Python code examples and provides tokenized datasets
for representation distillation of PureFeistelLM against teacher models.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


DEFAULT_CODE_EXAMPLES = [
    # Math & Arithmetic
    'def add(a: int, b: int) -> int:\n    """Return the sum of two integers."""\n    return a + b\n',
    'def subtract(a: int, b: int) -> int:\n    """Return the difference between two integers."""\n    return a - b\n',
    'def multiply(a: int, b: int) -> int:\n    """Return the product of two integers."""\n    return a * b\n',
    'def divide(a: float, b: float) -> float:\n    """Safely divide a by b."""\n    if b == 0:\n        raise ZeroDivisionError("Cannot divide by zero")\n    return a / b\n',
    'def power(base: float, exponent: int) -> float:\n    """Compute base raised to exponent."""\n    return base ** exponent\n',
    'def factorial(n: int) -> int:\n    """Calculate factorial of non-negative integer."""\n    if n < 0:\n        raise ValueError("Factorial undefined for negative numbers")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n',
    'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n',
    'def gcd(a: int, b: int) -> int:\n    """Greatest common divisor using Euclidean algorithm."""\n    while b != 0:\n        a, b = b, a % b\n    return abs(a)\n',
    'def lcm(a: int, b: int) -> int:\n    """Least common multiple of two integers."""\n    if a == 0 or b == 0:\n        return 0\n    def _gcd(x, y):\n        while y: x, y = y, x % y\n        return abs(x)\n    return abs(a * b) // _gcd(a, b)\n',
    'def is_prime(n: int) -> bool:\n    """Determine if an integer is prime."""\n    if n < 2:\n        return False\n    if n in (2, 3):\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n',
    'def prime_factors(n: int) -> list[int]:\n    """Return prime factorization of integer n."""\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d)\n            n //= d\n        d += 1\n    if n > 1:\n        factors.append(n)\n    return factors\n',
    'def sum_digits(n: int) -> int:\n    """Calculate sum of digits of an integer."""\n    return sum(int(d) for d in str(abs(n)))\n',
    'def clamp(val: float, min_val: float, max_val: float) -> float:\n    """Constrain value to lie within [min_val, max_val]."""\n    return max(min_val, min(val, max_val))\n',
    'def lerp(start: float, end: float, t: float) -> float:\n    """Linear interpolation between start and end by parameter t."""\n    return start + t * (end - start)\n',
    'def is_perfect_square(n: int) -> bool:\n    """Check if number is a perfect square."""\n    if n < 0:\n        return False\n    root = int(n ** 0.5)\n    return root * root == n\n',
    'def count_set_bits(n: int) -> int:\n    """Count number of 1-bits in binary representation."""\n    count = 0\n    while n > 0:\n        count += n & 1\n        n >>= 1\n    return count\n',
    'def int_to_binary_str(n: int) -> str:\n    """Convert positive integer to binary string."""\n    if n == 0:\n        return "0"\n    bits = []\n    while n > 0:\n        bits.append(str(n % 2))\n        n //= 2\n    return "".join(reversed(bits))\n',
    'def is_even(n: int) -> bool:\n    """Check if integer is even."""\n    return n % 2 == 0\n',
    'def is_odd(n: int) -> bool:\n    """Check if integer is odd."""\n    return n % 2 != 0\n',
    'def abs_val(x: float) -> float:\n    """Compute absolute value without builtins."""\n    return -x if x < 0 else x\n',

    # String Processing
    'def reverse_string(s: str) -> str:\n    """Return the reversed string."""\n    return s[::-1]\n',
    'def is_palindrome(s: str) -> bool:\n    """Check if string is a palindrome ignoring case and spaces."""\n    clean = "".join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n',
    'def count_vowels(s: str) -> int:\n    """Count English vowels in string."""\n    vowels = set("aeiouAEIOU")\n    return sum(1 for c in s if c in vowels)\n',
    'def count_consonants(s: str) -> int:\n    """Count consonants in English string."""\n    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")\n    return sum(1 for c in s if c in consonants)\n',
    'def capitalize_words(s: str) -> str:\n    """Capitalize the first letter of each word."""\n    return " ".join(word.capitalize() for word in s.split())\n',
    'def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:\n    """Truncate string to max_length with suffix."""\n    if len(s) <= max_length:\n        return s\n    return s[:max_length - len(suffix)] + suffix\n',
    'def to_snake_case(s: str) -> str:\n    """Convert CamelCase or mixed string to snake_case."""\n    res = []\n    for i, c in enumerate(s):\n        if c.isupper() and i > 0 and s[i - 1] != "_":\n            res.append("_")\n        res.append(c.lower())\n    return "".join(res)\n',
    'def to_camel_case(s: str) -> str:\n    """Convert snake_case string to CamelCase."""\n    parts = s.split("_")\n    return "".join(p.capitalize() for p in parts if p)\n',
    'def is_anagram(s1: str, s2: str) -> bool:\n    """Check if two strings are anagrams."""\n    c1 = sorted(s1.lower().replace(" ", ""))\n    c2 = sorted(s2.lower().replace(" ", ""))\n    return c1 == c2\n',
    'def run_length_encode(text: str) -> str:\n    """Perform simple run-length encoding on string."""\n    if not text:\n        return ""\n    encoded = []\n    count = 1\n    for i in range(1, len(text)):\n        if text[i] == text[i - 1]:\n            count += 1\n        else:\n            encoded.append(f"{text[i - 1]}{count}")\n            count = 1\n    encoded.append(f"{text[-1]}{count}")\n    return "".join(encoded)\n',
    'def caesar_cipher(text: str, shift: int) -> str:\n    """Encrypt text with Caesar cipher shifted by key."""\n    result = []\n    for char in text:\n        if char.isalpha():\n            base = ord("A") if char.isupper() else ord("a")\n            shifted = chr((ord(char) - base + shift) % 26 + base)\n            result.append(shifted)\n        else:\n            result.append(char)\n    return "".join(result)\n',
    'def word_frequency(text: str) -> dict[str, int]:\n    """Count frequency of each word in text."""\n    words = text.lower().split()\n    freq: dict[str, int] = {}\n    for w in words:\n        cleaned = "".join(c for c in w if c.isalnum())\n        if cleaned:\n            freq[cleaned] = freq.get(cleaned, 0) + 1\n    return freq\n',
    'def longest_common_prefix(strs: list[str]) -> str:\n    """Find longest common prefix across list of strings."""\n    if not strs:\n        return ""\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ""\n    return prefix\n',
    'def rot13(text: str) -> str:\n    """Apply ROT13 cipher to string."""\n    res = []\n    for c in text:\n        if "a" <= c <= "z":\n            res.append(chr((ord(c) - ord("a") + 13) % 26 + ord("a")))\n        elif "A" <= c <= "Z":\n            res.append(chr((ord(c) - ord("A") + 13) % 26 + ord("A")))\n        else:\n            res.append(c)\n    return "".join(res)\n',
    'def is_valid_parentheses(s: str) -> bool:\n    """Check if bracket sequences (), [], {} are balanced."""\n    stack = []\n    mapping = {")": "(", "]": "[", "}": "{"}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n    return len(stack) == 0\n',

    # List & Array Algorithms
    'def find_maximum(numbers: list[int]) -> int:\n    """Return maximum element in non-empty list."""\n    if not numbers:\n        raise ValueError("List is empty")\n    m = numbers[0]\n    for x in numbers[1:]:\n        if x > m:\n            m = x\n    return m\n',
    'def find_minimum(numbers: list[int]) -> int:\n    """Return minimum element in non-empty list."""\n    if not numbers:\n        raise ValueError("List is empty")\n    m = numbers[0]\n    for x in numbers[1:]:\n        if x < m:\n            m = x\n    return m\n',
    'def compute_average(values: list[float]) -> float:\n    """Compute mean of float values."""\n    if not values:\n        return 0.0\n    return sum(values) / len(values)\n',
    'def square_elements(nums: list[int]) -> list[int]:\n    """Return list of squared numbers."""\n    return [x ** 2 for x in nums]\n',
    'def filter_positive(nums: list[int]) -> list[int]:\n    """Filter list to keep only positive numbers."""\n    return [x for x in nums if x > 0]\n',
    'def filter_even(nums: list[int]) -> list[int]:\n    """Filter list to keep only even numbers."""\n    return [x for x in nums if x % 2 == 0]\n',
    'def remove_duplicates(lst: list) -> list:\n    """Remove duplicates while preserving order."""\n    seen = set()\n    res = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            res.append(item)\n    return res\n',
    'def flatten_list(nested: list) -> list:\n    """Flatten one level of nested lists."""\n    res = []\n    for elem in nested:\n        if isinstance(elem, list):\n            res.extend(elem)\n        else:\n            res.append(elem)\n    return res\n',
    'def chunk_list(lst: list, chunk_size: int) -> list[list]:\n    """Split list into consecutive chunks of chunk_size."""\n    if chunk_size <= 0:\n        raise ValueError("Chunk size must be positive")\n    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]\n',
    'def rotate_array(nums: list, k: int) -> list:\n    """Rotate array to the right by k positions."""\n    if not nums:\n        return []\n    k = k % len(nums)\n    return nums[-k:] + nums[:-k]\n',
    'def prefix_sums(nums: list[int]) -> list[int]:\n    """Compute cumulative prefix sum array."""\n    res = []\n    current = 0\n    for x in nums:\n        current += x\n        res.append(current)\n    return res\n',
    'def binary_search(arr: list[int], target: int) -> int:\n    """Binary search on sorted array; returns index or -1."""\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n',
    'def bubble_sort(arr: list[int]) -> list[int]:\n    """Sort array using bubble sort algorithm."""\n    res = list(arr)\n    n = len(res)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if res[j] > res[j + 1]:\n                res[j], res[j + 1] = res[j + 1], res[j]\n                swapped = True\n        if not swapped:\n            break\n    return res\n',
    'def insertion_sort(arr: list[int]) -> list[int]:\n    """Sort array using insertion sort algorithm."""\n    res = list(arr)\n    for i in range(1, len(res)):\n        key = res[i]\n        j = i - 1\n        while j >= 0 and res[j] > key:\n            res[j + 1] = res[j]\n            j -= 1\n        res[j + 1] = key\n    return res\n',
    'def merge_sorted_lists(l1: list[int], l2: list[int]) -> list[int]:\n    """Merge two sorted lists into single sorted list."""\n    res = []\n    i, j = 0, 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] <= l2[j]:\n            res.append(l1[i])\n            i += 1\n        else:\n            res.append(l2[j])\n            j += 1\n    res.extend(l1[i:])\n    res.extend(l2[j:])\n    return res\n',
    'def two_sum(nums: list[int], target: int) -> list[int]:\n    """Find indices of two numbers that add up to target."""\n    lookup = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in lookup:\n            return [lookup[complement], i]\n        lookup[num] = i\n    return []\n',
    'def max_subarray_sum(nums: list[int]) -> int:\n    """Find maximum subarray sum using Kadane\'s algorithm."""\n    if not nums:\n        return 0\n    max_so_far = max_ending_here = nums[0]\n    for x in nums[1:]:\n        max_ending_here = max(x, max_ending_here + x)\n        max_so_far = max(max_so_far, max_ending_here)\n    return max_so_far\n',
    'def intersect_lists(a: list, b: list) -> list:\n    """Return common elements preserving order of a."""\n    b_set = set(b)\n    return [x for x in a if x in b_set]\n',
    'def interleave_lists(a: list, b: list) -> list:\n    """Interleave elements of two lists."""\n    res = []\n    for x, y in zip(a, b):\n        res.extend([x, y])\n    min_len = min(len(a), len(b))\n    res.extend(a[min_len:])\n    res.extend(b[min_len:])\n    return res\n',

    # Data Structures
    'class Stack:\n    """LIFO Stack implementation."""\n    def __init__(self):\n        self._items = []\n    def push(self, item):\n        self._items.append(item)\n    def pop(self):\n        if self.is_empty():\n            raise IndexError("pop from empty stack")\n        return self._items.pop()\n    def peek(self):\n        return self._items[-1] if not self.is_empty() else None\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n    def size(self) -> int:\n        return len(self._items)\n',
    'class Queue:\n    """FIFO Queue implementation."""\n    def __init__(self):\n        self._items = []\n    def enqueue(self, item):\n        self._items.append(item)\n    def dequeue(self):\n        if self.is_empty():\n            raise IndexError("dequeue from empty queue")\n        return self._items.pop(0)\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n    def size(self) -> int:\n        return len(self._items)\n',
    'class ListNode:\n    """Singly linked list node."""\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n',
    'class LinkedList:\n    """Simple Singly Linked List."""\n    def __init__(self):\n        self.head = None\n    def append(self, val):\n        if not self.head:\n            self.head = ListNode(val)\n            return\n        curr = self.head\n        while curr.next:\n            curr = curr.next\n        curr.next = ListNode(val)\n    def to_list(self) -> list:\n        res = []\n        curr = self.head\n        while curr:\n            res.append(curr.val)\n            curr = curr.next\n        return res\n',
    'class TreeNode:\n    """Binary tree node."""\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n',
    'class BinarySearchTree:\n    """Binary Search Tree supporting insert and lookup."""\n    def __init__(self):\n        self.root = None\n    def insert(self, val: int):\n        if not self.root:\n            self.root = TreeNode(val)\n            return\n        curr = self.root\n        while True:\n            if val < curr.val:\n                if not curr.left:\n                    curr.left = TreeNode(val)\n                    break\n                curr = curr.left\n            else:\n                if not curr.right:\n                    curr.right = TreeNode(val)\n                    break\n                curr = curr.right\n    def search(self, val: int) -> bool:\n        curr = self.root\n        while curr:\n            if curr.val == val:\n                return True\n            curr = curr.left if val < curr.val else curr.right\n        return False\n',
    'class MinHeapSimple:\n    """Basic binary min heap array implementation."""\n    def __init__(self):\n        self.heap = []\n    def push(self, val: int):\n        self.heap.append(val)\n        self._sift_up(len(self.heap) - 1)\n    def pop_min(self) -> int:\n        if not self.heap:\n            raise IndexError("Heap is empty")\n        min_val = self.heap[0]\n        last = self.heap.pop()\n        if self.heap:\n            self.heap[0] = last\n            self._sift_down(0)\n        return min_val\n    def _sift_up(self, idx: int):\n        parent = (idx - 1) // 2\n        while idx > 0 and self.heap[idx] < self.heap[parent]:\n            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]\n            idx = parent\n            parent = (idx - 1) // 2\n    def _sift_down(self, idx: int):\n        n = len(self.heap)\n        while True:\n            smallest = idx\n            left, right = 2 * idx + 1, 2 * idx + 2\n            if left < n and self.heap[left] < self.heap[smallest]:\n                smallest = left\n            if right < n and self.heap[right] < self.heap[smallest]:\n                smallest = right\n            if smallest == idx:\n                break\n            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]\n            idx = smallest\n',
    'class LRUCacheSimple:\n    """Least Recently Used (LRU) Cache."""\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache: dict[str, Any] = {}\n    def get(self, key: str) -> Any:\n        if key not in self.cache:\n            return -1\n        val = self.cache.pop(key)\n        self.cache[key] = val\n        return val\n    def put(self, key: str, value: Any) -> None:\n        if key in self.cache:\n            self.cache.pop(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = next(iter(self.cache))\n            del self.cache[oldest]\n        self.cache[key] = value\n',

    # Logic, Conditionals, and Business Rules
    'def fizzbuzz(n: int) -> list[str]:\n    """Generate FizzBuzz sequence up to n."""\n    res = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            res.append("FizzBuzz")\n        elif i % 3 == 0:\n            res.append("Fizz")\n        elif i % 5 == 0:\n            res.append("Buzz")\n        else:\n            res.append(str(i))\n    return res\n',
    'def is_leap_year(year: int) -> bool:\n    """Check if year is a leap year in Gregorian calendar."""\n    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)\n',
    'def calculate_tax(income: float) -> float:\n    """Calculate progressive income tax under brackets."""\n    if income <= 10000:\n        return income * 0.10\n    elif income <= 50000:\n        return 1000 + (income - 10000) * 0.20\n    else:\n        return 9000 + (income - 50000) * 0.30\n',
    'def roman_to_int(s: str) -> int:\n    """Convert Roman numeral string to integer."""\n    val_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}\n    total = 0\n    prev = 0\n    for char in reversed(s):\n        curr = val_map.get(char, 0)\n        if curr < prev:\n            total -= curr\n        else:\n            total += curr\n        prev = curr\n    return total\n',
    'def evaluate_bool_expression(a: bool, b: bool, op: str) -> bool:\n    """Evaluate binary boolean operation."""\n    if op == "AND": return a and b\n    if op == "OR": return a or b\n    if op == "XOR": return a != b\n    if op == "NAND": return not (a and b)\n    raise ValueError(f"Unknown boolean op: {op}")\n',
    'def validate_password_strength(pw: str) -> bool:\n    """Verify password has length >= 8, digit, and uppercase."""\n    if len(pw) < 8:\n        return False\n    has_digit = any(c.isdigit() for c in pw)\n    has_upper = any(c.isupper() for c in pw)\n    has_lower = any(c.islower() for c in pw)\n    return has_digit and has_upper and has_lower\n',

    # Systems & Utilities
    'def safe_json_loads(json_str: str, default: Any = None) -> Any:\n    """Parse JSON string safely returning default on decode error."""\n    import json\n    try:\n        return json.loads(json_str)\n    except (json.JSONDecodeError, TypeError):\n        return default\n',
    'def format_file_size(bytes_val: int) -> str:\n    """Format bytes into human-readable unit string."""\n    for unit in ["B", "KB", "MB", "GB", "TB"]:\n        if bytes_val < 1024.0:\n            return f"{bytes_val:.1f} {unit}"\n        bytes_val /= 1024.0\n    return f"{bytes_val:.1f} PB"\n',
    'def deep_get(d: dict, keys: list[str], default: Any = None) -> Any:\n    """Traverse nested dictionary using sequence of keys."""\n    curr = d\n    for k in keys:\n        if isinstance(curr, dict) and k in curr:\n            curr = curr[k]\n        else:\n            return default\n    return curr\n',
    'def batch_iterable(items: list, batch_size: int):\n    """Yield batches of fixed size from list."""\n    for i in range(0, len(items), batch_size):\n        yield items[i:i + batch_size]\n',
    'def hash_djb2(s: str) -> int:\n    """Compute 32-bit djb2 string hash."""\n    hash_val = 5381\n    for char in s:\n        hash_val = ((hash_val << 5) + hash_val) + ord(char)\n        hash_val &= 0xFFFFFFFF\n    return hash_val\n',
]


def generate_default_code_dataset(
    output_path: str = "data/train_code.jsonl",
    target_count: int = 2000,
) -> int:
    """
    Generates a curated dataset of real Python functions from the standard library
    supplemented by algorithmic templates. Every function is verified with ast.parse().
    """
    import ast
    from pathlib import Path
    import sys

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    examples: List[Dict[str, Any]] = []

    # 1. Base algorithmic and utility templates
    for i, code in enumerate(DEFAULT_CODE_EXAMPLES):
        examples.append({
            "id": len(examples),
            "category": "base_template",
            "name": f"template_{i}",
            "code": code.strip(),
        })

    # 2. Extract real functions from Python Standard Library
    lib_dir = Path(sys.base_prefix) / "Lib"
    if lib_dir.exists():
        py_files = [
            p for p in lib_dir.rglob("*.py")
            if "test" not in p.parts and "site-packages" not in p.parts and not p.name.startswith("test_")
        ]

        for fpath in py_files:
            if len(examples) >= target_count:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    code_text = f.read()
                tree = ast.parse(code_text)
                lines = code_text.splitlines()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name.startswith("__"):
                            continue
                        start_line = node.lineno - 1
                        end_line = getattr(node, "end_lineno", start_line + 40)
                        func_code = "\n".join(lines[start_line:end_line]).strip()
                        func_lines = func_code.splitlines()
                        # Select functions with 5 to 50 lines
                        if 5 <= len(func_lines) <= 50:
                            try:
                                ast.parse(func_code)
                                examples.append({
                                    "id": len(examples),
                                    "category": "stdlib",
                                    "name": node.name,
                                    "module": fpath.stem,
                                    "code": func_code,
                                })
                                if len(examples) >= target_count:
                                    break
                            except Exception:
                                pass
            except Exception:
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset successfully created with {len(examples)} examples at: {output_path}")
    return len(examples)


class CodeDistillationDataset(Dataset):
    """
    Packs tokenized code samples into fixed-length contiguous token sequences
    for causal representation distillation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int = 256,
        min_samples: int = 1500,
        vocab_size: int = 151936,
        vocab_map_path: str = "checkpoints/vocab_16k/vocab_map.pt",
    ) -> None:
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.vocab_map_path = vocab_map_path

        # If data file does not exist or has fewer samples than required, regenerate
        should_generate = not os.path.exists(data_path)
        if not should_generate:
            with open(data_path, "r", encoding="utf-8") as f:
                line_count = sum(1 for line in f if line.strip())
            if line_count < min_samples:
                print(f"Existing dataset has only {line_count} samples (< {min_samples}). Regenerating...")
                should_generate = True

        if should_generate:
            print(f"Generating rich standard library code dataset (~2000 functions) at '{data_path}'...")
            generate_default_code_dataset(data_path, target_count=2000)

        # Load raw code texts
        code_texts: List[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    code = entry.get("code") or entry.get("text", "")
                    if code:
                        code_texts.append(code)

        if not code_texts:
            raise ValueError(f"No valid code samples found in {data_path}!")

        # Check for 16k vocabulary mapping
        use_16k = (vocab_size == 16384)
        old_to_new: Optional[torch.Tensor] = None
        if use_16k:
            if not os.path.exists(vocab_map_path):
                raise FileNotFoundError(
                    f"16k vocab map '{vocab_map_path}' not found! Run tools/prune_vocab.py first."
                )
            vocab_map_data = torch.load(vocab_map_path, map_location="cpu")
            old_to_new = vocab_map_data["old_to_new"]
            raw_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151643
            eos_id = old_to_new[raw_eos].item()
            if eos_id == -1:
                eos_id = 0
            print(f"[Dataset] Using 16k vocabulary mapping (EOS 16k ID: {eos_id}).")
        else:
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        # Tokenize and pack into continuous token stream
        all_token_ids: List[int] = []

        for text in code_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if use_16k and old_to_new is not None:
                tokens_mapped = []
                for tid in tokens:
                    mapped = old_to_new[tid].item() if tid < len(old_to_new) else -1
                    if mapped == -1:
                        # Fallback to eos_id if an unmapped token is encountered
                        mapped = eos_id
                    tokens_mapped.append(mapped)
                all_token_ids.extend(tokens_mapped)
            else:
                all_token_ids.extend(tokens)
            all_token_ids.append(eos_id)

        if use_16k:
            assert all(0 <= tid < 16384 for tid in all_token_ids), "Found token ID out of 16k range [0, 16383]!"

        # Slice into chunks of exact length seq_len
        self.chunks: List[torch.Tensor] = []
        for i in range(0, len(all_token_ids) - seq_len + 1, seq_len):
            chunk = all_token_ids[i : i + seq_len]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

        # Handle remaining tokens with padding if chunks list is too small
        if len(self.chunks) == 0:
            padded = all_token_ids + [eos_id] * (seq_len - len(all_token_ids))
            self.chunks.append(torch.tensor(padded[:seq_len], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]
