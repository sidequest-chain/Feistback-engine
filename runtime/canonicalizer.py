"""
AST Signature Canonicalization & Decanonicalization Adapter
===========================================================
Maps arbitrary out-of-distribution (OOD) function parameter names to the
PureFeistelLM model's orthogonal register pools ('x', 'a'/'b', 'u'/'v'/'w')
before inference, and restores original identifier bindings afterward.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, Tuple


KNOWN_SINGLE_VARS = {"n", "x", "val", "num", "value", "a", "k", "item", "target", "elem"}


def canonicalize_signature(prompt: str) -> Tuple[str, Dict[str, str]]:
    """
    Parses the function header in `prompt` using Python's `ast` module and maps
    parameter names to canonical orthogonal registers:
      - 1 argument : maps to 'x' (unless already in KNOWN_SINGLE_VARS).
      - 2 arguments: maps to ('a', 'b').
      - 3 arguments: maps to ('u', 'v', 'w').
      - 4+ arguments: preserved as-is.

    Returns:
        Tuple[str, Dict[str, str]]:
            - Canonicalized function signature string.
            - Inverse mapping dictionary {canonical_name: original_name}.
    """
    inv_map: Dict[str, str] = {}
    clean_p = prompt.strip()

    try:
        tree = ast.parse(clean_p + "\n    pass")
    except SyntaxError:
        return prompt, inv_map

    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return prompt, inv_map

    fn_node = tree.body[0]
    args = fn_node.args.args
    if not args:
        return prompt, inv_map

    num_args = len(args)
    if num_args == 1:
        orig = args[0].arg
        targets = [orig] if orig in KNOWN_SINGLE_VARS else ["x"]
    elif num_args == 2:
        targets = ["a", "b"]
    elif num_args == 3:
        targets = ["u", "v", "w"]
    else:
        targets = [arg.arg for arg in args]

    for arg, target in zip(args, targets):
        if arg.arg != target:
            inv_map[target] = arg.arg
            arg.arg = target

    if not inv_map:
        return prompt, inv_map

    canonical_header = ast.unparse(fn_node).split("\n")[0]
    canonical_prompt = canonical_header + "\n" if prompt.endswith("\n") else canonical_header

    return canonical_prompt, inv_map


def decanonicalize_code(completion: str, inv_map: Dict[str, str]) -> str:
    """
    Replaces canonical variable names in the generated code body back to the
    original prompt variable names using strict word-boundary matching.
    """
    if not inv_map:
        return completion

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in inv_map.keys()) + r")\b")
    return pattern.sub(lambda m: inv_map[m.group(0)], completion)


__all__ = [
    "KNOWN_SINGLE_VARS",
    "canonicalize_signature",
    "decanonicalize_code",
]
