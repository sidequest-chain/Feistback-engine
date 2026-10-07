"""
Verify Clean Algorithmic Python Dataset
=======================================
Validates data/train_code_clean.jsonl using Python's `ast` module:
1. Docstring count (must be exactly 0)
2. Variable distribution (no single argument > 15% of all argument occurrences)
3. Variable binding check (0 unbound / undeclared variables in function bodies)
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train_code_clean.jsonl"

PYTHON_BUILTINS: Set[str] = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "math", "isqrt", "pow",
}


def check_docstring(fn_node: ast.FunctionDef) -> bool:
    """Returns True if the function has a docstring."""
    if not fn_node.body:
        return False
    first = fn_node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return True
    return False


def get_assigned_names(target: ast.AST) -> Set[str]:
    """Extracts variable names bound by an assignment target."""
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(get_assigned_names(elt))
    elif isinstance(target, ast.Starred):
        names.update(get_assigned_names(target.value))
    return names


def verify_scope_binding(fn_node: ast.FunctionDef) -> List[str]:
    """
    Checks that every variable loaded in the function body was declared in
    the arguments, assigned locally, or is a standard Python builtin.
    Returns a list of error descriptions (empty if valid).
    """
    errors: List[str] = []
    declared_args: Set[str] = {arg.arg for arg in fn_node.args.args}
    
class ScopeChecker(ast.NodeVisitor):
    def __init__(self, declared_args: Set[str]):
        # Scope stack: list of sets of variable names
        self.scopes: List[Set[str]] = [set(declared_args)]
        self.errors: List[str] = []

    def current_scope(self) -> Set[str]:
        return self.scopes[-1]

    def is_bound(self, name: str) -> bool:
        if name in PYTHON_BUILTINS:
            return True
        for scope in reversed(self.scopes):
            if name in scope:
                return True
        return False

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for target in node.targets:
            self.current_scope().update(get_assigned_names(target))

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.value)
        self.visit(node.target)
        self.current_scope().update(get_assigned_names(node.target))

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self.current_scope().update(get_assigned_names(node.target))
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While):
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_ListComp(self, node: ast.ListComp):
        # A listcomp opens a new nested lexical scope
        new_scope = set()
        for gen in node.generators:
            self.visit(gen.iter)
            new_scope.update(get_assigned_names(gen.target))
        self.scopes.append(new_scope)
        for gen in node.generators:
            for if_clause in gen.ifs:
                self.visit(if_clause)
        self.visit(node.elt)
        self.scopes.pop()

    def visit_SetComp(self, node: ast.SetComp):
        new_scope = set()
        for gen in node.generators:
            self.visit(gen.iter)
            new_scope.update(get_assigned_names(gen.target))
        self.scopes.append(new_scope)
        for gen in node.generators:
            for if_clause in gen.ifs:
                self.visit(if_clause)
        self.visit(node.elt)
        self.scopes.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        new_scope = set()
        for gen in node.generators:
            self.visit(gen.iter)
            new_scope.update(get_assigned_names(gen.target))
        self.scopes.append(new_scope)
        for gen in node.generators:
            for if_clause in gen.ifs:
                self.visit(if_clause)
        self.visit(node.elt)
        self.scopes.pop()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            if not self.is_bound(node.id):
                self.errors.append(f"Unbound variable '{node.id}'")


def verify_scope_binding(fn_node: ast.FunctionDef) -> List[str]:
    """
    Checks that every variable loaded in the function body was declared in
    the arguments, assigned locally, or is a standard Python builtin.
    Returns a list of error descriptions (empty if valid).
    """
    declared_args: Set[str] = {arg.arg for arg in fn_node.args.args}
    checker = ScopeChecker(declared_args)

    for stmt in fn_node.body:
        checker.visit(stmt)

    return checker.errors


def verify_dataset() -> bool:
    print("=" * 70)
    print("VERIFYING DATASET VIA AST (data/train_code_clean.jsonl)")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset file not found: {DATA_PATH}")
        return False

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_funcs = len(lines)
    print(f"Total function samples: {total_funcs}")

    total_docstrings = 0
    arg_counter: Counter[str] = Counter()
    total_args = 0
    binding_errors: List[str] = []
    syntax_errors: List[str] = []

    for idx, line in enumerate(lines):
        try:
            entry = json.loads(line)
            code = entry.get("code", "")
        except Exception as e:
            syntax_errors.append(f"Line {idx}: JSON parse error: {e}")
            continue

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            syntax_errors.append(f"Line {idx}: SyntaxError: {e}")
            continue

        func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not func_nodes:
            syntax_errors.append(f"Line {idx}: No FunctionDef found")
            continue

        fn = func_nodes[0]

        # 1. Docstring check
        if check_docstring(fn):
            total_docstrings += 1

        # Check for any rogue string constants as statements
        for stmt in fn.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    total_docstrings += 1

        # 2. Argument distribution
        for arg in fn.args.args:
            arg_counter[arg.arg] += 1
            total_args += 1

        # 3. Variable binding check
        fn_errors = verify_scope_binding(fn)
        if fn_errors:
            binding_errors.extend([f"Line {idx} ({fn.name}): {err}" for err in fn_errors])

    # Report 1: Docstrings
    print("\n[CHECK 1: Docstrings]")
    print(f"Docstrings found: {total_docstrings}")
    docstring_pass = (total_docstrings == 0)
    print(f"Status: {'PASSED' if docstring_pass else 'FAILED'}")

    # Report 2: Argument distribution
    print("\n[CHECK 2: Argument Frequency Distribution]")
    print(f"Total arguments observed: {total_args}")
    print(f"Unique argument names: {len(arg_counter)}")
    print(f"{'Argument':<15} | {'Count':<8} | {'Percentage':<10} | {'Status'}")
    print("-" * 50)

    arg_dist_pass = True
    for arg, count in arg_counter.most_common():
        pct = (count / total_args) * 100.0 if total_args > 0 else 0.0
        is_ok = pct <= 15.0
        if not is_ok:
            arg_dist_pass = False
        print(f"{arg:<15} | {count:<8} | {pct:>8.2f}% | {'OK' if is_ok else 'EXCEEDS 15%'}")

    print(f"Status: {'PASSED' if arg_dist_pass else 'FAILED'}")

    # Report 3: Binding check
    print("\n[CHECK 3: Variable Binding & Undeclared References]")
    print(f"Syntax/Parse errors: {len(syntax_errors)}")
    print(f"Unbound variable errors: {len(binding_errors)}")
    if binding_errors:
        print("Sample errors:")
        for err in binding_errors[:10]:
            print(f"  - {err}")
    binding_pass = (len(syntax_errors) == 0 and len(binding_errors) == 0)
    print(f"Status: {'PASSED' if binding_pass else 'FAILED'}")

    print("\n" + "=" * 70)
    overall_pass = docstring_pass and arg_dist_pass and binding_pass
    print(f"OVERALL VERIFICATION: {'PASSED' if overall_pass else 'FAILED'}")
    print("=" * 70)

    return overall_pass


if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
