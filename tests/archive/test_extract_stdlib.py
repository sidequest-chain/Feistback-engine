import ast
import inspect
import os
import sys
from pathlib import Path

lib_dir = Path(sys.base_prefix) / "Lib"
print("Lib dir:", lib_dir)

py_files = list(lib_dir.glob("*.py"))
print("Top-level py files count:", len(py_files))

# Count functions extractable via ast
extracted = 0
module_names = [
    "pathlib", "collections", "json", "functools", "itertools", "urllib",
    "dataclasses", "os", "re", "shutil", "tempfile", "hashlib", "typing",
    "enum", "contextlib", "heapq", "bisect", "copy", "textwrap", "datetime",
    "decimal", "csv", "difflib", "inspect", "dis", "statistics", "random",
    "string", "glob", "fnmatch", "uuid", "zipfile", "tarfile", "configparser",
    "argparse", "logging", "asyncio", "socket", "http", "email", "xml", "html",
    "doctest", "unittest", "tokenize", "token", "symbol", "keyword", "operator"
]

snippets = []
for mod_name in module_names:
    py_path = lib_dir / f"{mod_name}.py"
    if not py_path.exists():
        py_dir = lib_dir / mod_name
        if py_dir.exists():
            py_files_sub = list(py_dir.glob("*.py"))
        else:
            continue
    else:
        py_files_sub = [py_path]

    for fpath in py_files_sub:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                code_text = f.read()
            tree = ast.parse(code_text)
            lines = code_text.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get source code of function
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 20)
                    func_code = "\n".join(lines[start_line:end_line]).strip()
                    # Filter functions that are reasonable size (between 3 and 60 lines)
                    func_lines = func_code.splitlines()
                    if 3 <= len(func_lines) <= 60 and not func_code.startswith("def _"):
                        snippets.append({
                            "module": mod_name,
                            "name": node.name,
                            "code": func_code,
                        })
        except Exception:
            continue

print(f"Total extracted functions (non-private, 3-60 lines): {len(snippets)}")
