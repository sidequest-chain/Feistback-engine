import ast
from pathlib import Path
import sys

lib_dir = Path(sys.base_prefix) / "Lib"
py_files = [p for p in lib_dir.rglob("*.py") if "test" not in p.parts and "site-packages" not in p.parts]
print(f"Candidate python files: {len(py_files)}")

snippets = []
for fpath in py_files:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            code_text = f.read()
        tree = ast.parse(code_text)
        lines = code_text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", start_line + 25)
                func_code = "\n".join(lines[start_line:end_line]).strip()
                func_lines = func_code.splitlines()
                # Keep functions with 3 to 70 lines
                if 3 <= len(func_lines) <= 70 and not node.name.startswith("__"):
                    snippets.append({
                        "name": node.name,
                        "file": fpath.name,
                        "code": func_code,
                    })
    except Exception:
        continue

print(f"Total extracted functions: {len(snippets)}")
