import ast
from pathlib import Path
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
lib_dir = Path(sys.base_prefix) / "Lib"
py_files = [p for p in lib_dir.rglob("*.py") if "test" not in p.parts and "site-packages" not in p.parts]

snippets = []
for fpath in py_files:
    if len(snippets) >= 3000:
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
                end_line = getattr(node, "end_lineno", start_line + 25)
                func_code = "\n".join(lines[start_line:end_line]).strip()
                func_lines = func_code.splitlines()
                # Clean function criteria: 4 to 45 lines, must be parsable
                if 4 <= len(func_lines) <= 45:
                    try:
                        ast.parse(func_code)
                        snippets.append(func_code)
                        if len(snippets) >= 3000:
                            break
                    except Exception:
                        pass
    except Exception:
        continue

print(f"Extracted {len(snippets)} clean functions!")
# Tokenize sample
sample_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in snippets[:100])
print(f"Average tokens per function: {sample_tokens / 100:.1f}")
print(f"Estimated total tokens for 3000: {sample_tokens * 30}")
