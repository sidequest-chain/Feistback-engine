import ast
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16, device_map="cuda")

def extract_function_body(completion: str) -> str:
    lines = completion.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            if any(line.startswith(prefix) for prefix in ("def ", "class ", "@", "if __name__")):
                break
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

ast_prompts = [
    'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
    'def is_even(n: int) -> bool:\n    """Check if number is even."""\n',
    'def reverse_string(s: str) -> str:\n    """Reverse the given string."""\n',
    'def find_max(numbers: list[int]) -> int:\n    """Return the maximum element."""\n',
    'def count_vowels(text: str) -> int:\n    """Count vowels in the string."""\n',
    'def is_palindrome(word: str) -> bool:\n    """Check if word is a palindrome."""\n',
    'def square_elements(nums: list[int]) -> list[int]:\n    """Return list of squared numbers."""\n',
    'def get_first_item(items: list) -> any:\n    """Return the first item or None."""\n',
    'def compute_average(values: list[float]) -> float:\n    """Compute average of float values."""\n',
    'def filter_positive(nums: list[int]) -> list[int]:\n    """Filter only positive numbers."""\n',
    'def repeat_string(s: str, n: int) -> str:\n    """Repeat string n times."""\n',
    'def is_empty(container: list) -> bool:\n    """Check if list is empty."""\n',
]

passed = 0
for p in ast_prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_function_body(raw)
    code = p + cleaned
    try:
        ast.parse(code)
        passed += 1
        print(f"PASS: {p.strip().split(chr(10))[0]}")
    except Exception as e:
        print(f"FAIL: {p.strip().split(chr(10))[0]} -> {e}")
        print("CODE:\n", code)

print(f"\nTotal AST passed: {passed}/{len(ast_prompts)}")
