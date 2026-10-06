import sys
import subprocess
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

functional_tasks = [
    {
        "name": "add",
        "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
        "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n",
    },
    {
        "name": "is_even",
        "prompt": 'def is_even(n: int) -> bool:\n    """Return True if n is even, False otherwise."""\n',
        "test": "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True\n",
    },
    {
        "name": "reverse_str",
        "prompt": 'def reverse_str(s: str) -> str:\n    """Return the reversed string."""\n',
        "test": "assert reverse_str('hello') == 'olleh'\nassert reverse_str('') == ''\n",
    },
    {
        "name": "max_num",
        "prompt": 'def max_num(lst: list) -> int:\n    """Return the maximum number in the list."""\n',
        "test": "assert max_num([1, 5, 3, 9, 2]) == 9\nassert max_num([-5, -1]) == -1\n",
    },
    {
        "name": "count_vowels",
        "prompt": 'def count_vowels(s: str) -> int:\n    """Return the number of vowels in the string."""\n',
        "test": "assert count_vowels('hello world') == 3\nassert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5\n",
    },
    {
        "name": "is_palindrome",
        "prompt": 'def is_palindrome(s: str) -> bool:\n    """Return True if s is a palindrome, False otherwise."""\n',
        "test": "assert is_palindrome('radar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('a') is True\n",
    },
    {
        "name": "square_list",
        "prompt": 'def square_list(lst: list) -> list:\n    """Return a new list with each element squared."""\n',
        "test": "assert square_list([1, 2, 3]) == [1, 4, 9]\nassert square_list([]) == []\n",
    },
    {
        "name": "fibonacci",
        "prompt": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n',
        "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8\n",
    },
]

for task in functional_tasks:
    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    cleaned = extract_function_body(raw)
    script_code = prompt + cleaned + "\n\n" + task["test"]
    proc = subprocess.run([sys.executable, "-c", script_code], capture_output=True, text=True, timeout=3.0)
    print(f"Task {task['name']:<15}: ReturnCode={proc.returncode}")
    if proc.returncode != 0:
        print("STDERR:", proc.stderr.strip()[:100])
        print("CLEANED CODE:\n", script_code)
