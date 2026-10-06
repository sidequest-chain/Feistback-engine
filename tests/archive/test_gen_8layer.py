import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(num_grafted_layers=8, layer_scale=0.0)
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'

text_graft, ids_graft = grafted.generate(prompt, max_new_tokens=40, mode="grafted")
print("Generated Code with 16 retained + 8 Feistel layers:")
print(repr(text_graft))
print("\nFormatted:")
print(prompt + text_graft)
