import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'

text_base, ids_base = grafted.generate(prompt, max_new_tokens=30, mode="baseline")
print("--- BASELINE ---")
print(repr(text_base))

text_graft, ids_graft = grafted.generate(prompt, max_new_tokens=30, mode="grafted")
print("--- GRAFTED ---")
print(repr(text_graft))
