import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
tokenizer = grafted.tokenizer

p = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if n'
inputs = tokenizer(p, return_tensors='pt').to('cuda')

curr = inputs['input_ids'].clone()
for step in range(30):
    logits = grafted.forward(curr, mode="grafted")
    next_tok = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True).unsqueeze(0)
    curr = torch.cat([curr, next_tok], dim=-1)
    if next_tok.item() == tokenizer.eos_token_id:
        break

print(tokenizer.decode(curr[0], skip_special_tokens=False))
