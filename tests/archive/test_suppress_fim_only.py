import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Only suppress FIM tokens
fim_ids = [151659, 151660, 151661, 151662, 151663, 151664]

curr = inputs['input_ids'].clone()
for step in range(35):
    logits = grafted.forward(curr, mode="grafted")
    next_tok_logits = logits[0, -1, :].clone()
    for fid in fim_ids:
        next_tok_logits[fid] = -float("inf")
    next_tok = torch.argmax(next_tok_logits, dim=-1, keepdim=True).unsqueeze(0)
    tok_id = next_tok.item()
    curr = torch.cat([curr, next_tok], dim=-1)
    if tok_id == tokenizer.eos_token_id:
        break

print(prompt + tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True))
