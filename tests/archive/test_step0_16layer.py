import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(num_grafted_layers=8, layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

logits = grafted.forward(inputs['input_ids'], mode="grafted")
next_logits = logits[0, -1, :]

top10 = torch.topk(next_logits, 10)
print("Top 10 tokens at step 0 for 16-layer grafted:")
for idx, val in zip(top10.indices, top10.values):
    print(f"  {idx.item()}: {repr(tokenizer.decode([idx.item()]))} ({val.item():.2f})")
