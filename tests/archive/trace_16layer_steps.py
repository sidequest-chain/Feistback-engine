import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(num_grafted_layers=8, layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

curr = inputs['input_ids'].clone()
suppressed_token_ids = list(range(151644, 151665))

for step in range(15):
    logits = grafted.forward(curr, mode="grafted")
    next_logits = logits[0, -1, :].clone()
    for fid in suppressed_token_ids:
        if fid < next_logits.shape[-1]:
            next_logits[fid] = -float("inf")
    
    # Check top 5
    top5 = torch.topk(next_logits, 5)
    print(f"\nStep {step}:")
    for idx, v in zip(top5.indices, top5.values):
        s = repr(tokenizer.decode([idx.item()])).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"  {idx.item()}: {s} ({v.item():.2f})")

    # Greedy with repetition penalty
    for tid in set(curr[0].tolist()):
        if next_logits[tid] > 0:
            next_logits[tid] /= 1.15
        else:
            next_logits[tid] *= 1.15
            
    next_tok = torch.argmax(next_logits, dim=-1, keepdim=True).unsqueeze(0)
    curr = torch.cat([curr, next_tok], dim=-1)
    if next_tok.item() == tokenizer.eos_token_id:
        break

print("\nFinal generated text:")
print(repr(tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)))
