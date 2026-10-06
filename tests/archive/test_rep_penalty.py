import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# What if repetition penalty is 1.15?
def generate_with_penalty(rep_penalty=1.15):
    curr = inputs['input_ids'].clone()
    suppressed_token_ids = list(range(151644, 151665))
    
    for _ in range(35):
        logits = grafted.forward(curr, mode="grafted")
        next_logits = logits[0, -1, :].clone()
        for fid in suppressed_token_ids:
            if fid < next_logits.shape[-1]:
                next_logits[fid] = -float("inf")
        
        # Repetition penalty
        for tid in set(curr[0].tolist()):
            if next_logits[tid] > 0:
                next_logits[tid] /= rep_penalty
            else:
                next_logits[tid] *= rep_penalty
                
        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True).unsqueeze(0)
        curr = torch.cat([curr, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

print("Repetition penalty 1.15:")
print(repr(generate_with_penalty(1.15)))

print("Repetition penalty 1.25:")
print(repr(generate_with_penalty(1.25)))
