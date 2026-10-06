import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Let's test with repetition penalty and suppressing FIM tokens
fim_ids = [151659, 151660, 151661, 151662, 151663, 151664]

def test_gen(temp=0.0, rep_penalty=1.1, top_k=50, top_p=0.9):
    curr = inputs['input_ids'].clone()
    for step in range(45):
        logits = grafted.forward(curr, mode="grafted")
        next_tok_logits = logits[0, -1, :].clone()
        for fid in fim_ids:
            next_tok_logits[fid] = -float("inf")
        
        # Apply repetition penalty
        if rep_penalty != 1.0:
            for token_id in set(curr[0].tolist()):
                if next_tok_logits[token_id] > 0:
                    next_tok_logits[token_id] /= rep_penalty
                else:
                    next_tok_logits[token_id] *= rep_penalty

        if temp > 0.0:
            probs = torch.softmax(next_tok_logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        else:
            next_tok = torch.argmax(next_tok_logits, dim=-1, keepdim=True).unsqueeze(0)

        tok_id = next_tok.item()
        curr = torch.cat([curr, next_tok], dim=-1)
        if tok_id == tokenizer.eos_token_id:
            break

    code = tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return code

print("--- Temp=0.0, Rep=1.2 ---")
print(test_gen(temp=0.0, rep_penalty=1.2))

print("--- Temp=0.2, Rep=1.1 ---")
print(test_gen(temp=0.2, rep_penalty=1.1))

print("--- Temp=0.5, Rep=1.1 ---")
print(test_gen(temp=0.5, rep_penalty=1.1))
