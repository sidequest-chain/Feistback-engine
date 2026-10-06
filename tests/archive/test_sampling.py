import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
tokenizer = grafted.tokenizer
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Suppress FIM and special tokens except EOS
fim_and_special = [151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656, 151657, 151658, 151659, 151660, 151661, 151662, 151663, 151664]

# Let's test what happens if we suppress non-ASCII or rare tokens, or if we use top_k / top_p:
for top_p in [0.8, 0.9, 0.95]:
    for temp in [0.2, 0.4, 0.7]:
        curr = inputs['input_ids'].clone()
        torch.manual_seed(42)
        for _ in range(30):
            logits = grafted.forward(curr, mode="grafted")
            next_logits = logits[0, -1, :].clone()
            for fid in fim_and_special:
                next_logits[fid] = -float("inf")
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits / temp, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(next_logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            curr = torch.cat([curr, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
        
        gen = tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(f"--- top_p={top_p}, temp={temp} ---")
        print(repr(gen))
