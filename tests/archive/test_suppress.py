import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
encoded = grafted.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(grafted.device)

# Banned tokens: all tokens >= 151643 (special tokens, FIM tokens, etc.) except None
# Or specifically FIM tokens [151659, 151660, 151661, 151662, 151663, 151664] or all tokens > 151645
print("Testing with FIM tokens suppressed:")
curr_ids = input_ids.clone()
for step in range(35):
    logits = grafted.forward(curr_ids, mode="grafted")
    next_tok_logits = logits[0, -1, :].clone()
    # Suppress special tokens >= 151643 except eos_token_id (151643)
    next_tok_logits[151644:] = -float("inf")
    next_tok = torch.argmax(next_tok_logits, dim=-1, keepdim=True).unsqueeze(0)
    if next_tok.item() == grafted.tokenizer.eos_token_id:
        break
    curr_ids = torch.cat([curr_ids, next_tok], dim=-1)

gen_text = grafted.tokenizer.decode(curr_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
print("Generated code with suppressed special tokens:")
print(prompt + gen_text)
