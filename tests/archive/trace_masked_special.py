import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
encoded = grafted.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(grafted.device)

curr_ids = input_ids.clone()
for step in range(40):
    logits = grafted.forward(curr_ids, mode="grafted")
    next_tok_logits = logits[0, -1, :].clone()
    # Mask all special tokens including FIM tokens
    next_tok_logits[151644:] = -float("inf")
    next_tok = torch.argmax(next_tok_logits, dim=-1, keepdim=True).unsqueeze(0)
    tok_id = next_tok.item()
    tok_str = repr(grafted.tokenizer.decode([tok_id])).encode('ascii', 'backslashreplace').decode('ascii')
    print(f"Step {step}: {tok_id} -> {tok_str}")
    curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
    if tok_id == grafted.tokenizer.eos_token_id:
        break

print("\nResult:")
print(grafted.tokenizer.decode(curr_ids[0], skip_special_tokens=True))
