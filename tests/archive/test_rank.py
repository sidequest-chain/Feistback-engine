import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')

p = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if'
inputs = tokenizer(p, return_tensors='pt').to('cuda')
out = model(**inputs, output_hidden_states=True)
hs_20 = out.hidden_states[20]
logits_20 = model.lm_head(model.model.norm(hs_20))[0, -1, :]
logits_24 = out.logits[0, -1, :]

n_token_id = tokenizer.encode(' n', add_special_tokens=False)[0]
print('Token id for " n":', n_token_id)
print('Rank of " n" at layer 24:', (logits_24 > logits_24[n_token_id]).sum().item())
print('Rank of " n" at layer 20:', (logits_20 > logits_20[n_token_id]).sum().item())
print('Value of " n" at layer 24:', logits_24[n_token_id].item())
print('Value of " n" at layer 20:', logits_20[n_token_id].item())
