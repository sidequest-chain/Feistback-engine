import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

out = model(**inputs, output_hidden_states=True)
hs_20 = out.hidden_states[20]
hs_24 = out.hidden_states[24]

print('Norm hs_20:', hs_20.norm(dim=-1).mean().item())
print('Norm hs_24:', hs_24.norm(dim=-1).mean().item())

logits_20 = model.lm_head(model.model.norm(hs_20))
logits_24 = out.logits

probs_20 = torch.softmax(logits_20[:, -1, :].float(), dim=-1)
probs_24 = torch.softmax(logits_24[:, -1, :].float(), dim=-1)

print('Probs cosine similarity:', torch.cosine_similarity(probs_20, probs_24, dim=-1).item())
