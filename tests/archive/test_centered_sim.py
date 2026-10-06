import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

out = model(**inputs, output_hidden_states=True)
hs_20 = out.hidden_states[20]
logits_24 = out.logits[:, -1, :].float()
logits_20 = model.lm_head(model.model.norm(hs_20))[:, -1, :].float()

print('Raw logits cosine sim:', torch.cosine_similarity(logits_24, logits_20, dim=-1).item())

# Mean-centered logits (since softmax is invariant to constant shift!):
l24_centered = logits_24 - logits_24.mean(dim=-1, keepdim=True)
l20_centered = logits_20 - logits_20.mean(dim=-1, keepdim=True)
print('Centered logits cosine sim:', torch.cosine_similarity(l24_centered, l20_centered, dim=-1).item())

# Softmax probs cosine sim:
p24 = torch.softmax(logits_24, dim=-1)
p20 = torch.softmax(logits_20, dim=-1)
print('Softmax probs cosine sim:', torch.cosine_similarity(p24, p20, dim=-1).item())

# Top-100 logits cosine sim:
topk_idx = torch.topk(logits_24, 100).indices
print('Top-100 logits cosine sim:', torch.cosine_similarity(logits_24[:, topk_idx[0]], logits_20[:, topk_idx[0]], dim=-1).item())
