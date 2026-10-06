import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tok(prompt, return_tensors='pt').to('cuda')
out = model(**inputs, output_hidden_states=True)
# hs[16] is output of layer 15 (after 16 layers: 0..15)
hs_16 = out.hidden_states[16]
logits_16 = model.lm_head(model.model.norm(hs_16))[:, -1, :].float()
logits_24 = out.logits[:, -1, :].float()

p16 = torch.softmax(logits_16, dim=-1)
p24 = torch.softmax(logits_24, dim=-1)
print('Probability cosine sim layer 16 vs 24:', torch.cosine_similarity(p16, p24, dim=-1).item())
print('Raw logits cosine sim layer 16 vs 24:', torch.cosine_similarity(logits_16, logits_24, dim=-1).item())

top_tok_24 = torch.argmax(logits_24, dim=-1).item()
top_tok_16 = torch.argmax(logits_16, dim=-1).item()
print('Top token 24:', repr(tok.decode([top_tok_24])))
print('Top token 16:', repr(tok.decode([top_tok_16])))
