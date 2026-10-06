import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# Base 24 layers forward
out = model(**inputs, output_hidden_states=True)
logits_24 = out.logits
hs_20 = out.hidden_states[20] # after layer 19
hs_24 = out.hidden_states[24] # after layer 23

# Norm + lm_head on hs_20:
norm_20 = model.model.norm(hs_20)
logits_from_20 = model.lm_head(norm_20)

cos_sim_hs = torch.cosine_similarity(hs_20[:, -1, :].float(), hs_24[:, -1, :].float(), dim=-1).item()
cos_sim_logits = torch.cosine_similarity(logits_from_20[:, -1, :].float(), logits_24[:, -1, :].float(), dim=-1).item()

print('Cosine sim hidden states layer 20 vs 24:', cos_sim_hs)
print('Cosine sim logits layer 20 vs 24:', cos_sim_logits)

top_token_24 = torch.argmax(logits_24[:, -1, :], dim=-1).item()
top_token_20 = torch.argmax(logits_from_20[:, -1, :], dim=-1).item()
print('Top token 24:', top_token_24, repr(tokenizer.decode([top_token_24])))
print('Top token 20:', top_token_20, repr(tokenizer.decode([top_token_20])))

# Let's inspect top 5 tokens for both
top5_24 = torch.topk(logits_24[:, -1, :], 5)
top5_20 = torch.topk(logits_from_20[:, -1, :], 5)

print("Top 5 at 24:")
for id, val in zip(top5_24.indices[0], top5_24.values[0]):
    print(f"  {id.item()}: {repr(tokenizer.decode([id.item()]))} ({val.item():.2f})")

print("Top 5 at 20:")
for id, val in zip(top5_20.indices[0], top5_20.values[0]):
    print(f"  {id.item()}: {repr(tokenizer.decode([id.item()]))} ({val.item():.2f})")
