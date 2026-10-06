import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')

# Prompt + '    if isinstance'
p = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if isinstance'
inputs = tokenizer(p, return_tensors='pt').to('cuda')
out = model(**inputs, output_hidden_states=True)

# 24 layer logits
logits_24 = out.logits[0, -1, :]
# 20 layer logits
hs_20 = out.hidden_states[20]
logits_20 = model.lm_head(model.model.norm(hs_20))[0, -1, :]

top5_24 = torch.topk(logits_24, 10)
top5_20 = torch.topk(logits_20, 10)

print('At layer 24 (Base):')
for idx, v in zip(top5_24.indices, top5_24.values):
    print(f'  {idx.item()}: {repr(tokenizer.decode([idx.item()]))} ({v.item():.2f})')

print('\nAt layer 20 (Grafted Identity):')
for idx, v in zip(top5_20.indices, top5_20.values):
    print(f'  {idx.item()}: {repr(tokenizer.decode([idx.item()]))} ({v.item():.2f})')
