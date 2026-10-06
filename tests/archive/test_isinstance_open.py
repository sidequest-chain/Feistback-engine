import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')

p = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if isinstance('
inputs = tokenizer(p, return_tensors='pt').to('cuda')
out = model(**inputs, output_hidden_states=True)

logits_24 = out.logits[0, -1, :]
hs_20 = out.hidden_states[20]
logits_20 = model.lm_head(model.model.norm(hs_20))[0, -1, :]

top10_24 = torch.topk(logits_24, 10)
top10_20 = torch.topk(logits_20, 10)

print('Layer 24 top 10:')
for idx, v in zip(top10_24.indices, top10_24.values):
    print(f'  {idx.item()}: {repr(tokenizer.decode([idx.item()]))} ({v.item():.2f})')

print('\nLayer 20 top 10:')
for idx, v in zip(top10_20.indices, top10_20.values):
    print(f'  {idx.item()}: {repr(tokenizer.decode([idx.item()]))} ({v.item():.2f})')
