import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')

# Prompt + '    if isinstance('
p = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n    if isinstance('
inputs = tokenizer(p, return_tensors='pt').to('cuda')
out = model(**inputs, output_hidden_states=True)

hs_20 = out.hidden_states[20]
logits_20 = model.lm_head(model.model.norm(hs_20))[0, -1, :]

# Filter out special/FIM tokens
logits_20[151643:] = -float('inf')

top10 = torch.topk(logits_20, 20)
print('Top 20 tokens at layer 20 after \"    if isinstance(\":')
for idx, v in zip(top10.indices, top10.values):
    s = repr(tokenizer.decode([idx.item()])).encode('ascii', 'backslashreplace').decode('ascii')
    print(f'  {idx.item()}: {s} ({v.item():.2f})')
