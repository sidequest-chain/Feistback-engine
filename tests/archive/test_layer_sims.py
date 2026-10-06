import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

out = model(**inputs, output_hidden_states=True)
hs = out.hidden_states # tuple of 25 tensors: embed + 24 layers

print("Cosine similarity of hidden states at last token:")
for i in range(len(hs) - 1):
    sim = torch.cosine_similarity(hs[i][:, -1, :].float(), hs[i+1][:, -1, :].float(), dim=-1).item()
    print(f"Layer {i} -> {i+1}: {sim:.4f}")

print("\nCosine similarity of logits (norm + lm_head) at last token across layers:")
norm_final = model.model.norm
lm_head = model.lm_head
logits_final = lm_head(norm_final(hs[-1]))

for i in range(len(hs)):
    l_i = lm_head(norm_final(hs[i]))
    sim = torch.cosine_similarity(l_i[:, -1, :].float(), logits_final[:, -1, :].float(), dim=-1).item()
    top_tok = torch.argmax(l_i[:, -1, :], dim=-1).item()
    print(f"Layer {i} logits vs Final (Layer 24): {sim:.4f} | top token: {repr(tokenizer.decode([top_tok]))}")
