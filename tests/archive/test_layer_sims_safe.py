import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

out = model(**inputs, output_hidden_states=True)
hs = out.hidden_states

print("Length of hs:", len(hs))
norm_final = model.model.norm
lm_head = model.lm_head
logits_final = lm_head(norm_final(hs[-1]))

for i in range(len(hs)):
    l_i = lm_head(norm_final(hs[i]))
    sim = torch.cosine_similarity(l_i[:, -1, :].float(), logits_final[:, -1, :].float(), dim=-1).item()
    top_tok = torch.argmax(l_i[:, -1, :], dim=-1).item()
    tok_str = repr(tokenizer.decode([top_tok])).encode('ascii', 'backslashreplace').decode('ascii')
    print(f"Index {i} (Layer {i-1 if i>0 else 'embed'}) logits vs Final: {sim:.4f} | top token: {tok_str}")
