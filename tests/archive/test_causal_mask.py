import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')

prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to('cuda')

# Baseline 24 layers
with torch.no_grad():
    out_base = model(input_ids)
    logits_base = out_base.logits

# 20 layers with proper causal mask
inputs_embeds = model.model.embed_tokens(input_ids)
position_ids = torch.arange(inputs_embeds.shape[1], device='cuda').unsqueeze(0)
mask_kwargs = {
    'config': model.config,
    'inputs_embeds': inputs_embeds,
    'attention_mask': None,
    'past_key_values': None,
    'position_ids': position_ids,
}
causal_mask = create_causal_mask(**mask_kwargs)
position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)

hidden = inputs_embeds
with torch.no_grad():
    for i in range(20):
        hidden = model.model.layers[i](
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    hidden_norm = model.model.norm(hidden)
    logits_20 = model.lm_head(hidden_norm)

cos_sim = torch.cosine_similarity(logits_base[:, -1, :].float(), logits_20[:, -1, :].float(), dim=-1).item()
print('Cosine similarity with CAUSAL mask at last token:', cos_sim)
top_base = torch.topk(logits_base[:, -1, :], 5).indices[0].tolist()
top_20 = torch.topk(logits_20[:, -1, :], 5).indices[0].tolist()
print('Top 5 tokens base:', [tokenizer.decode([t]) for t in top_base])
print('Top 5 tokens 20 layers:', [tokenizer.decode([t]) for t in top_20])
