import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# 1. Full base forward
with torch.no_grad():
    out_base = model(input_ids=inputs['input_ids'], use_cache=False)
    logits_base = out_base.logits

# 2. Retain 20 layers with correct causal mask
hidden_states = model.model.embed_tokens(inputs['input_ids'])
batch_size, seq_len = inputs['input_ids'].shape
position_ids = torch.arange(seq_len, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)

mask_kwargs = {
    "config": model.config,
    "inputs_embeds": hidden_states,
    "attention_mask": None,
    "past_key_values": None,
    "position_ids": position_ids,
}
causal_mask_mapping = {
    "full_attention": create_causal_mask(**mask_kwargs),
}
if model.model.has_sliding_layers:
    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

with torch.no_grad():
    for i in range(20):
        layer = model.model.layers[i]
        mask = causal_mask_mapping[model.config.layer_types[i]]
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states[0]

    hidden_states = model.model.norm(hidden_states)
    logits_20 = model.lm_head(hidden_states)

cos_sim_raw = torch.cosine_similarity(logits_base[:, -1, :].float(), logits_20[:, -1, :].float(), dim=-1).item()
probs_base = torch.softmax(logits_base[:, -1, :].float(), dim=-1)
probs_20 = torch.softmax(logits_20[:, -1, :].float(), dim=-1)
cos_sim_probs = torch.cosine_similarity(probs_base, probs_20, dim=-1).item()

print(f"Logits raw cosine similarity: {cos_sim_raw:.4f}")
print(f"Probabilities cosine similarity: {cos_sim_probs:.4f}")
print("Top token base:", repr(tokenizer.decode([torch.argmax(logits_base[:, -1, :]).item()])))
print("Top token layer 20:", repr(tokenizer.decode([torch.argmax(logits_20[:, -1, :]).item()])))
