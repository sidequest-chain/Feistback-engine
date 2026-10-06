import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask
from core.reversible_feistel import ReversibleFeistelNetwork

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
feistel = ReversibleFeistelNetwork(num_layers=4, d_model=896, max_seq_len=512, layer_scale=0.0).to('cuda', dtype=torch.bfloat16)

def forward_grafted(input_ids):
    hidden_states = model.model.embed_tokens(input_ids)
    batch_size, seq_len = input_ids.shape
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
    if getattr(model.model, "has_sliding_layers", False):
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    for i in range(20):
        layer = model.model.layers[i]
        layer_type = model.config.layer_types[i] if hasattr(model.config, "layer_types") else "full_attention"
        mask = causal_mask_mapping.get(layer_type, causal_mask_mapping["full_attention"])
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states[0]

    hidden_states = feistel(hidden_states)
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits

prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
curr = inputs['input_ids'].clone()

# Let's inspect generation with:
# 1. Banning FIM tokens
fim_ids = [151659, 151660, 151661, 151662, 151663, 151664]

for step in range(35):
    logits = forward_grafted(curr)
    next_logits = logits[0, -1, :].clone()
    for fid in fim_ids:
        next_logits[fid] = -float("inf")
    
    # What if we also check top 5 tokens at each step?
    top_tok = torch.argmax(next_logits, dim=-1, keepdim=True).unsqueeze(0)
    tok_id = top_tok.item()
    curr = torch.cat([curr, top_tok], dim=-1)
    if tok_id == tokenizer.eos_token_id:
        break

print("Result with proper causal mask and FIM suppression:")
print(prompt + tokenizer.decode(curr[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True))
