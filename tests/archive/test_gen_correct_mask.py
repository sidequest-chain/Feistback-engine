import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask
from core.reversible_feistel import ReversibleFeistelNetwork

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
base_model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
base_model.eval()

feistel = ReversibleFeistelNetwork(num_layers=4, d_model=896, max_seq_len=512, layer_scale=0.0).to('cuda', dtype=torch.bfloat16)
feistel.eval()

def grafted_forward(input_ids):
    hidden_states = base_model.model.embed_tokens(input_ids)
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)

    mask_kwargs = {
        "config": base_model.config,
        "inputs_embeds": hidden_states,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }
    if base_model.model.has_sliding_layers:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base_model.model.rotary_emb(hidden_states, position_ids)

    for i in range(20):
        layer = base_model.model.layers[i]
        mask = causal_mask_mapping[base_model.config.layer_types[i]]
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
    hidden_states = base_model.model.norm(hidden_states)
    logits = base_model.lm_head(hidden_states)
    return logits

prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
encoded = tokenizer(prompt, return_tensors='pt')
input_ids = encoded['input_ids'].to('cuda')

curr_ids = input_ids.clone()
for step in range(30):
    logits = grafted_forward(curr_ids)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    curr_ids = torch.cat([curr_ids, next_token], dim=-1)
    if next_token.item() == tokenizer.eos_token_id:
        break

gen_text = tokenizer.decode(curr_ids[0, input_ids.shape[-1]:], skip_special_tokens=False)
print("Grafted with correct causal mask (skip_special_tokens=False):")
print(repr(gen_text))
print("\nReadable:")
print(prompt + gen_text)
