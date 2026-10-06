import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask
from core.reversible_feistel import ReversibleFeistelNetwork

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
feistel = ReversibleFeistelNetwork(num_layers=4, d_model=896, max_seq_len=512, layer_scale=0.0).to('cuda', dtype=torch.bfloat16)

prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to('cuda')

def forward_grafted(ids):
    hidden = model.model.embed_tokens(ids)
    batch_size, seq_len = ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device='cuda').unsqueeze(0)
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    for i in range(20):
        hidden = model.model.layers[i](
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    hidden = feistel(hidden)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)

# Generate 35 tokens greedily
curr_ids = input_ids.clone()
with torch.no_grad():
    for _ in range(35):
        logits = forward_grafted(curr_ids)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

gen_text = tokenizer.decode(curr_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
print("Generated text with proper causal mask and layer_scale=0.0:\n")
print(prompt + gen_text)
