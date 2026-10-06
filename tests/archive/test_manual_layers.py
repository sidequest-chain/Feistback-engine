import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer
from core.qwen_grafted import GraftedQwenCoder

model = Qwen2ForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', torch_dtype=torch.bfloat16, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')

# 1. Full base forward
with torch.no_grad():
    out_base = model(input_ids=inputs['input_ids'], use_cache=False)
    logits_base = out_base.logits

# 2. Compare with our manual loop through all 24 layers of model.model.layers!
hidden_states = model.model.embed_tokens(inputs['input_ids'])
batch_size, seq_len = inputs['input_ids'].shape
position_ids = torch.arange(seq_len, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)

causal_mask = model.model._update_causal_mask(None, hidden_states, position_ids, None, False)
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

with torch.no_grad():
    for layer in model.model.layers:
        hidden_states = layer(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states[0]

    hidden_states = model.model.norm(hidden_states)
    logits_manual_24 = model.lm_head(hidden_states)

diff = (logits_base - logits_manual_24).abs().max().item()
print("Diff between base_model forward and manual 24-layer loop:", diff)
