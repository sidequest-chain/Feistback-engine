import torch
from core.qwen_grafted import GraftedQwenCoder

grafted = GraftedQwenCoder(layer_scale=0.0)
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
encoded = grafted.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(grafted.device)

curr_ids = input_ids.clone()
for step in range(10):
    logits = grafted.forward(curr_ids, mode="grafted")
    next_tok_logits = logits[0, -1, :]
    top5 = torch.topk(next_tok_logits, 5)
    print(f"\nStep {step}:")
    for id, val in zip(top5.indices, top5.values):
        tok_repr = repr(grafted.tokenizer.decode([id.item()])).encode('ascii', 'backslashreplace').decode('ascii')
        print(f"  {id.item()}: {tok_repr} ({val.item():.2f})")
    next_token = top5.indices[0].unsqueeze(0).unsqueeze(0)
    curr_ids = torch.cat([curr_ids, next_token], dim=-1)
