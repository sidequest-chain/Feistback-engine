import torch
from core.qwen_grafted import GraftedQwenCoder

print("Initializing GraftedQwenCoder...")
model = GraftedQwenCoder()

prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)

print("Testing baseline forward...")
logits_base = model(input_ids, mode="baseline")
print("Baseline logits shape:", logits_base.shape)

print("Testing grafted forward...")
logits_graft = model(input_ids, mode="grafted")
print("Grafted logits shape:", logits_graft.shape)

cos_sim = torch.cosine_similarity(logits_base[:, -1, :], logits_graft[:, -1, :], dim=-1).item()
print("Logits Cosine Similarity at last token:", cos_sim)

print("\nGenerating with baseline...")
text_base, _ = model.generate(prompt, max_new_tokens=25, mode="baseline")
print("Baseline generated:\n", text_base)

print("\nGenerating with grafted...")
text_graft, _ = model.generate(prompt, max_new_tokens=25, mode="grafted")
print("Grafted generated:\n", text_graft)
