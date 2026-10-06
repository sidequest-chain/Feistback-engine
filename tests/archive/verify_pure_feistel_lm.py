import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import torch
from core.pure_feistel_lm import PureFeistelLM

print("Initializing PureFeistelLM (from Qwen2.5-Coder embeddings & lm_head, 12 Feistel layers)...")
t0 = time.perf_counter()
model = PureFeistelLM(
    d_model=896,
    num_layers=12,
    max_seq_len=512,
    layer_scale=0.05,
    torch_dtype=torch.bfloat16,
    from_pretrained_qwen=True,
)
print(f"Loaded PureFeistelLM in {time.perf_counter() - t0:.2f}s")

# 1. Parameter Summary
summary = model.get_parameter_summary()
print("\n" + "=" * 60)
print("PURE FEISTEL LM ARCHITECTURAL SUMMARY")
print("=" * 60)
for k, v in summary.items():
    print(f"  {k:<25}: {v}")

# 2. Test forward pass
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
encoded = model.tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(model.device)

logits = model.forward(input_ids)
print("\nForward pass successful!")
print(f"Input shape: {input_ids.shape}, Output logits shape: {logits.shape}")

# 3. Test get_hidden_states
hidden_states = model.get_hidden_states(input_ids)
print(f"Extracted {len(hidden_states)} intermediate states (input + {model.num_layers} layers)")
for i, h in enumerate(hidden_states):
    print(f"  State {i}: shape={h.shape}, norm={h.norm(dim=-1).mean().item():.2f}")

# 4. Profile Forward Latency & Throughput
print("\nProfiling forward pass throughput (100 warmup, 100 repeats)...")
B, L = 1, 64
dummy_ids = torch.randint(0, 1000, (B, L), device=model.device)

with torch.no_grad():
    for _ in range(20):
        _ = model.forward(dummy_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    num_runs = 100
    for _ in range(num_runs):
        _ = model.forward(dummy_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start

avg_latency_ms = (total_time / num_runs) * 1000.0
tokens_per_sec = (num_runs * L) / total_time
print(f"Forward Latency (L={L}): {avg_latency_ms:.2f} ms")
print(f"Throughput: {tokens_per_sec:.1f} tokens/second")

# 5. Test generation
print("\nTesting autoregressive generation (max_new_tokens=20)...")
gen_text, gen_ids = model.generate(prompt, max_new_tokens=20)
print(f"Generated text: {repr(gen_text)}")
print("=" * 60)
