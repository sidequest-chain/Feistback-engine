import torch
import torch.nn.functional as F

device = "cuda"
B, L, V = 8, 256, 151936
T = 2.0

student_logits = torch.randn(B, L, V, device=device, dtype=torch.bfloat16, requires_grad=True)
teacher_logits = torch.randn(B, L, V, device=device, dtype=torch.bfloat16)

t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)

t0.record()
# Test full KL-div
student_log_probs = F.log_softmax(student_logits / T, dim=-1)
teacher_probs = F.softmax(teacher_logits / T, dim=-1)
loss_kd = (T ** 2) * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
loss_kd.backward()
t1.record()
torch.cuda.synchronize()

print(f"Full KL-div step time: {t0.elapsed_time(t1):.2f} ms")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
