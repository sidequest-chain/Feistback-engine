import torch
import torch.nn.functional as F

device = "cuda"
B, L, V = 8, 256, 151936
T = 2.0
K = 64

student_logits = torch.randn(B, L, V, device=device, dtype=torch.bfloat16, requires_grad=True)
teacher_logits = torch.randn(B, L, V, device=device, dtype=torch.bfloat16)

# Warmup
top_teacher_logits, top_indices = torch.topk(teacher_logits, k=K, dim=-1)
top_student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
loss_kd = (T ** 2) * F.kl_div(
    F.log_softmax(top_student_logits / T, dim=-1),
    F.softmax(top_teacher_logits / T, dim=-1),
    reduction="batchmean",
)
loss_kd.backward()

torch.cuda.reset_peak_memory_stats()
t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)

t0.record()
top_teacher_logits, top_indices = torch.topk(teacher_logits, k=K, dim=-1)
top_student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
loss_kd = (T ** 2) * F.kl_div(
    F.log_softmax(top_student_logits / T, dim=-1),
    F.softmax(top_teacher_logits / T, dim=-1),
    reduction="batchmean",
)
loss_kd.backward()
t1.record()
torch.cuda.synchronize()

print(f"Top-{K} KL-div step time: {t0.elapsed_time(t1):.2f} ms")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
