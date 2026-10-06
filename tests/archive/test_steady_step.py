import time
import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from core.pure_feistel_lm import PureFeistelLM

device = "cuda"
teacher = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16, device_map=device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = PureFeistelLM(
    d_model=896,
    num_layers=12,
    torch_dtype=torch.bfloat16,
    device=device,
    from_pretrained_qwen=False,
    embed_tokens=teacher.model.embed_tokens,
    norm=teacher.model.norm,
    lm_head=teacher.lm_head,
)
student.embed_tokens.requires_grad_(False)
student.norm.requires_grad_(False)
student.lm_head.requires_grad_(False)
student.backbone.requires_grad_(True)

optimizer = torch.optim.AdamW(student.backbone.parameters(), lr=6e-4, weight_decay=0.01)

batch = torch.randint(0, 1000, (8, 256), device=device)
T = 2.0
K = 64

# Warmup 2 steps
for _ in range(2):
    with torch.no_grad():
        teacher_hs = teacher.model(input_ids=batch).last_hidden_state
        teacher_logits = teacher.lm_head(teacher_hs)
    student_hs = student.forward_backbone(batch, with_norm=True)
    student_logits = student.lm_head(student_hs)
    loss_mse = F.mse_loss(student_hs, teacher_hs)
    cos_sim = F.cosine_similarity(student_hs, teacher_hs, dim=-1).mean()
    loss_rep = loss_mse + 0.5 * (1.0 - cos_sim)
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()
    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    top_teacher_logits, top_indices = torch.topk(teacher_logits, k=K, dim=-1)
    top_student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
    loss_kd = (T ** 2) * F.kl_div(
        F.log_softmax(top_student_logits.view(-1, K) / T, dim=-1),
        F.softmax(top_teacher_logits.view(-1, K) / T, dim=-1),
        reduction="batchmean",
    )
    loss_total = 0.1 * loss_rep + 0.5 * loss_ce + 1.0 * loss_kd
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.no_grad():
        teacher_hs = teacher.model(input_ids=batch).last_hidden_state
        teacher_logits = teacher.lm_head(teacher_hs)
    student_hs = student.forward_backbone(batch, with_norm=True)
    student_logits = student.lm_head(student_hs)
    loss_mse = F.mse_loss(student_hs, teacher_hs)
    cos_sim = F.cosine_similarity(student_hs, teacher_hs, dim=-1).mean()
    loss_rep = loss_mse + 0.5 * (1.0 - cos_sim)
    shift_logits = student_logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()
    loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    top_teacher_logits, top_indices = torch.topk(teacher_logits, k=K, dim=-1)
    top_student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
    loss_kd = (T ** 2) * F.kl_div(
        F.log_softmax(top_student_logits.view(-1, K) / T, dim=-1),
        F.softmax(top_teacher_logits.view(-1, K) / T, dim=-1),
        reduction="batchmean",
    )
    loss_total = 0.1 * loss_rep + 0.5 * loss_ce + 1.0 * loss_kd
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 10
print(f"Steady-state step time: {dt * 1000:.2f} ms ({8 * 256 / dt:.0f} tokens/sec)")
