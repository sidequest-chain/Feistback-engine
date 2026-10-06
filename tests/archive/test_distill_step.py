import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM
from core.pure_feistel_lm import PureFeistelLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Load teacher
teacher = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B",
    torch_dtype=torch.bfloat16,
    device_map=device,
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

# 2. Load student
student = PureFeistelLM(
    d_model=896,
    num_layers=12,
    torch_dtype=torch.bfloat16,
    device=device,
)
student.embed_tokens.requires_grad_(False)
student.norm.requires_grad_(False)
student.lm_head.requires_grad_(False)
student.feistel_backbone.requires_grad_(True)

trainable_params = sum(p.numel() for p in student.feistel_backbone.parameters() if p.requires_grad)
print(f"Trainable student backbone parameters: {trainable_params / 1e6:.2f}M ({trainable_params})")

# 3. Test forward pass
dummy_input = torch.randint(0, 1000, (4, 64), device=device)

with torch.no_grad():
    teacher_hs = teacher.model(dummy_input).last_hidden_state

x = student.embed_tokens(dummy_input)
h = student.feistel_backbone(x)
student_hs = student.norm(h)

mse = F.mse_loss(student_hs, teacher_hs)
cos_sim = F.cosine_similarity(student_hs, teacher_hs, dim=-1).mean()
loss = mse + 0.5 * (1.0 - cos_sim)

print(f"Initial loss: {loss.item():.4f}, mse: {mse.item():.4f}, cos_sim: {cos_sim.item():.4f}")

optimizer = torch.optim.AdamW(student.feistel_backbone.parameters(), lr=1e-3, weight_decay=0.01)
loss.backward()

print("Backbone grad is None?", student.feistel_backbone.weights.grad is None)
if student.feistel_backbone.weights.grad is not None:
    print(f"Backbone grad norm: {student.feistel_backbone.weights.grad.norm().item():.4f}")

optimizer.step()
optimizer.zero_grad()
print("Step successful!")
