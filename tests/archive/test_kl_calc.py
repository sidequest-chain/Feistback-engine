import torch
import torch.nn.functional as F

B, L, K = 8, 256, 64
T = 2.0
s = torch.randn(B, L, K)
t = torch.randn(B, L, K)

s_flat = s.view(-1, K)
t_flat = t.view(-1, K)

kd = (T ** 2) * F.kl_div(
    F.log_softmax(s_flat / T, dim=-1),
    F.softmax(t_flat / T, dim=-1),
    reduction="batchmean",
)
print("Per-token batchmean KL:", kd.item())
