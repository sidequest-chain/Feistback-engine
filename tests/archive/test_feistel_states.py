import torch
from core.reversible_feistel import ReversibleFeistelNetwork
from core.holographic_mixer import functional_holographic_mixer

f = ReversibleFeistelNetwork(num_layers=12, d_model=896, max_seq_len=512, layer_scale=0.05).to('cuda', dtype=torch.bfloat16)
x = torch.randn(2, 16, 896, device='cuda', dtype=torch.bfloat16)

# Standard forward
y_standard = f(x)

# Step by step intermediate
L, R = torch.chunk(x, 2, dim=-1)
mixer_callable = lambda r, w: functional_holographic_mixer(r, w, activation=f.activation)

states = [x]
with torch.no_grad():
    for i in range(f.num_layers):
        W_i = f.weights[i]
        L_next = R
        R_next = L + f.layer_scale * mixer_callable(R, W_i)
        L, R = L_next, R_next
        states.append(torch.cat([L, R], dim=-1))

y_step = states[-1]
diff = (y_standard - y_step).abs().max().item()
print("Diff between standard forward and step-by-step intermediate:", diff)
print("Number of intermediate states:", len(states))
