"""
Associative Memory Block (RAF: Reversible Associative Feistel)
==============================================================
Matrix-valued associative memory module with O(1) inference state footprint (0 bytes KV-cache)
and O(L^2) / O(L) fully parallelized causal training via log-space cumulative decay scans.

Theoretical Foundations:
  - Fast Weight Programmers / Linear Attention with learnable exponential decay.
  - State matrix M_t in R^{num_heads x key_dim x value_dim}.
  - Recurrent update:
      M_t = lambda_t * M_{t-1} + k_t^T @ v_t
      y_t = q_t @ M_t
  - Parallel causal equivalent via associative law (q_t @ k_tau^T) @ v_tau:
      A_{t, tau} = (q_t . k_tau) * exp(G_t - G_tau)  for tau <= t
      y_t = sum_{tau <= t} A_{t, tau} * v_tau
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociativeMemoryBlock(nn.Module):
    """
    Associative Memory Block for Reversible Associative Feistel (RAF) dynamics.
    Maintains a matrix state M_t per head to store key-value associations with strictly
    O(1) memory during inference, and supports exact parallel causal computation during training.
    """

    def __init__(
        self,
        d_model: int = 448,
        num_heads: int = 4,
        key_dim: int = 16,
        value_dim: int = 16,
        decay_bias_init: float = 2.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = 1.0 / math.sqrt(key_dim)

        # Input RMSNorm to stabilize linear attention variance across recursive Feistel rounds
        self.norm = nn.RMSNorm(d_model, eps=1e-6)

        # Projections
        self.W_q = nn.Linear(d_model, num_heads * key_dim, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * key_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * value_dim, bias=False)

        # Learnable decay gate per head with bias initialized towards long memory
        self.W_decay = nn.Linear(d_model, num_heads, bias=True)
        nn.init.zeros_(self.W_decay.weight)
        nn.init.constant_(self.W_decay.bias, decay_bias_init)

        # Output projection back to d_model
        self.W_out = nn.Linear(num_heads * value_dim, d_model, bias=False)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent forward pass.
        Maintains an O(1) state matrix M_t of shape (B, num_heads, key_dim, value_dim).

        Args:
            x: Input tensor of shape (B, 1, d_model) or (B, d_model).
            state: Optional state tensor of shape (B, num_heads, key_dim, value_dim).

        Returns:
            out: Output tensor of shape matching x (B, 1, d_model) or (B, d_model).
            new_state: Updated state tensor of shape (B, num_heads, key_dim, value_dim).
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(1)

        B, L, _ = x.shape
        assert L == 1, f"forward_recurrent expects L=1, got L={L}"

        # Normalize input to stabilize variance across rounds
        x_norm = self.norm(x)

        # Linear projections & head partitioning
        q = self.W_q(x_norm).view(B, 1, self.num_heads, self.key_dim).transpose(1, 2) * self.scale  # (B, H, 1, K)
        k = self.W_k(x_norm).view(B, 1, self.num_heads, self.key_dim).transpose(1, 2)              # (B, H, 1, K)
        v = self.W_v(x_norm).view(B, 1, self.num_heads, self.value_dim).transpose(1, 2)            # (B, H, 1, V)

        # L2 normalization over key_dim to preserve spectral stability of M_t
        k = F.normalize(k, p=2, dim=-1)

        # Decay gate: gamma_t in (0, 1)
        decay_raw = self.W_decay(x_norm).view(B, 1, self.num_heads).transpose(1, 2)  # (B, H, 1)
        gamma = torch.sigmoid(decay_raw.float()).to(dtype=x.dtype).unsqueeze(-1)  # (B, H, 1, 1)

        if state is None:
            state = torch.zeros(
                B, self.num_heads, self.key_dim, self.value_dim,
                device=x.device, dtype=x.dtype,
            )

        # Outer product update: M_t = gamma * M_{t-1} + k_t^T @ v_t
        kv = torch.matmul(k.transpose(-1, -2), v)  # (B, H, K, V)
        new_state = gamma * state + kv

        # Readout: y_t = q_t @ M_t
        y = torch.matmul(q, new_state)  # (B, H, 1, V)

        # Merge heads & project to d_model
        y = y.transpose(1, 2).contiguous().view(B, 1, self.num_heads * self.value_dim)
        out = self.W_out(y)  # (B, 1, d_model)

        if is_2d:
            out = out.squeeze(1)

        return out, new_state

    def forward_parallel(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parallel causal sequence forward pass across sequence length L without recurrence loops.
        Uses log-space cumulative decay scan for numerical equivalence to the recurrent formulation.

        Args:
            x: Input tensor of shape (B, L, d_model).
            state: Optional initial state matrix of shape (B, num_heads, key_dim, value_dim).
            return_state: If True, returns (out, final_state).

        Returns:
            out: Output tensor of shape (B, L, d_model).
            (optional) final_state: State matrix at step L of shape (B, num_heads, key_dim, value_dim).
        """
        B, L, _ = x.shape

        # Normalize input to stabilize variance across rounds
        x_norm = self.norm(x)

        # Projections & head partitioning
        q = self.W_q(x_norm).view(B, L, self.num_heads, self.key_dim).transpose(1, 2) * self.scale  # (B, H, L, K)
        k = self.W_k(x_norm).view(B, L, self.num_heads, self.key_dim).transpose(1, 2)              # (B, H, L, K)
        v = self.W_v(x_norm).view(B, L, self.num_heads, self.value_dim).transpose(1, 2)            # (B, H, L, V)

        # L2 normalization over key_dim
        k = F.normalize(k, p=2, dim=-1)

        # Log-space cumulative decay scan: g_t = log(sigmoid(z_t)) = -softplus(-z_t)
        # Computed in float32 to prevent cumulative precision drift and overflow
        decay_raw = self.W_decay(x_norm).view(B, L, self.num_heads).transpose(1, 2)  # (B, H, L)
        g = -F.softplus(-decay_raw.float())
        G = torch.cumsum(g, dim=-1)  # (B, H, L)

        # Pairwise decay weights: exp(G_t - G_tau) for tau <= t
        diff = G.unsqueeze(-1) - G.unsqueeze(-2)  # (B, H, L, L)
        causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        # Mask out upper triangle before exp to prevent exp(+large_pos) = inf -> nan gradients
        diff_masked = diff.masked_fill(~causal_mask, -1e4)
        decay_matrix = torch.exp(diff_masked).masked_fill(~causal_mask, 0.0).to(dtype=x.dtype)  # (B, H, L, L)

        # Linear attention: A = (Q @ K^T) * decay_matrix
        scores = torch.matmul(q, k.transpose(-1, -2))  # (B, H, L, L)
        A = scores * decay_matrix  # (B, H, L, L)
        y_attn = torch.matmul(A, v)  # (B, H, L, V)

        if state is not None:
            # Contribution from initial state M_0: (q_t @ M_0) * exp(G_t)
            q_M0 = torch.matmul(q, state)  # (B, H, L, V)
            decay_init = torch.exp(torch.clamp(G, max=0.0)).to(dtype=x.dtype).unsqueeze(-1)  # (B, H, L, 1)
            y = y_attn + q_M0 * decay_init
        else:
            y = y_attn

        # Merge heads & project to d_model
        y = y.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.value_dim)
        out = self.W_out(y)  # (B, L, d_model)

        if return_state:
            # Compute final state M_L in closed form:
            # M_L = sum_{tau=1}^L exp(G_L - G_tau) * (k_tau^T @ v_tau) + exp(G_L) * M_0
            diff_L = torch.clamp(G[:, :, -1:, None] - G[:, :, :, None], max=0.0)
            w_L = torch.exp(diff_L).to(dtype=x.dtype)  # (B, H, L, 1)
            k_weighted = k * w_L  # (B, H, L, K)
            M_new = torch.matmul(k_weighted.transpose(-1, -2), v)  # (B, H, K, V)
            if state is not None:
                decay_total = torch.exp(torch.clamp(G[:, :, -1:, None], max=0.0)).to(dtype=x.dtype)  # (B, H, 1, 1)
                final_state = decay_total * state + M_new
            else:
                final_state = M_new
            return out, final_state

        return out

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified forward interface. Automatically routes to recurrent mode for L=1
        and parallel mode for L > 1 while returning (out, state).
        """
        if x.shape[1] == 1:
            return self.forward_recurrent(x, state=state)
        else:
            return self.forward_parallel(x, state=state, return_state=True)
