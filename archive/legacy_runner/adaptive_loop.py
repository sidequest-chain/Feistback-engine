"""
Adaptive Latent Deliberation Runner with Early-Exit.
Implements dynamic computational depth based on Poincaré attractor convergence (delta < epsilon).
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from core.reversible_feistel import ReversibleFeistelNetwork


class AdaptiveFeistelRunner(nn.Module):
    """
    Runner that iteratively applies a reversible Feistel block to hidden states
    with early-exit when the state change falls below a convergence threshold epsilon.
    """

    def __init__(
        self,
        feistel_block: Optional[nn.Module] = None,
        d_model: int = 128,
        max_seq_len: int = 128,
        default_max_steps: int = 8,
        default_epsilon: float = 0.01,
    ):
        super().__init__()
        if feistel_block is not None:
            self.block = feistel_block
        else:
            self.block = ReversibleFeistelNetwork(
                num_layers=2,
                d_model=d_model,
                max_seq_len=max_seq_len,
                layer_scale=0.05,
            )
        self.default_max_steps = default_max_steps
        self.default_epsilon = default_epsilon

    def forward_static(
        self, z: torch.Tensor, steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Executes a fixed number of recurrent iterations without early exit.
        """
        num_steps = steps or self.default_max_steps
        current_z = z
        for _ in range(num_steps):
            current_z = self.block(current_z)
        return current_z

    def forward_adaptive(
        self,
        z: torch.Tensor,
        max_steps: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Executes recurrent unrolling with early exit based on ||z_{t+1} - z_t||_2.
        Returns:
            final_z: Resulting hidden state tensor.
            steps_taken: Number of iterations executed before convergence or timeout.
            deltas: List of per-step delta norms.
        """
        limit = max_steps or self.default_max_steps
        eps = epsilon or self.default_epsilon

        current_z = z
        deltas: List[float] = []
        steps_taken = 0

        for t in range(limit):
            next_z = self.block(current_z)
            delta = torch.norm(next_z - current_z, dim=-1).mean().item()
            deltas.append(delta)
            current_z = next_z
            steps_taken = t + 1

            if delta < eps:
                break

        return current_z, steps_taken, deltas
