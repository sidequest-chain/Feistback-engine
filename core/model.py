"""
Model Architecture: Feistel Holographic Language Model (FeistelHolographicLM)
=============================================================================
Combines:
  - Token embedding + learnable positional encoding
  - N-layer ReversibleFeistelNetwork with FFT frequency-domain circular convolution
  - Final LayerNorm
  - LM prediction head for autoregressive token prediction

Activation memory across layer depth N is O(1).
Quadratic attention complexity O(L^2) is replaced with O(L log L) frequency-domain mixing.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.reversible_feistel import ReversibleFeistelNetwork


class FeistelHolographicLM(nn.Module):
    """
    Complete language model without quadratic attention matrices and with O(1) activation memory.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 16,
        max_seq_len: int = 128,
        activation: str = "silu",
        layer_scale: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.layer_scale = layer_scale

        # Token and positional representations
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Reversible Feistel frequency-domain backbone (O(1) activation memory)
        self.feistel_backbone = ReversibleFeistelNetwork(
            num_layers=num_layers,
            d_model=d_model,
            max_seq_len=max_seq_len,
            activation=activation,
            layer_scale=layer_scale
        )

        # Final LayerNorm and vocabulary projection head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Language model forward pass.

        Args:
          input_ids: (B, L) integer token IDs tensor.
          targets: (B, L) target token IDs tensor (optional).

        Returns:
          logits: (B, L, vocab_size)
          loss: scalar CrossEntropy loss if targets provided, else None.
        """
        B, L = input_ids.shape
        assert L <= self.max_seq_len, f"Sequence length L={L} exceeds max_seq_len={self.max_seq_len}!"

        # Combine embeddings
        h = self.token_embed(input_ids) + self.pos_embed[:, :L, :]

        # Reversible backbone
        h = self.feistel_backbone(h)

        # Normalization and logits
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            # Padding token is ignored (index 0 = <pad>)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=0
            )

        return logits, loss

    def compute_reversal_error(self, input_ids: torch.Tensor) -> float:
        """
        Computes maximum numerical reconstruction error across all layers.
        """
        B, L = input_ids.shape
        with torch.no_grad():
            h = self.token_embed(input_ids) + self.pos_embed[:, :L, :]
            max_err, _ = self.feistel_backbone.verify_reversibility_step_by_step(h)
        return max_err


# Backward compatibility alias
ReversibleHolographicTransformer = FeistelHolographicLM
