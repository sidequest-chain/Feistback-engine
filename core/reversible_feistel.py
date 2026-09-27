"""
Reversible Feistel Network
==========================
ReversibleFeistelBlock executes bit-accurate reversible Feistel network dynamics:
  Forward pass (i = 0 ... N-1):
      L_{i+1} = R_i
      R_{i+1} = L_i + scale * F(R_i, W_i)

  Backward pass (i = N-1 ... 0):
      R_i = L_{i+1}
      L_i = R_{i+1} - scale * F(L_{i+1}, W_i)

Zero-Activation Memory Invariant:
  - ctx.save_for_backward saves ONLY the final layer output Y and parameter tensor W.
  - Intermediate activations are NOT cached.
  - During backpropagation, input activations for each layer are reconstructed mathematically,
    achieving O(1) activation memory regardless of network depth.
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn

from core.holographic_mixer import functional_holographic_mixer, compute_param_len


class ReversibleFeistelBlock(torch.autograd.Function):
    """
    Custom PyTorch autograd function for multi-layer reversible Feistel network.
    Guarantees:
      - O(1) activation memory with respect to layer depth.
      - Exact mathematical reconstruction on backward pass.
      - Compatible with both single-layer and N-layer stacks.
    """
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weights: torch.Tensor,
        mixer_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = functional_holographic_mixer,
        layer_scale: float = 0.05
    ) -> torch.Tensor:
        """
        Forward pass without caching intermediate activations.

        Args:
          x: Input tensor of shape (B, L, D). Split into (L_0, R_0) along dim=-1.
          weights: Parameter tensor of shape (N, param_dim) or (param_dim,) if N=1.
          mixer_fn: Layer function of form F(R, W_i) -> output.
          layer_scale: Scaling factor for numerical stability.
        """
        is_single_layer = (weights.dim() == 1)
        w_tensor = weights.unsqueeze(0) if is_single_layer else weights
        num_layers = w_tensor.shape[0]

        ctx.mixer_fn = mixer_fn
        ctx.is_single_layer = is_single_layer
        ctx.num_layers = num_layers
        ctx.layer_scale = layer_scale

        # Split input into two branches: L_0 and R_0 (both shape B, L, D/2)
        L, R = torch.chunk(x, 2, dim=-1)

        # Forward through all N layers without retaining autograd graph
        with torch.no_grad():
            for i in range(num_layers):
                W_i = w_tensor[i]
                L_next = R
                # Feistel step: R_{i+1} = L_i + scale * F(R_i, W_i)
                R_next = L + layer_scale * mixer_fn(R, W_i)
                L, R = L_next, R_next

        # Output Y is final layer state: Y = [L_N, R_N]
        y = torch.cat([L, R], dim=-1)

        # Save only final layer output and weight parameters
        ctx.save_for_backward(y, weights)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass: reconstructs intermediate layer states in reverse order and
        computes exact gradients for weights and inputs.
        """
        y, weights = ctx.saved_tensors
        is_single_layer = ctx.is_single_layer
        num_layers = ctx.num_layers
        mixer_fn = ctx.mixer_fn
        layer_scale = ctx.layer_scale

        w_tensor = weights.unsqueeze(0) if is_single_layer else weights

        L, R = torch.chunk(y, 2, dim=-1)
        grad_L, grad_R = torch.chunk(grad_output, 2, dim=-1)

        grad_weights_list = [None] * num_layers

        # Reconstruct layer-by-layer backwards: N-1 ... 0
        for i in reversed(range(num_layers)):
            W_i = w_tensor[i]

            # 1. Feistel inverse:
            #    R_{i-1} = L_i
            #    L_{i-1} = R_i - scale * F(R_{i-1}, W_i)
            R_prev = L
            with torch.no_grad():
                F_val = mixer_fn(R_prev, W_i)
            L_prev = R - layer_scale * F_val

            # 2. Compute local Vector-Jacobian Product (VJP) for layer i
            with torch.enable_grad():
                R_in = R_prev.detach().requires_grad_(True)
                W_in = W_i.detach().requires_grad_(True)
                F_out = mixer_fn(R_in, W_in)

                dF_dR, dF_dW = torch.autograd.grad(
                    outputs=F_out,
                    inputs=(R_in, W_in),
                    grad_outputs=(layer_scale * grad_R),
                    retain_graph=False
                )

            # 3. Gradient accumulation via Feistel chain rule:
            #    grad_L_{i-1} = grad_R_i
            #    grad_R_{i-1} = grad_L_i + dF_dR
            grad_L_prev = grad_R
            grad_R_prev = grad_L + dF_dR
            grad_weights_list[i] = dF_dW

            L, R = L_prev, R_prev
            grad_L, grad_R = grad_L_prev, grad_R_prev

        # Reconstructed initial input X_0 gradients
        grad_x = torch.cat([grad_L, grad_R], dim=-1)

        if is_single_layer:
            grad_weights = grad_weights_list[0]
        else:
            grad_weights = torch.stack(grad_weights_list, dim=0)

        return grad_x, grad_weights, None, None


class ReversibleFeistelBlockModule(nn.Module):
    """
    nn.Module wrapper for a single reversible Feistel layer.
    """
    def __init__(
        self,
        d_half: int,
        max_seq_len: int = 128,
        activation: str = "silu",
        layer_scale: float = 0.05
    ):
        super().__init__()
        self.d_half = d_half
        self.max_seq_len = max_seq_len
        self.activation = activation
        self.layer_scale = layer_scale
        self.param_len = compute_param_len(max_seq_len, d_half)

        init_weight = torch.randn(self.param_len) * 0.02
        self.weight = nn.Parameter(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixer_callable = lambda r, w: functional_holographic_mixer(r, w, activation=self.activation)
        return ReversibleFeistelBlock.apply(x, self.weight, mixer_callable, self.layer_scale)


class ReversibleFeistelNetwork(nn.Module):
    """
    Full N-layer reversible Feistel network with O(1) activation memory.
    All N layers run under a single ReversibleFeistelBlock autograd function.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        max_seq_len: int = 128,
        activation: str = "silu",
        layer_scale: float = 0.05
    ):
        super().__init__()
        assert d_model % 2 == 0, f"d_model ({d_model}) must be divisible by 2 for Feistel branches!"
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_half = d_model // 2
        self.max_seq_len = max_seq_len
        self.activation = activation
        self.layer_scale = layer_scale

        self.param_len_per_layer = compute_param_len(max_seq_len, self.d_half)
        weights_init = torch.randn(num_layers, self.param_len_per_layer) * 0.02
        self.weights = nn.Parameter(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all N layers with O(1) activation memory.
        """
        mixer_callable = lambda r, w: functional_holographic_mixer(r, w, activation=self.activation)
        return ReversibleFeistelBlock.apply(x, self.weights, mixer_callable, self.layer_scale)

    def verify_reversibility_step_by_step(
        self,
        x: torch.Tensor
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Validates bit-accurate reconstructibility of input x.

        Returns:
          max_error: Maximum absolute difference between original and reconstructed input.
          layer_errors: List of per-layer reconstruction errors (layer_idx, error).
        """
        L_cur, R_cur = torch.chunk(x.detach(), 2, dim=-1)
        forward_states = [(L_cur.clone(), R_cur.clone())]

        mixer_callable = lambda r, w: functional_holographic_mixer(r, w, activation=self.activation)
        with torch.no_grad():
            for i in range(self.num_layers):
                W_i = self.weights[i]
                L_next = R_cur
                R_next = L_cur + self.layer_scale * mixer_callable(R_cur, W_i)
                L_cur, R_cur = L_next, R_next
                forward_states.append((L_cur.clone(), R_cur.clone()))

        layer_errors = []
        L_rec, R_rec = L_cur, R_cur

        for i in reversed(range(self.num_layers)):
            W_i = self.weights[i]
            L_prev = R_rec - self.layer_scale * mixer_callable(L_rec, W_i)
            R_prev = L_rec

            orig_L, orig_R = forward_states[i]
            err_L = torch.max(torch.abs(L_prev - orig_L)).item()
            err_R = torch.max(torch.abs(R_prev - orig_R)).item()
            layer_max_err = max(err_L, err_R)
            layer_errors.append((i, layer_max_err))

            L_rec, R_rec = L_prev, R_prev

        orig_L0, orig_R0 = forward_states[0]
        max_error = max(
            torch.max(torch.abs(L_rec - orig_L0)).item(),
            torch.max(torch.abs(R_rec - orig_R0)).item()
        )

        return max_error, layer_errors
