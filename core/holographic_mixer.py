"""
Holographic Mixer
=================
Performs frequency-domain circular convolution (FFT-based associative frequency memory),
replacing the O(N^2) Transformer attention matrix (QK^T) with O(L log L) circular convolution.

Formulas:
  1. Causal context mixing along the sequence dimension (dim=1, O(L log L)):
     R_seq = irfft(rfft(pad(R)) * rfft(pad(W_seq)))[:, :L]
  2. Holographic binding along the hidden dimension (dim=-1, O(D log D)):
     R_chan = irfft(rfft(R_seq) * rfft(W_chan))
  3. Non-linear projection with gradient-preserving activation (SiLU / scaled Tanh):
     F(R) = SiLU(R_chan @ W_proj + b)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def holographic_circular_conv_1d(
    x: torch.Tensor,
    filter_weight: torch.Tensor,
    dim: int = -1,
    n: Optional[int] = None,
    causal: bool = False
) -> torch.Tensor:
    """
    Computes 1D circular convolution in frequency domain via the Circular Convolution Theorem:
        conv(x, w) = irfft(rfft(x) * rfft(w))

    Args:
        x: Input tensor.
        filter_weight: 1D filter weights in time domain.
        dim: Axis along which circular convolution is performed.
        n: FFT length (default is x.shape[dim]).
        causal: If True, pads sequence to 2*n to ensure strict causality.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    signal_len = x.shape[dim] if n is None else n
    orig_dtype = x.dtype

    if causal:
        # Causal FFT convolution via zero padding
        if dim == 1:
            x_padded = F.pad(x, (0, 0, 0, signal_len))
        elif dim == -1 or dim == x.ndim - 1:
            x_padded = F.pad(x, (0, signal_len))
        else:
            raise NotImplementedError(f"Causal padding for dim={dim} is not supported!")

        fft_len = 2 * signal_len
        x_freq = torch.fft.rfft(x_padded, n=fft_len, dim=dim)
        w_freq = torch.fft.rfft(filter_weight, n=fft_len, dim=0)

        shape = [1] * x_padded.ndim
        shape[dim] = w_freq.shape[0]
        w_freq = w_freq.view(*shape)

        out_freq = x_freq * w_freq
        out = torch.fft.irfft(out_freq, n=fft_len, dim=dim)

        if dim == 1:
            out = out[:, :signal_len, :]
        elif dim == -1 or dim == x.ndim - 1:
            out = out[..., :signal_len]
    else:
        # Standard frequency-domain circular convolution
        fft_len = signal_len
        x_freq = torch.fft.rfft(x, n=fft_len, dim=dim)
        w_freq = torch.fft.rfft(filter_weight, n=fft_len, dim=0)

        shape = [1] * x.ndim
        shape[dim] = w_freq.shape[0]
        w_freq = w_freq.view(*shape)

        out_freq = x_freq * w_freq
        out = torch.fft.irfft(out_freq, n=fft_len, dim=dim)

    # Ensure returned tensor preserves original precision
    if out.dtype != orig_dtype:
        out = out.to(dtype=orig_dtype)

    return out


def compute_param_len(seq_len: int, d_half: int) -> int:
    """Computes total parameter count for a flattened HolographicMixer layer."""
    return d_half + (d_half * d_half) + d_half + seq_len


def unpack_mixer_weights(
    w_flat: torch.Tensor,
    seq_len: int,
    d_half: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpacks flattened layer parameter vector into component weights:
      - w_chan: (d_half,)
      - w_proj: (d_half, d_half)
      - b:      (d_half,)
      - w_seq:  (seq_len,)
    """
    idx = 0
    w_chan = w_flat[idx:idx + d_half]
    idx += d_half

    proj_size = d_half * d_half
    w_proj = w_flat[idx:idx + proj_size].view(d_half, d_half)
    idx += proj_size

    b = w_flat[idx:idx + d_half]
    idx += d_half

    w_seq = w_flat[idx:idx + seq_len]
    return w_seq, w_chan, w_proj, b


def functional_holographic_mixer(
    R: torch.Tensor,
    W_flat: torch.Tensor,
    activation: str = "silu",
    causal: bool = True
) -> torch.Tensor:
    """
    Functional HolographicMixer execution (pure deterministic function for autograd).

    Steps:
      1. Causal sequence mixing along sequence dimension (dim=1) using FFT convolution.
      2. Holographic channel binding along hidden dimension (dim=-1) using circular convolution.
      3. Linear projection and bias.
      4. Gradient-preserving non-linearity (SiLU or scaled Tanh).
    """
    B, L, D_half = R.shape
    w_seq, w_chan, w_proj, b = unpack_mixer_weights(W_flat, L, D_half)

    # 1. Causal sequence mixing
    R_seq = holographic_circular_conv_1d(R, w_seq, dim=1, n=L, causal=causal)

    # 2. Holographic channel binding
    R_chan = holographic_circular_conv_1d(R_seq, w_chan, dim=-1, n=D_half, causal=False)

    # 3. Dense channel projection
    proj_out = torch.matmul(R_chan, w_proj) + b

    # 4. Activation function
    if activation == "silu":
        return F.silu(proj_out)
    elif activation == "tanh_scaled":
        return 1.7159 * torch.tanh((2.0 / 3.0) * proj_out)
    elif activation == "gelu":
        return F.gelu(proj_out)
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class HolographicMixer(nn.Module):
    """
    PyTorch Module wrapper around the HolographicMixer core.
    """
    def __init__(
        self,
        d_half: int,
        max_seq_len: int = 128,
        activation: str = "silu",
        causal: bool = True
    ):
        super().__init__()
        self.d_half = d_half
        self.max_seq_len = max_seq_len
        self.activation = activation
        self.causal = causal

        # Initialize weights with standard scaling
        self.w_chan = nn.Parameter(torch.randn(d_half) * (1.0 / math.sqrt(d_half)))
        self.w_proj = nn.Parameter(torch.randn(d_half, d_half) * (1.0 / math.sqrt(d_half)))
        self.b = nn.Parameter(torch.zeros(d_half))
        self.w_seq = nn.Parameter(torch.randn(max_seq_len) * (1.0 / math.sqrt(max_seq_len)))

    @property
    def flat_param_len(self) -> int:
        return compute_param_len(self.max_seq_len, self.d_half)

    def get_flat_weights(self, current_seq_len: Optional[int] = None) -> torch.Tensor:
        """Flattens parameters into a contiguous 1D tensor."""
        L = current_seq_len if current_seq_len is not None else self.max_seq_len
        return torch.cat([
            self.w_chan,
            self.w_proj.reshape(-1),
            self.b,
            self.w_seq[:L]
        ], dim=0)

    def forward(self, R: torch.Tensor, W_flat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes forward mixing pass.
        If W_flat is provided, uses dynamic weights; otherwise uses internal module parameters.
        """
        if W_flat is not None:
            return functional_holographic_mixer(R, W_flat, activation=self.activation, causal=self.causal)

        B, L, D_half = R.shape
        w_seq = self.w_seq[:L]
        R_seq = holographic_circular_conv_1d(R, w_seq, dim=1, n=L, causal=self.causal)
        R_chan = holographic_circular_conv_1d(R_seq, self.w_chan, dim=-1, n=self.d_half, causal=False)
        proj_out = torch.matmul(R_chan, self.w_proj) + self.b

        if self.activation == "silu":
            return F.silu(proj_out)
        elif self.activation == "tanh_scaled":
            return 1.7159 * torch.tanh((2.0 / 3.0) * proj_out)
        else:
            return F.silu(proj_out)
