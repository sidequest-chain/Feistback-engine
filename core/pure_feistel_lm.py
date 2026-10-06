"""
Pure Feistel Language Model (PureFeistelLM)
===========================================
Pure non-hybrid neural architecture without any Transformer attention layers:
  - Takes over pre-trained Qwen2.5-Coder token embeddings (vocab_size=151936, hidden_size=896)
    and language modeling head (lm_head).
  - Neural backbone: Feistel round function combining SpectralMixer (frequency-domain
    circular convolution) and GatedLinearRecurrence (O(1) state vector for indentation/syntax tracking).
  - Memory footprint: Exactly 0 bytes of KV-cache across all sequence lengths and batch sizes.
  - Supports:
      - forward(input_ids, states=None) -> logits, loss (O(1) recurrence state tracking)
      - forward_backbone(input_ids, states=None) -> representation vectors for fast distillation
      - generate(prompt, max_new_tokens, ...) -> O(1) state-passing autoregressive token generation
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from core.holographic_mixer import functional_holographic_mixer, compute_param_len


class GatedLinearRecurrence(nn.Module):
    """
    Gated Linear Recurrence (GLR) module.
    Maintains an O(1) state vector to track code indentations, scope nesting,
    and syntactic dependencies without any KV-cache.

    Equations:
      u_t = silu(in_proj(x_t))
      g   = sigmoid(decay_bias)
      s_t = g * s_{t-1} + (1 - g) * u_t
      y_t = s_t @ in_proj.weight  (tied projection for parameter efficiency)
    """

    def __init__(self, dim: int, state_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        # Tied projection: in_proj maps dim -> state_dim, output projection uses in_proj.weight.t()
        self.in_proj = nn.Linear(dim, state_dim, bias=False)
        # Learnable decay gate bias initialized to ~0.92 (log-odds 2.5) for persistent syntax tracking
        self.decay_bias = nn.Parameter(torch.ones(state_dim) * 2.5)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass supporting both single-step autoregressive decoding (L=1)
        and sequence-level parallel processing (L > 1) via causal 1D depthwise convolution.

        Args:
            x: Input tensor of shape (B, L, dim).
            state: Optional state tensor of shape (B, state_dim).

        Returns:
            out: Output tensor of shape (B, L, dim).
            new_state: Updated state tensor of shape (B, state_dim).
        """
        B, L, D = x.shape
        u = F.silu(self.in_proj(x))  # (B, L, state_dim)
        g = torch.sigmoid(self.decay_bias).view(1, 1, self.state_dim)

        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)

        if L == 1:
            g_vec = g.squeeze(1)  # (1, state_dim)
            new_state = g_vec * state + (1.0 - g_vec) * u.squeeze(1)  # (B, state_dim)
            # Tied output projection: (B, 1, state_dim) @ (state_dim, dim) -> (B, 1, dim)
            out = new_state.unsqueeze(1) @ self.in_proj.weight
            return out, new_state
        else:
            # Parallel causal sequence processing via depthwise 1D conv
            steps = torch.arange(L, device=x.device, dtype=torch.float32).view(1, 1, L)
            g_f = torch.sigmoid(self.decay_bias.float()).view(1, self.state_dim, 1)
            decay_kernel = (g_f ** steps) * (1.0 - g_f)
            weight = decay_kernel.transpose(0, 1).flip(-1).to(dtype=x.dtype)

            u_t = u.transpose(1, 2)  # (B, state_dim, L)
            u_padded = F.pad(u_t, (L - 1, 0))
            conv_out = F.conv1d(u_padded, weight, groups=self.state_dim)  # (B, state_dim, L)

            decay_s0 = (state.unsqueeze(-1).float() * (g_f ** (steps + 1))).to(dtype=x.dtype)
            states_seq = (conv_out + decay_s0).transpose(1, 2)  # (B, L, state_dim)
            new_state = states_seq[:, -1, :]
            out = states_seq @ self.in_proj.weight
            return out, new_state


class SpectralMixer(nn.Module):
    """
    Spectral / Holographic circular convolution filter for causal frequency-domain mixing.
    """

    def __init__(self, d_half: int, max_seq_len: int = 512, activation: str = "silu") -> None:
        super().__init__()
        self.d_half = d_half
        self.max_seq_len = max_seq_len
        self.activation = activation
        self.param_len = compute_param_len(max_seq_len, d_half)
        self.weight = nn.Parameter(torch.randn(self.param_len) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional_holographic_mixer(x, self.weight, activation=self.activation, causal=True)


class FeistelBlock(nn.Module):
    """
    Feistel round function block combining SpectralMixer and GatedLinearRecurrence:
        F(x) = SpectralMixer(x) + GLR(x, state)
        L_{i+1} = R_i
        R_{i+1} = L_i + layer_scale * F(R_i, state_i)
    """

    def __init__(
        self,
        d_half: int,
        state_dim: int = 128,
        max_seq_len: int = 512,
        layer_scale: float = 0.05,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.d_half = d_half
        self.state_dim = state_dim
        self.layer_scale = layer_scale
        self.spectral_mixer = SpectralMixer(d_half=d_half, max_seq_len=max_seq_len, activation=activation)
        self.glr = GatedLinearRecurrence(dim=d_half, state_dim=state_dim)

    def forward(
        self,
        L: torch.Tensor,
        R: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_spectral = self.spectral_mixer(R)
        f_glr, new_state = self.glr(R, state=state)
        f_total = f_spectral + f_glr

        L_next = R
        R_next = L + self.layer_scale * f_total
        return L_next, R_next, new_state


class PureFeistelBackbone(nn.Module):
    """
    Multi-layer Pure Feistel Backbone equipped with O(1) GatedLinearRecurrence states
    and holographic frequency mixers across all layers.
    Zero Transformer layers, exactly 0 bytes KV-cache.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        state_dim: int = 128,
        max_seq_len: int = 512,
        layer_scale: float = 0.05,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model ({d_model}) must be divisible by 2!"
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_half = d_model // 2
        self.state_dim = state_dim
        self.max_seq_len = max_seq_len
        self.layer_scale = layer_scale
        self.activation = activation

        self.layers = nn.ModuleList([
            FeistelBlock(
                d_half=self.d_half,
                state_dim=state_dim,
                max_seq_len=max_seq_len,
                layer_scale=layer_scale,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L, D = x.shape
        if states is None:
            states = [torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]

        L_cur, R_cur = torch.chunk(x, 2, dim=-1)
        new_states = []

        for i, layer in enumerate(self.layers):
            s_i = states[i] if states is not None and i < len(states) else None
            L_cur, R_cur, new_s = layer(L_cur, R_cur, state=s_i)
            new_states.append(new_s)

        y = torch.cat([L_cur, R_cur], dim=-1)
        return y, new_states


class PureFeistelLM(nn.Module):
    """
    Pure Feistel Language Model with zero Transformer attention layers and 0 bytes KV-cache.
    Replaces quadratic attention mechanisms with reversible holographic frequency-domain dynamics
    and O(1) GatedLinearRecurrence state tracking.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 896,
        num_layers: int = 12,
        state_dim: int = 128,
        max_seq_len: int = 512,
        layer_scale: float = 0.05,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        from_pretrained_qwen: bool = True,
        qwen_model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
        pruned_weights_path: str = "checkpoints/vocab_16k/pruned_weights.pt",
        vocab_map_path: str = "checkpoints/vocab_16k/vocab_map.pt",
        tokenizer: Optional[Any] = None,
        embed_tokens: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.max_seq_len = max_seq_len
        self.layer_scale = layer_scale
        self.torch_dtype = torch_dtype
        self.old_to_new: Optional[torch.Tensor] = None
        self.new_to_old: Optional[torch.Tensor] = None

        # 1. Token Embeddings, Final Norm, and LM Head
        if embed_tokens is not None and norm is not None and lm_head is not None:
            self.embed_tokens = embed_tokens
            self.norm = norm
            self.lm_head = lm_head
            self.tokenizer = tokenizer
        elif vocab_size == 16384:
            self.embed_tokens = nn.Embedding(vocab_size, d_model).to(self.device, dtype=torch_dtype)
            self.norm = nn.RMSNorm(d_model, eps=1e-6).to(self.device, dtype=torch_dtype)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device, dtype=torch_dtype)

            if os.path.exists(pruned_weights_path):
                pruned_weights = torch.load(pruned_weights_path, map_location=self.device)
                self.embed_tokens.weight.data.copy_(pruned_weights["embed_tokens"].to(self.device, dtype=torch_dtype))
                self.lm_head.weight.data.copy_(pruned_weights["lm_head"].to(self.device, dtype=torch_dtype))
                if "norm" in pruned_weights:
                    self.norm.weight.data.copy_(pruned_weights["norm"].to(self.device, dtype=torch_dtype))
                print(f"[PureFeistelLM] Loaded pretrained pruned weights from {pruned_weights_path}")
            else:
                print(f"[PureFeistelLM] Warning: pruned weights '{pruned_weights_path}' not found!")

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        elif from_pretrained_qwen:
            qwen = Qwen2ForCausalLM.from_pretrained(
                qwen_model_name,
                torch_dtype=torch_dtype,
                device_map=str(self.device),
            )
            self.embed_tokens = qwen.model.embed_tokens
            self.norm = qwen.model.norm
            self.lm_head = qwen.lm_head

            del qwen.model.layers
            del qwen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        else:
            self.embed_tokens = nn.Embedding(vocab_size, d_model).to(self.device, dtype=torch_dtype)
            self.norm = nn.RMSNorm(d_model).to(self.device, dtype=torch_dtype)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(self.device, dtype=torch_dtype)
            self.tokenizer = tokenizer

        # Load vocab mapping for 16k space if available
        if self.vocab_size == 16384 and os.path.exists(vocab_map_path):
            vmap = torch.load(vocab_map_path, map_location="cpu")
            self.old_to_new = vmap["old_to_new"].to(self.device)
            self.new_to_old = vmap["new_to_old"].to(self.device)

        # 2. Feistel Holographic Backbone with GLR (0 bytes KV cache)
        self.feistel_backbone = PureFeistelBackbone(
            num_layers=num_layers,
            d_model=d_model,
            state_dim=state_dim,
            max_seq_len=max_seq_len,
            layer_scale=layer_scale,
        ).to(self.device, dtype=torch_dtype)

        # Freeze embeddings, norm, and lm_head by default; core trainable backbone is enabled
        self.embed_tokens.requires_grad_(False)
        self.norm.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.feistel_backbone.requires_grad_(True)

        self.eval()

    @property
    def backbone(self) -> PureFeistelBackbone:
        """Alias for feistel_backbone to support standard student.backbone interface."""
        return self.feistel_backbone

    def count_trainable_parameters(self) -> int:
        """Counts trainable parameters in the core Feistel backbone."""
        return sum(p.numel() for p in self.feistel_backbone.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]],
    ]:
        """
        Forward computation returning vocabulary logits.

        Args:
            input_ids: Tensor of shape (B, L) with token IDs.
            targets: Optional tensor of shape (B, L) with target token IDs for loss computation.
            states: Optional list of O(1) recurrent states for each Feistel layer.
            return_states: If True, also returns updated list of layer states.

        Returns:
            logits: Vocabulary prediction logits of shape (B, L, vocab_size).
            (optional) loss: Cross-entropy loss if targets is provided.
            (optional) states: Updated layer states if return_states=True.
        """
        input_ids = input_ids.to(self.device)
        B, L = input_ids.shape
        assert L <= self.max_seq_len, f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}!"

        # 1. Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # 2. Pure Feistel Backbone with GLR (0 bytes KV cache)
        hidden_states, next_states = self.feistel_backbone(hidden_states, states=states)

        # 3. RMSNorm and LM Head projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if targets is not None:
            targets = targets.to(self.device)
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if return_states:
                return logits, loss, next_states
            return logits, loss

        if return_states:
            return logits, next_states

        return logits

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        with_norm: bool = True,
        states: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Computes forward pass through embeddings, Feistel backbone, and optional RMSNorm.
        Bypasses the 151k lm_head for fast representation distillation.
        """
        input_ids = input_ids.to(self.device)
        x = self.embed_tokens(input_ids)
        h, next_states = self.feistel_backbone(x, states=states)
        if with_norm:
            h = self.norm(h)
        if return_states:
            return h, next_states
        return h

    def load_backbone(self, checkpoint: Union[str, dict]) -> None:
        """
        Loads trained backbone parameters from a checkpoint file path or state dict.
        Supports both modern GLR backbone state dicts and legacy weights tensors.
        """
        if isinstance(checkpoint, str):
            state = torch.load(checkpoint, map_location=self.device)
        else:
            state = checkpoint

        if isinstance(state, dict):
            if "backbone_state_dict" in state:
                state_dict = state["backbone_state_dict"]
            elif "backbone" in state:
                state_dict = state["backbone"]
            elif "state_dict" in state:
                state_dict = state["state_dict"]
            else:
                state_dict = state

            # Check if this is a legacy checkpoint with a flat 'weights' tensor (12, param_len)
            if "weights" in state_dict and not any(k.startswith("layers.") for k in state_dict):
                legacy_w = state_dict["weights"]
                with torch.no_grad():
                    for i, layer in enumerate(self.feistel_backbone.layers):
                        if i < legacy_w.shape[0]:
                            layer.spectral_mixer.weight.copy_(legacy_w[i])
                print("[PureFeistelLM] Loaded legacy spectral weights into Feistel backbone.")
            else:
                missing, unexpected = self.feistel_backbone.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[PureFeistelLM] Note: {len(missing)} keys not found in checkpoint.")
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extracts step-by-step intermediate representations across all Feistel layers.
        Essential for representation distillation and layer-wise feature alignment.
        """
        input_ids = input_ids.to(self.device)
        x = self.embed_tokens(input_ids)
        states = [x]

        L_cur, R_cur = torch.chunk(x, 2, dim=-1)
        B = x.shape[0]
        glr_states = [torch.zeros(B, self.feistel_backbone.state_dim, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]

        with torch.no_grad():
            for i, layer in enumerate(self.feistel_backbone.layers):
                L_cur, R_cur, glr_states[i] = layer(L_cur, R_cur, state=glr_states[i])
                states.append(torch.cat([L_cur, R_cur], dim=-1))

        return states

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        suppress_special_tokens: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """
        Autoregressive token generation maintaining an O(1) persistent state vector.
        Guarantees 0 bytes KV-cache allocation.
        """
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required to generate from string prompt!")
            encoded = self.tokenizer(prompt, return_tensors="pt")
            prompt_tokens_qwen = encoded["input_ids"].to(self.device)
            if self.vocab_size == 16384 and self.old_to_new is not None:
                prompt_tokens_16k = self.old_to_new[prompt_tokens_qwen]
                eos_16k = self.old_to_new[self.tokenizer.eos_token_id].item() if self.tokenizer.eos_token_id is not None else 0
                prompt_tokens_16k = torch.where(
                    prompt_tokens_16k >= 0,
                    prompt_tokens_16k,
                    torch.tensor(eos_16k, device=self.device, dtype=prompt_tokens_16k.dtype),
                )
                input_ids = prompt_tokens_16k
            else:
                input_ids = prompt_tokens_qwen
        else:
            input_ids = prompt.to(self.device)

        B = input_ids.shape[0]
        generated_ids = input_ids.clone()
        suppressed_ids = []
        if suppress_special_tokens:
            if self.vocab_size == 16384 and self.old_to_new is not None:
                for fid in range(151644, 151665):
                    if fid < len(self.old_to_new):
                        nid = self.old_to_new[fid].item()
                        if nid >= 0:
                            suppressed_ids.append(nid)
            else:
                suppressed_ids = list(range(151644, 151665))

        # Initialize persistent layer states: list of (B, state_dim) tensors across all layers
        layer_states = [
            torch.zeros(B, self.feistel_backbone.state_dim, device=self.device, dtype=self.torch_dtype)
            for _ in range(self.num_layers)
        ]

        # Cache for whitespace / indentation tokens to avoid repeated decoding
        ws_cache: Dict[int, bool] = {}

        def is_whitespace_token(tid: int) -> bool:
            if tid not in ws_cache:
                if self.tokenizer is not None:
                    if self.vocab_size == 16384 and self.new_to_old is not None:
                        qwen_tid = self.new_to_old[tid].item()
                        tok_str = self.tokenizer.decode([qwen_tid])
                    else:
                        tok_str = self.tokenizer.decode([tid])
                    ws_cache[tid] = (not tok_str.strip())
                else:
                    ws_cache[tid] = False
            return ws_cache[tid]

        # Track consecutive whitespace tokens to prevent whitespace avalanches
        consecutive_ws = 0

        eos_id = (
            self.old_to_new[self.tokenizer.eos_token_id].item()
            if (self.vocab_size == 16384 and self.old_to_new is not None and self.tokenizer is not None and self.tokenizer.eos_token_id is not None)
            else (self.tokenizer.eos_token_id if self.tokenizer is not None else -1)
        )

        with torch.no_grad():
            # 1. Prefill prompt to establish initial recurrent states
            logits, layer_states = self.forward(input_ids, states=layer_states, return_states=True)
            next_token_logits = logits[:, -1, :].clone()

            for _ in range(max_new_tokens):
                for fid in suppressed_ids:
                    if fid < next_token_logits.shape[-1]:
                        next_token_logits[:, fid] = -float("inf")

                if temperature > 0.0:
                    scaled_logits = next_token_logits / temperature

                    # Repetition penalty (whitespace exempt only for up to 2 consecutive tokens)
                    if repetition_penalty != 1.0:
                        for tid in set(generated_ids[0].tolist()):
                            is_ws = is_whitespace_token(tid)
                            # If whitespace and consecutive count is under 2, exempt it (allow indentation)
                            if is_ws and consecutive_ws < 2:
                                continue

                            if scaled_logits[0, tid] < 0:
                                scaled_logits[0, tid] *= repetition_penalty
                            else:
                                scaled_logits[0, tid] /= repetition_penalty

                    # Top-P (Nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens exceeding cumulative probability threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift right to preserve first token exceeding threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False

                        # Scatter mask back to original index space
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            dim=1, index=sorted_indices, src=sorted_indices_to_remove
                        )
                        scaled_logits[indices_to_remove] = -float("inf")

                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    if repetition_penalty != 1.0:
                        for tid in set(generated_ids[0].tolist()):
                            is_ws = is_whitespace_token(tid)
                            if is_ws and consecutive_ws < 2:
                                continue

                            if next_token_logits[0, tid] < 0:
                                next_token_logits[0, tid] *= repetition_penalty
                            else:
                                next_token_logits[0, tid] /= repetition_penalty
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Update running consecutive whitespace counter
                next_token_id = next_token.item()
                if is_whitespace_token(next_token_id):
                    consecutive_ws += 1
                else:
                    consecutive_ws = 0

                if next_token_id == eos_id:
                    break

                # Autoregressively forward ONLY the new token (L=1) with persistent layer_states
                # Guarantees strictly O(1) computation and 0 bytes KV-cache
                step_logits, layer_states = self.forward(next_token, states=layer_states, return_states=True)
                next_token_logits = step_logits[:, -1, :].clone()

        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        if self.tokenizer is not None:
            if self.vocab_size == 16384 and self.new_to_old is not None:
                qwen_gen_ids = self.new_to_old[new_token_ids]
                generated_text = self.tokenizer.decode(qwen_gen_ids[0], skip_special_tokens=True)
            else:
                generated_text = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        else:
            generated_text = ""
        return generated_text, generated_ids

    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Computes exact parameter counts, weight memory, and KV-cache footprint.
        """
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        norm_params = sum(p.numel() for p in self.norm.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        feistel_params = sum(p.numel() for p in self.feistel_backbone.parameters())

        are_tied = self.embed_tokens.weight.data_ptr() == self.lm_head.weight.data_ptr()
        if are_tied:
            total_params = embed_params + norm_params + feistel_params
        else:
            total_params = embed_params + norm_params + lm_head_params + feistel_params

        bytes_per_param = 2 if self.torch_dtype in (torch.bfloat16, torch.float16) else 4
        weight_memory_mb = (total_params * bytes_per_param) / (1024**2)

        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "embed_params": embed_params,
            "norm_params": norm_params,
            "lm_head_params": lm_head_params,
            "feistel_backbone_params": feistel_params,
            "are_embeddings_tied": are_tied,
            "total_params": total_params,
            "weight_memory_mb": round(weight_memory_mb, 2),
            "kv_cache_bytes": 0,
            "state_dim": self.feistel_backbone.state_dim,
            "dtype": str(self.torch_dtype),
        }
