"""
Pure Feistel Language Model (PureFeistelLM)
===========================================
Pure non-hybrid neural architecture without any Transformer attention layers:
  - Takes over pre-trained Qwen2.5-Coder token embeddings (vocab_size=151936, hidden_size=896)
    and language modeling head (lm_head).
  - Neural backbone: ReversibleFeistelNetwork with HolographicMixer circular convolution
    and dense channel projection blocks.
  - Memory footprint: Exactly 0 bytes of KV-cache across all sequence lengths and batch sizes.
  - Supports:
      - forward(input_ids) -> logits (with optional cross-entropy loss calculation)
      - get_hidden_states(input_ids) -> intermediate layer representations for distillation
      - generate(prompt, max_new_tokens, ...) -> autoregressive token generation
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from core.holographic_mixer import functional_holographic_mixer
from core.reversible_feistel import ReversibleFeistelNetwork


class PureFeistelLM(nn.Module):
    """
    Pure Feistel Language Model with zero Transformer attention layers and 0 bytes KV-cache.
    Replaces quadratic attention mechanisms with reversible holographic frequency-domain dynamics.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 896,
        num_layers: int = 12,
        max_seq_len: int = 512,
        layer_scale: float = 0.05,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        from_pretrained_qwen: bool = True,
        qwen_model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
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
        self.max_seq_len = max_seq_len
        self.layer_scale = layer_scale
        self.torch_dtype = torch_dtype

        # 1. Token Embeddings, Final Norm, and LM Head
        if embed_tokens is not None and norm is not None and lm_head is not None:
            self.embed_tokens = embed_tokens
            self.norm = norm
            self.lm_head = lm_head
            self.tokenizer = tokenizer
        elif from_pretrained_qwen:
            # Transfer embeddings and head directly from pre-trained Qwen2.5-Coder
            qwen = Qwen2ForCausalLM.from_pretrained(
                qwen_model_name,
                torch_dtype=torch_dtype,
                device_map=str(self.device),
            )
            self.embed_tokens = qwen.model.embed_tokens
            self.norm = qwen.model.norm
            self.lm_head = qwen.lm_head
            
            # Explicitly delete all Transformer decoder layers to free GPU memory
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

        # 2. Reversible Feistel Holographic Backbone (0 bytes KV cache)
        self.feistel_backbone = ReversibleFeistelNetwork(
            num_layers=num_layers,
            d_model=d_model,
            max_seq_len=max_seq_len,
            layer_scale=layer_scale,
        ).to(self.device, dtype=torch_dtype)

        self.eval()

    @property
    def backbone(self) -> ReversibleFeistelNetwork:
        """Alias for feistel_backbone to support standard student.backbone interface."""
        return self.feistel_backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward computation returning vocabulary logits.

        Args:
            input_ids: Tensor of shape (B, L) with token IDs.
            targets: Optional tensor of shape (B, L) with target token IDs for loss computation.

        Returns:
            logits: Vocabulary prediction logits of shape (B, L, vocab_size).
            (optional) loss: Cross-entropy loss if targets is provided.
        """
        input_ids = input_ids.to(self.device)
        B, L = input_ids.shape
        assert L <= self.max_seq_len, f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}!"

        # 1. Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # 2. Reversible Feistel Backbone (O(1) activation memory, 0 bytes KV cache)
        hidden_states = self.feistel_backbone(hidden_states)

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
            return logits, loss

        return logits

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        with_norm: bool = True,
    ) -> torch.Tensor:
        """
        Computes forward pass through embeddings, Feistel backbone, and optional RMSNorm.
        Bypasses the 151k lm_head for fast representation distillation.
        """
        input_ids = input_ids.to(self.device)
        x = self.embed_tokens(input_ids)
        h = self.feistel_backbone(x)
        if with_norm:
            h = self.norm(h)
        return h

    def load_backbone(self, checkpoint: Union[str, dict]) -> None:
        """
        Loads trained backbone parameters from a checkpoint file path or state dict.
        """
        if isinstance(checkpoint, str):
            state = torch.load(checkpoint, map_location=self.device)
        else:
            state = checkpoint

        if isinstance(state, dict):
            if "backbone_state_dict" in state:
                self.feistel_backbone.load_state_dict(state["backbone_state_dict"])
            elif "backbone" in state:
                self.feistel_backbone.load_state_dict(state["backbone"])
            elif "weights" in state:
                self.feistel_backbone.load_state_dict(state)
            elif "state_dict" in state:
                self.load_state_dict(state["state_dict"], strict=False)
            else:
                self.feistel_backbone.load_state_dict(state)
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extracts step-by-step intermediate representations across all Feistel layers.
        Essential for representation distillation and layer-wise feature alignment
        against teacher models.

        Returns:
            List of Tensors [h_0, h_1, ..., h_N] where:
              - h_0 is the initial token embedding.
              - h_i is the representation after Feistel layer i.
        """
        input_ids = input_ids.to(self.device)
        x = self.embed_tokens(input_ids)
        states = [x]

        L_cur, R_cur = torch.chunk(x, 2, dim=-1)
        mixer_callable = lambda r, w: functional_holographic_mixer(
            r, w, activation=self.feistel_backbone.activation
        )

        with torch.no_grad():
            for i in range(self.feistel_backbone.num_layers):
                W_i = self.feistel_backbone.weights[i]
                L_next = R_cur
                R_next = L_cur + self.feistel_backbone.layer_scale * mixer_callable(R_cur, W_i)
                L_cur, R_cur = L_next, R_next
                states.append(torch.cat([L_cur, R_cur], dim=-1))

        return states

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        suppress_special_tokens: bool = True,
    ) -> Tuple[str, torch.Tensor]:
        """
        Autoregressive token generation using pure Feistel forward dynamics.
        """
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required to generate from string prompt!")
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
        else:
            input_ids = prompt.to(self.device)

        generated_ids = input_ids.clone()
        suppressed_ids = list(range(151644, 151665)) if suppress_special_tokens else []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated_ids)
                next_token_logits = logits[:, -1, :].clone()

                for fid in suppressed_ids:
                    if fid < next_token_logits.shape[-1]:
                        next_token_logits[:, fid] = -float("inf")

                if repetition_penalty != 1.0:
                    for tid in set(generated_ids[0].tolist()):
                        if next_token_logits[0, tid] > 0:
                            next_token_logits[0, tid] /= repetition_penalty
                        else:
                            next_token_logits[0, tid] *= repetition_penalty

                if temperature > 0.0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if (
                    self.tokenizer is not None
                    and next_token.item() == self.tokenizer.eos_token_id
                ):
                    break

        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        generated_text = (
            self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
            if self.tokenizer is not None
            else ""
        )
        return generated_text, generated_ids

    def get_parameter_summary(self) -> Dict[str, Any]:
        """
        Computes exact parameter counts, weight memory, and KV-cache footprint.
        """
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        norm_params = sum(p.numel() for p in self.norm.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        feistel_params = sum(p.numel() for p in self.feistel_backbone.parameters())

        # Check if embeddings and lm_head share memory
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
            "dtype": str(self.torch_dtype),
        }
