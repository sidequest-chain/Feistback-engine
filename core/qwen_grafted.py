"""
Grafted Qwen2.5-Coder Model (EXP-010 / EXP-011)
==============================================
Grafts a pre-trained Qwen2.5-Coder-0.5B model by replacing its final K attention
layers with a custom ReversibleFeistelNetwork (num_layers=K, d_model=896).
  - EXP-010: K=4 (20 Qwen + 4 Feistel, 16.7% KV-cache reduction)
  - EXP-011: K=8 (16 Qwen + 8 Feistel, 33.3% KV-cache reduction)

Architecture:
  - Input tokens -> Token embeddings
  - First (24 - K) QwenDecoderLayer blocks (e.g. layers 0..15 when K=8)
  - K Reversible Feistel layers with O(1) activation memory and zero KV-cache growth
  - Qwen RMSNorm and lm_head projection to vocabulary logits.

Supports:
  - Both pure baseline mode and grafted mode for direct ablation and benchmark comparison.
  - Forward logit calculation and autoregressive token generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM

from core.reversible_feistel import ReversibleFeistelNetwork


class GraftedQwenCoder(nn.Module):
    """
    Hybrid model grafting ReversibleFeistelNetwork onto the final layers of Qwen2.5-Coder-0.5B.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
        num_grafted_layers: int = 8,
        max_seq_len: int = 512,
        layer_scale: float = 0.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        base_model: Optional[Qwen2ForCausalLM] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_grafted_layers = num_grafted_layers
        self.torch_dtype = torch_dtype
        self.max_seq_len = max_seq_len
        self.layer_scale = layer_scale

        # 1. Load Base Model and Tokenizer
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = Qwen2ForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=str(self.device),
            )
        self.base_model.eval()

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Total layers and dimensions
        self.total_layers = len(self.base_model.model.layers)
        self.num_retained_layers = self.total_layers - num_grafted_layers
        self.d_model = self.base_model.config.hidden_size  # 896 for Qwen2.5-Coder-0.5B

        # 2. Initialize Reversible Feistel Graft
        # Mathematical identity mapping invariant:
        # When layer_scale = 0.0, the Feistel round simplifies to:
        #   R_out = R_in + 0.0 * F(L_in) = R_in
        #   L_out = L_in + 0.0 * F(R_out) = L_in
        # Over an even number of layers (e.g. N=8 or N=4), the alternating swaps restore
        # L_N = L_0 and R_N = R_0, guaranteeing a transparent identity transformation
        # that preserves the pre-trained semantic representations of retained Qwen layers.
        self.feistel = ReversibleFeistelNetwork(
            num_layers=num_grafted_layers,
            d_model=self.d_model,
            max_seq_len=max_seq_len,
            layer_scale=layer_scale,
        ).to(self.device, dtype=torch_dtype)
        self.feistel.eval()

        # Operational mode: "grafted" or "baseline"
        self.mode = "grafted"

    def set_mode(self, mode: str) -> None:
        """Switch between 'grafted' and 'baseline' execution modes."""
        if mode not in ("grafted", "baseline"):
            raise ValueError(f"Invalid mode '{mode}'. Choose 'grafted' or 'baseline'.")
        self.mode = mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward computation returning vocabulary logits.
        """
        active_mode = mode if mode is not None else self.mode
        input_ids = input_ids.to(self.device)

        if active_mode == "baseline":
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            return outputs.logits

        # Grafted mode:
        # 1. Embed tokens
        hidden_states = self.base_model.model.embed_tokens(input_ids)

        # 2. Retained Transformer layers (layers 0 .. num_retained_layers - 1)
        batch_size, seq_len = input_ids.shape
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        mask_kwargs = {
            "config": self.base_model.config,
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        try:
            from transformers.models.qwen2.modeling_qwen2 import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if getattr(self.base_model.model, "has_sliding_layers", False):
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )
        except (ImportError, Exception):
            causal_mask_mapping = {"full_attention": None}

        position_embeddings = None
        if hasattr(self.base_model.model, "rotary_emb"):
            position_embeddings = self.base_model.model.rotary_emb(
                hidden_states, position_ids
            )

        with torch.no_grad():
            for i in range(self.num_retained_layers):
                layer = self.base_model.model.layers[i]
                layer_type = (
                    self.base_model.config.layer_types[i]
                    if hasattr(self.base_model.config, "layer_types")
                    else "full_attention"
                )
                mask = causal_mask_mapping.get(
                    layer_type, causal_mask_mapping["full_attention"]
                )
                layer_kwargs = {
                    "hidden_states": hidden_states,
                    "attention_mask": mask,
                    "position_ids": position_ids,
                    "use_cache": False,
                }
                if position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings

                layer_outputs = layer(**layer_kwargs)
                hidden_states = (
                    layer_outputs
                    if isinstance(layer_outputs, torch.Tensor)
                    else layer_outputs[0]
                )

            # 3. Grafted Reversible Feistel layers (replacing layers 20..23)
            # ReversibleFeistelNetwork processes hidden_states (B, L, D) with O(1) activation memory
            hidden_states = self.feistel(hidden_states)

            # 4. Final RMSNorm and LM Head
            hidden_states = self.base_model.model.norm(hidden_states)
            logits = self.base_model.lm_head(hidden_states)

        return logits

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        mode: Optional[str] = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Autoregressive text generation with FIM / control token suppression to prevent tag drift.
        """
        if isinstance(prompt, str):
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
        else:
            input_ids = prompt.to(self.device)

        active_mode = mode if mode is not None else self.mode
        generated_ids = input_ids.clone()

        # Suppress non-EOS control and FIM tokens (tokens >= 151644) to prevent representation drift
        # towards special control tags like <|fim_middle|> during causal code synthesis
        suppressed_token_ids = list(range(151644, 151665))

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated_ids, mode=active_mode)
                next_token_logits = logits[:, -1, :].clone()

                for fid in suppressed_token_ids:
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

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        new_token_ids = generated_ids[:, input_ids.shape[-1] :]
        generated_text = self.tokenizer.decode(new_token_ids[0], skip_special_tokens=True)
        return generated_text, generated_ids
