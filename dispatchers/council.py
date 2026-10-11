"""
Council of Specialties Intent Dispatcher (Tier 2)
=================================================
Couples the specialized eesti_keele_sl2ng micro-model (semantic Estonian & IT-slang expert)
with the PureFeistelLM code generation backbone and MiniStackVM execution sandbox.
Provides conversational multi-turn slot filling, AST validation, and dynamic contract execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

from dispatchers.base import (
    BaseDispatcher,
    extract_numbers,
    is_synthesis_query,
)

try:
    from runtime.engine import FeistbackEngine, EngineResult
    from runtime.models.pure_feistel_lm import PureFeistelLM
except ImportError:
    from runner.engine import FeistbackEngine, EngineResult
    from core.pure_feistel_lm import PureFeistelLM


class CouncilDispatcher(BaseDispatcher):
    """
    CouncilDispatcher routes queries dynamically through the trained eesti_keele_sl2ng
    micro-model to synthesize formal execution contracts, falling back to deterministic
    number parsing and MiniStackVM execution.
    """

    def __init__(
        self,
        engine: FeistbackEngine,
        model_path: str = "models/eesti_keele_sl2ng/slang_expert_feistel.pt",
        tokenizer_path: str = "models/eesti_keele_sl2ng/tokenizer.json",
        device: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.expert_model: Optional[PureFeistelLM] = None
        self.expert_tokenizer: Optional[Tokenizer] = None

        self._load_expert()

    def _load_expert(self) -> None:
        """Loads eesti_keele_sl2ng micro-model weights and tokenizer."""
        # Resolve model path with fallbacks
        mp = Path(self.model_path)
        if not mp.is_absolute():
            mp = Path(__file__).resolve().parent.parent / self.model_path
        if not mp.exists():
            alt_mp = Path(__file__).resolve().parent.parent / "checkpoints" / "slang_expert_feistel.pt"
            if alt_mp.exists():
                mp = alt_mp

        tp = Path(self.tokenizer_path)
        if not tp.is_absolute():
            tp = Path(__file__).resolve().parent.parent / self.tokenizer_path
        if not tp.exists():
            alt_tp = Path(__file__).resolve().parent.parent / "checkpoints" / "slang_tokenizer_4k" / "tokenizer.json"
            if alt_tp.exists():
                tp = alt_tp

        if mp.exists() and tp.exists():
            try:
                self.expert_tokenizer = Tokenizer.from_file(str(tp))
                ckpt = torch.load(mp, map_location=self.device)
                cfg = ckpt.get("config", {})
                self.expert_model = PureFeistelLM(
                    vocab_size=cfg.get("vocab_size", 1327),
                    d_model=cfg.get("d_model", 512),
                    num_layers=cfg.get("num_layers", 12),
                    max_seq_len=cfg.get("max_seq_len", 128),
                    memory_type=cfg.get("memory_type", "raf"),
                    num_heads=cfg.get("num_heads", 4),
                    key_dim=cfg.get("key_dim", 16),
                    value_dim=cfg.get("value_dim", 16),
                    torch_dtype=torch.float32,
                    device=self.device,
                    from_pretrained_qwen=False,
                )
                self.expert_model.load_state_dict(ckpt["model_state_dict"])
                self.expert_model.eval()
            except Exception as e:
                print(f"Warning: Failed to load eesti_keele_sl2ng expert: {e}")
                self.expert_model = None

    def predict_contract(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Invokes eesti_keele_sl2ng micro-model to generate structured contract JSON."""
        if self.expert_model is None or self.expert_tokenizer is None:
            return None

        prompt_str = f"<|user|> {user_query} <|contract|>"
        prompt_ids = self.expert_tokenizer.encode(prompt_str).ids
        curr_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        eos_id = self.expert_tokenizer.token_to_id("<|endoftext|>")

        gen_ids: List[int] = []
        with torch.no_grad():
            for _ in range(96):
                logits = self.expert_model(curr_tensor)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tid = next_tok.item()
                if tid == eos_id:
                    break
                gen_ids.append(tid)
                curr_tensor = torch.cat([curr_tensor, next_tok], dim=-1)

        raw_text = self.expert_tokenizer.decode(gen_ids).strip()

        # Parse outermost JSON object
        start_brace = raw_text.find("{")
        end_brace = raw_text.rfind("}")
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            try:
                return json.loads(raw_text[start_brace : end_brace + 1])
            except json.JSONDecodeError:
                pass
        return None

    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Dispatches queries using neural language expert or pending intent state."""
        # 1. Resolve pending intent
        if pending_intent is not None:
            new_numbers = extract_numbers(user_query)
            args = (pending_args or []) + new_numbers
            signature = pending_intent.get("signature", "def add(a: int, b: int) -> int:")
            intent_name = pending_intent.get("intent_name", "calculation")

            # Determine expected count from signature or hint
            expected_count = len(extract_numbers(signature)) or 2

            if len(args) < expected_count:
                return {
                    "status": "NEED_INPUT",
                    "mode": "execute",
                    "intent": pending_intent,
                    "intent_name": intent_name,
                    "provided_args": args,
                    "expected_count": expected_count,
                    "message": f"Kavatsus '{intent_name}' vajab {expected_count} arvu (leiti {len(args)}). Palun sisesta puuduvad arvud.",
                    "query": user_query,
                }

            target_args = args[:expected_count]
            engine_res = self.engine.generate_and_verify(
                prompt=signature,
                test_inputs=[target_args],
            )
            vm_res = engine_res.test_outcomes[0].vm_result if engine_res.test_outcomes else None
            return {
                "status": "SUCCESS" if engine_res.verified else "VERIFICATION_FAILED",
                "mode": "execute",
                "intent_id": intent_name,
                "intent_name": intent_name,
                "args": target_args,
                "result": vm_res,
                "verified": engine_res.verified,
                "generated_code": engine_res.code,
                "disassembly": engine_res.disassembly,
                "vm_latency_us": engine_res.latency_vm_avg_us,
                "query": user_query,
            }

        # 2. Check for synthesis request
        is_synth = is_synthesis_query(user_query)

        # 3. Predict contract via Language Expert
        contract = self.predict_contract(user_query)
        if contract is None:
            # Fallback to pure number calculation
            nums = extract_numbers(user_query)
            if len(nums) >= 2:
                contract = {
                    "intent_name": "add",
                    "strategy": "single_pass",
                    "signature": "def add(a: int, b: int) -> int:",
                    "args": nums[:2],
                }
            else:
                return {
                    "status": "UNRECOGNIZED_INTENT",
                    "error": f"Keelespetsialist ei suutnud tuvastada lepingut päringule: '{user_query}'",
                    "query": user_query,
                }

        intent_name = contract.get("intent_name", "custom_intent")
        strategy = contract.get("strategy", "single_pass")
        signature = contract.get("signature", "def func(a: int, b: int) -> int:")
        args = contract.get("args", [])

        # Fallback argument extraction if empty in contract
        if not args and strategy != "need_input":
            extracted = extract_numbers(user_query)
            if extracted:
                args = extracted

        # If slot filling is required
        if strategy == "need_input" or (not is_synth and not args):
            return {
                "status": "NEED_INPUT",
                "mode": "execute",
                "intent": contract,
                "intent_name": intent_name,
                "provided_args": args,
                "expected_count": 2,
                "message": f"Kavatsus '{intent_name}' vajab täiendavaid arve. Palun sisesta arvud.",
                "query": user_query,
            }

        # Synthesis mode
        if is_synth:
            t0 = time.perf_counter()
            gen_raw, _ = self.engine.model.generate(
                prompt=signature,
                max_new_tokens=64,
                temperature=0.0,
                early_stopping=True,
            )
            gen_lat_s = time.perf_counter() - t0
            indented = "\n".join("    " + line.strip() for line in gen_raw.split("\n") if line.strip())
            full_code = signature.strip() + "\n" + indented
            return {
                "status": "SUCCESS",
                "mode": "synthesize",
                "intent_id": intent_name,
                "intent_name": intent_name,
                "signature": signature,
                "generated_code": full_code,
                "generation_latency_s": gen_lat_s,
                "message": f"Genereeritud puhas funktsioonikood kavatsusele '{intent_name}'.",
                "query": user_query,
            }

        # Execution mode
        engine_res = self.engine.generate_and_verify(
            prompt=signature,
            test_inputs=[args],
        )
        vm_res = engine_res.test_outcomes[0].vm_result if engine_res.test_outcomes else None

        return {
            "status": "SUCCESS" if engine_res.verified else "VERIFICATION_FAILED",
            "mode": "execute",
            "intent_id": intent_name,
            "intent_name": intent_name,
            "args": args,
            "result": vm_res,
            "verified": engine_res.verified,
            "generated_code": engine_res.code,
            "disassembly": engine_res.disassembly,
            "vm_latency_us": engine_res.latency_vm_avg_us,
            "query": user_query,
        }
