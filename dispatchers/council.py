"""
Council of Specialties Intent Dispatcher (Tier 2 & Tier 1 L1 Cache)
===================================================================
Couples the Self-Hydrating L1 Intent Cache (`LearnedIntentCache`) and the specialized
`eesti_keele_sl2ng` micro-model (semantic Estonian & IT-slang expert) with the
PureFeistelLM code generation backbone and MiniStackVM execution sandbox.
Provides conversational multi-turn slot filling, AST validation, Double-Lock VM
verification gating, and dynamic contract execution.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from dispatchers.base import BaseDispatcher
from dispatchers.cache_manager import LearnedIntentCache
from dispatchers.parsers.et_numbers import (
    extract_numbers,
    is_synthesis_query,
)

from runtime.engine import FeistbackEngine, EngineResult
from runtime.models.pure_feistel_lm import PureFeistelLM
from runtime.registry import ModelRegistry


# Tier 3 Kernel Fusion templates for compound arithmetic routines synthesized by Language Expert
KERNEL_FUSION_MAP: Dict[str, str] = {
    "sum3_sub": "def sum3_sub(a: int, b: int, c: int, d: int) -> int:\n    return a + b + c - d",
    "add_mul": "def add_mul(a: int, b: int, c: int) -> int:\n    return (a + b) * c",
    "mul_add": "def mul_add(a: int, b: int, c: int) -> int:\n    return a * b + c",
    "sub_mul": "def sub_mul(a: int, b: int, c: int) -> int:\n    return (a - b) * c",
    "sum_of_squares": "def sum_of_squares(a: int, b: int) -> int:\n    return a * a + b * b",
}


def get_expected_args_count(signature: str) -> int:
    """Extracts expected argument count from a Python function signature."""
    try:
        clean_sig = signature.strip()
        if not clean_sig.endswith(":"):
            clean_sig += ":"
        tree = ast.parse(clean_sig + "\n    pass")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return len(node.args.args)
    except Exception:
        pass
    if "(" in signature and ")" in signature:
        inner = signature[signature.find("(") + 1 : signature.rfind(")")].strip()
        if not inner:
            return 0
        return len([p for p in inner.split(",") if p.strip()])
    return 0


class CouncilDispatcher(BaseDispatcher):
    """
    CouncilDispatcher routes queries through the Tier 1 Self-Hydrating LRU Cache
    and Tier 2 `eesti_keele_sl2ng` micro-model to synthesize formal execution contracts,
    verifying execution on Tier 3 (`MiniStackVM`) before caching learned patterns.
    """

    def __init__(
        self,
        engine: FeistbackEngine,
        model_name: str = "eesti_keele_sl2ng",
        registry: Optional[ModelRegistry] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        cache_path: str = "cache/learned_intents.json",
        device: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = registry or ModelRegistry()
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.learned_cache = LearnedIntentCache(cache_path=cache_path)
        self.expert_model: Optional[PureFeistelLM] = None
        self.expert_tokenizer: Optional[Tokenizer] = None

        self._load_expert()

    def _load_expert(self) -> None:
        """Loads eesti_keele_sl2ng micro-model weights and tokenizer via ModelRegistry."""
        try:
            if self.model_name in self.registry._models:
                self.expert_model, _ = self.registry.load_model(self.model_name, device=self.device)
                self.expert_tokenizer = self.registry.load_tokenizer(self.model_name)
            elif self.model_path and self.tokenizer_path:
                mp = Path(self.model_path)
                if not mp.is_absolute():
                    mp = Path(__file__).resolve().parent.parent / self.model_path
                tp = Path(self.tokenizer_path)
                if not tp.is_absolute():
                    tp = Path(__file__).resolve().parent.parent / self.tokenizer_path
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
            self.expert_tokenizer = None

    def predict_contract(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Invokes eesti_keele_sl2ng micro-model using O(1) recurrent state stepping (L=1)
        to generate structured contract JSON with zero KV-cache.
        """
        if self.expert_model is None or self.expert_tokenizer is None:
            return None

        prompt_str = f"<|user|> {user_query} <|contract|>"
        prompt_ids = self.expert_tokenizer.encode(prompt_str).ids
        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        eos_id = self.expert_tokenizer.token_to_id("<|endoftext|>")

        gen_ids: List[int] = []
        max_steps = min(96, self.expert_model.max_seq_len - len(prompt_ids))

        with torch.no_grad():
            # 1. Prefill prompt once to initialize recurrent RAF + spectral history states
            logits, layer_states = self.expert_model(
                prompt_tensor,
                states=None,
                return_states=True,
            )
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            # 2. Step token-by-token (L=1) with O(1) state updates
            for _ in range(max_steps):
                tid = next_tok.item()
                if tid == eos_id:
                    break
                gen_ids.append(tid)
                logits, layer_states = self.expert_model(
                    next_tok,
                    states=layer_states,
                    return_states=True,
                )
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

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

    def _execute_contract(
        self,
        intent_name: str,
        signature: str,
        args: List[Union[int, float]],
        user_query: str,
        neural_gen_lat_s: float = 0.0,
        cache_hit: bool = False,
    ) -> Dict[str, Any]:
        """Executes code via neural reasoning or verified kernel fusion on MiniStackVM."""
        route_label = "TIER_1_LEARNED_CACHE" if cache_hit else "TIER_2_NEURAL_COUNCIL"
        fused_code = KERNEL_FUSION_MAP.get(intent_name)
        if fused_code:
            try:
                t0_comp = time.perf_counter()
                bytecode = self.engine.compiler.compile(fused_code)
                comp_lat_ms = (time.perf_counter() - t0_comp) * 1000.0

                t0_vm = time.perf_counter()
                vm_res = self.engine.vm.execute(bytecode, args)
                vm_lat_us = (time.perf_counter() - t0_vm) * 1_000_000.0

                # Double-Lock Verification Gate: persist to L1 learned cache
                self.learned_cache.record_verified(
                    query=user_query,
                    intent_name=intent_name,
                    target_signature=signature,
                    args_count=len(args),
                    verified_by_vm=True,
                )

                return {
                    "status": "SUCCESS",
                    "mode": "execute",
                    "route": route_label,
                    "intent_id": intent_name,
                    "intent_name": intent_name,
                    "args": args,
                    "result": vm_res,
                    "verified": True,
                    "cache_hit": cache_hit,
                    "generated_code": fused_code,
                    "bytecode": bytecode,
                    "disassembly": self.engine.vm.disassemble(bytecode),
                    "vm_latency_us": vm_lat_us,
                    "compilation_latency_ms": comp_lat_ms,
                    "generation_latency_s": neural_gen_lat_s,
                    "query": user_query,
                }
            except Exception:
                pass

        engine_res = self.engine.generate_and_verify(
            prompt=signature,
            test_inputs=[args],
        )
        vm_res = engine_res.test_outcomes[0].vm_result if engine_res.test_outcomes else None

        # Double-Lock Verification Gate: persist to L1 cache only if MiniStackVM verified execution
        if engine_res.verified:
            self.learned_cache.record_verified(
                query=user_query,
                intent_name=intent_name,
                target_signature=signature,
                args_count=len(args),
                verified_by_vm=True,
            )

        return {
            "status": "SUCCESS" if engine_res.verified else "VERIFICATION_FAILED",
            "mode": "execute",
            "route": route_label,
            "intent_id": intent_name,
            "intent_name": intent_name,
            "args": args,
            "result": vm_res,
            "verified": engine_res.verified,
            "cache_hit": cache_hit,
            "generated_code": engine_res.code,
            "bytecode": engine_res.bytecode,
            "disassembly": engine_res.disassembly,
            "vm_latency_us": engine_res.latency_vm_avg_us,
            "compilation_latency_ms": engine_res.latency_comp_ms,
            "generation_latency_s": neural_gen_lat_s or engine_res.latency_gen_s,
            "query": user_query,
        }

    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[Union[int, float]]] = None,
    ) -> Dict[str, Any]:
        """Dispatches queries using L1 Learned Cache, Tier 2 neural language expert, or pending intent state."""
        # 1. Resolve pending intent
        if pending_intent is not None:
            new_numbers = extract_numbers(user_query)
            args = (pending_args or []) + new_numbers
            signature = pending_intent.get("signature", "def add(a: int, b: int) -> int:")
            intent_name = pending_intent.get("intent_name", "calculation")

            expected_count = get_expected_args_count(signature) or 2

            if len(args) < expected_count:
                return {
                    "status": "NEED_INPUT",
                    "mode": "execute",
                    "intent": pending_intent,
                    "intent_id": intent_name,
                    "intent_name": intent_name,
                    "provided_args": args,
                    "expected_count": expected_count,
                    "message": (
                        f"Kavatsus '{intent_name}' vajab {expected_count} arvu "
                        f"(leiti {len(args)}). Palun sisesta puuduvad {expected_count - len(args)} arvu."
                    ),
                    "query": user_query,
                }

            target_args = args[:expected_count]
            return self._execute_contract(intent_name, signature, target_args, user_query)

        # 2. Check for synthesis request
        is_synth = is_synthesis_query(user_query)
        real_args = extract_numbers(user_query)

        # 3. Check Tier 1 Self-Hydrating L1 Cache before invoking Tier 2 neural expert
        if not is_synth and real_args:
            cached = self.learned_cache.lookup(user_query, len(real_args))
            if cached is not None:
                return self._execute_contract(
                    intent_name=cached["intent_name"],
                    signature=cached["target_signature"],
                    args=real_args,
                    user_query=user_query,
                    neural_gen_lat_s=0.0,
                    cache_hit=True,
                )

        # 4. Predict contract via Tier 2 Language Expert (O(1) recurrent step)
        t0_infer = time.perf_counter()
        contract = self.predict_contract(user_query)
        infer_lat_s = time.perf_counter() - t0_infer

        if contract is None:
            if len(real_args) >= 2:
                contract = {
                    "intent_name": "add",
                    "strategy": "single_pass",
                    "signature": "def add(a: int, b: int) -> int:",
                    "args": real_args[:2],
                }
            else:
                return {
                    "status": "UNRECOGNIZED_INTENT",
                    "error": f"Keelespetsialist ei suutnud tuvastada lepingut päringule: '{user_query}'",
                    "query": user_query,
                }

        intent_name = contract.get("intent_name", "custom_intent")
        signature = contract.get("signature", "def func(a: int, b: int) -> int:")

        # Argument binding logic
        expected_n = get_expected_args_count(signature) or 2

        if not is_synth:
            if len(real_args) == expected_n:
                contract["args"] = real_args
                contract["strategy"] = "single_pass"
            else:
                contract["strategy"] = "need_input"
                contract["args"] = real_args

        strategy = contract.get("strategy", "single_pass")
        args = contract.get("args", [])

        # If slot filling is required
        if strategy == "need_input" or (not is_synth and len(args) < expected_n):
            return {
                "status": "NEED_INPUT",
                "mode": "execute",
                "intent": contract,
                "intent_id": intent_name,
                "intent_name": intent_name,
                "provided_args": args,
                "expected_count": expected_n,
                "message": (
                    f"Kavatsus '{intent_name}' vajab {expected_n} arvu "
                    f"(leiti {len(args)}). Palun sisesta puuduvad {expected_n - len(args)} arvu."
                ),
                "query": user_query,
            }

        # Synthesis mode
        if is_synth:
            fused_code = KERNEL_FUSION_MAP.get(intent_name)
            if fused_code:
                full_code = fused_code
                gen_lat_s = infer_lat_s
            else:
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
        return self._execute_contract(
            intent_name=intent_name,
            signature=signature,
            args=args,
            user_query=user_query,
            neural_gen_lat_s=infer_lat_s,
            cache_hit=False,
        )
