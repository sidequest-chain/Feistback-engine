"""
Council of Specialties Intent Dispatcher (Two-Layer Cache & Confidence Gate)
============================================================================
Couples the Layer A Semantic Cache (`LearnedIntentCache` with Confidence Margin Gate
and `model_hash` binding) and the specialized `eesti_keele_sl2ng` micro-model with
the PureFeistelLM logic engine, Layer B JIT `BytecodeCache`, and `MiniStackVM`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import torch
from tokenizers import Tokenizer

from dispatchers.base import BaseDispatcher
from dispatchers.cache_manager import LearnedIntentCache
from dispatchers.parsers.et_numbers import (
    extract_numbers,
    is_synthesis_query,
)

from runtime.bytecode_cache import compute_file_sha256
from runtime.engine import FeistbackEngine
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

CONFIRMATION_WORDS = {"jah", "jaa", "jep", "okei", "ok", "yes", "ja", "davai", "sobib"}


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
    CouncilDispatcher routes queries through the Layer A `LearnedIntentCache`
    and Tier 2 `eesti_keele_sl2ng` micro-model, enforcing a Confidence Margin Gate
    (`min_margin`) to prevent semantic cache poisoning and hallucinated execution.
    """

    def __init__(
        self,
        engine: FeistbackEngine,
        model_name: str = "eesti_keele_sl2ng",
        registry: Optional[ModelRegistry] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        cache_path: str = "cache/learned_intents.json",
        min_cache_margin: float = 0.85,
        min_exec_margin: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = registry or ModelRegistry()
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.min_cache_margin = min_cache_margin
        self.min_exec_margin = min_exec_margin
        self.expert_model: Optional[PureFeistelLM] = None
        self.expert_tokenizer: Optional[Tokenizer] = None
        self.expert_model_hash: str = "default"

        self._load_expert()
        self.learned_cache = LearnedIntentCache(
            cache_path=cache_path,
            model_hash=self.expert_model_hash,
            min_confidence_margin=self.min_cache_margin,
        )

    def _load_expert(self) -> None:
        """Loads eesti_keele_sl2ng micro-model weights and tokenizer via ModelRegistry."""
        try:
            if self.model_name in self.registry._models:
                self.expert_model, cfg = self.registry.load_model(self.model_name, device=self.device)
                self.expert_tokenizer = self.registry.load_tokenizer(self.model_name)
                weights_file = Path(cfg.get("_model_dir", "")) / cfg.get("weights", "")
                self.expert_model_hash = compute_file_sha256(weights_file)
            elif self.model_path and self.tokenizer_path:
                mp = Path(self.model_path)
                if not mp.is_absolute():
                    mp = Path(__file__).resolve().parent.parent / self.model_path
                tp = Path(self.tokenizer_path)
                if not tp.is_absolute():
                    tp = Path(__file__).resolve().parent.parent / self.tokenizer_path
                self.expert_tokenizer = Tokenizer.from_file(str(tp))
                self.expert_model_hash = compute_file_sha256(mp)
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
        to generate structured contract JSON with zero KV-cache, while tracking
        token-level Top-1 probability (`min_prob`) and Top-1 vs Top-2 margin (`min_margin`).
        """
        if self.expert_model is None or self.expert_tokenizer is None:
            return None

        prompt_str = f"<|user|> {user_query} <|contract|>"
        prompt_ids = self.expert_tokenizer.encode(prompt_str).ids
        prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
        eos_id = self.expert_tokenizer.token_to_id("<|endoftext|>")

        gen_ids: List[int] = []
        step_probs: List[float] = []
        step_margins: List[float] = []
        max_steps = min(96, self.expert_model.max_seq_len - len(prompt_ids))

        with torch.no_grad():
            # 1. Prefill prompt once to initialize recurrent RAF + spectral history states
            logits, layer_states = self.expert_model(
                prompt_tensor,
                states=None,
                return_states=True,
            )

            # 2. Step token-by-token (L=1) with O(1) state updates & confidence tracking
            for _ in range(max_steps):
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                top2 = torch.topk(probs, k=2, dim=-1)
                p1 = top2.values[0, 0].item()
                p2 = top2.values[0, 1].item()
                next_tok = top2.indices[:, :1]
                tid = next_tok.item()

                if tid == eos_id:
                    break

                gen_ids.append(tid)
                step_probs.append(p1)
                step_margins.append(p1 - p2)

                logits, layer_states = self.expert_model(
                    next_tok,
                    states=layer_states,
                    return_states=True,
                )

        raw_text = self.expert_tokenizer.decode(gen_ids).strip()

        # Parse outermost JSON object
        start_brace = raw_text.find("{")
        end_brace = raw_text.rfind("}")
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            try:
                contract = json.loads(raw_text[start_brace : end_brace + 1])
                # Focus confidence measurement on the intent decision region (first 16 tokens)
                decision_probs = step_probs[:16] if step_probs else [0.0]
                decision_margins = step_margins[:16] if step_margins else [0.0]
                contract["confidence"] = round(min(decision_probs), 4)
                contract["confidence_margin"] = round(min(decision_margins), 4)
                return contract
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
        confidence_margin: float = 1.0,
    ) -> Dict[str, Any]:
        """Executes code via neural reasoning, Layer B JIT BytecodeCache, or kernel fusion on MiniStackVM."""
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

                # Double-Lock + Confidence Margin Gate
                self.learned_cache.record_verified(
                    query=user_query,
                    intent_name=intent_name,
                    target_signature=signature,
                    args_count=len(args),
                    verified_by_vm=True,
                    confidence_margin=confidence_margin,
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
                    "bytecode_cache_hit": True,
                    "confidence_margin": confidence_margin,
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

        # Double-Lock + Confidence Margin Gate: persist to Layer A cache only if VM verified AND margin >= threshold
        if engine_res.verified:
            self.learned_cache.record_verified(
                query=user_query,
                intent_name=intent_name,
                target_signature=signature,
                args_count=len(args),
                verified_by_vm=True,
                confidence_margin=confidence_margin,
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
            "bytecode_cache_hit": engine_res.bytecode_cache_hit,
            "confidence_margin": confidence_margin,
            "generated_code": engine_res.code,
            "bytecode": engine_res.bytecode,
            "disassembly": engine_res.disassembly,
            "vm_latency_us": engine_res.latency_vm_avg_us,
            "compilation_latency_ms": engine_res.latency_comp_ms,
            "generation_latency_s": neural_gen_lat_s + engine_res.latency_gen_s,
            "query": user_query,
        }

    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[Union[int, float]]] = None,
    ) -> Dict[str, Any]:
        """Dispatches queries using Layer A Cache, Tier 2 neural language expert with Confidence Gate, or pending state."""
        # 1. Resolve pending intent or user confirmation
        if pending_intent is not None:
            clean_reply = user_query.strip().lower()
            if pending_intent.get("_awaiting_confirmation") and clean_reply in CONFIRMATION_WORDS:
                signature = pending_intent.get("signature", "def add(a: int, b: int) -> int:")
                intent_name = pending_intent.get("intent_name", "calculation")
                orig_query = pending_intent.get("_original_query", user_query)
                args = pending_args or pending_intent.get("args", [])
                # User explicitly confirmed -> confidence_margin = 1.0
                return self._execute_contract(
                    intent_name=intent_name,
                    signature=signature,
                    args=args,
                    user_query=orig_query,
                    confidence_margin=1.0,
                )

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
            return self._execute_contract(
                intent_name=intent_name,
                signature=signature,
                args=target_args,
                user_query=user_query,
                confidence_margin=float(pending_intent.get("confidence_margin", 1.0)),
            )

        # 2. Check for synthesis request
        is_synth = is_synthesis_query(user_query)
        real_args = extract_numbers(user_query)

        # 3. Check Layer A Learned Cache before invoking Tier 2 neural expert
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
                    confidence_margin=float(cached.get("confidence_margin", 1.0)),
                )

        # 4. Predict contract via Tier 2 Language Expert (O(1) recurrent step + margin measurement)
        t0_infer = time.perf_counter()
        contract = self.predict_contract(user_query)
        infer_lat_s = time.perf_counter() - t0_infer

        if contract is None:
            return {
                "status": "UNRECOGNIZED_INTENT",
                "error": f"Keelespetsialist ei suutnud tuvastada lepingut päringule: '{user_query}'",
                "query": user_query,
            }

        intent_name = contract.get("intent_name", "custom_intent")
        signature = contract.get("signature", "def func(a: int, b: int) -> int:")
        conf_margin = float(contract.get("confidence_margin", 0.0))

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

        # 5. Confidence Gate Check: if model hesitated below min_exec_margin, ask confirmation instead of poisoning!
        if conf_margin < self.min_exec_margin:
            contract["_awaiting_confirmation"] = True
            contract["_original_query"] = user_query
            return {
                "status": "NEED_CONFIRMATION",
                "mode": "execute",
                "route": "CONFIDENCE_GATE_BLOCKED",
                "intent": contract,
                "intent_id": intent_name,
                "intent_name": intent_name,
                "signature": signature,
                "args": args,
                "confidence_margin": conf_margin,
                "cache_hit": False,
                "message": (
                    f"Madal mudeli kindlus ({conf_margin * 100:.1f}%). "
                    f"Kas pidasid silmas tehet '{intent_name}' ({signature}) argumentidega {args}?"
                ),
                "query": user_query,
            }

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
                "confidence_margin": conf_margin,
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
                "confidence_margin": conf_margin,
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
            confidence_margin=conf_margin,
        )
