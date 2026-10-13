"""
Fast-Path Declarative Config Intent Dispatcher (Tier 1)
=======================================================
Matches user queries against static/declarative rules defined in config/intents.json
using weighted regex and keyword scoring. Executes within sub-millisecond latencies.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from dispatchers.base import BaseDispatcher
from dispatchers.parsers.et_numbers import (
    extract_numbers,
    is_synthesis_query,
)

from runtime.canonicalizer import canonicalize_signature, decanonicalize_code
from runtime.engine import FeistbackEngine, EngineResult


class ConfigDispatcher(BaseDispatcher):
    """
    Configuration-driven Intent Dispatcher matching natural language and
    estengrish queries to predefined algorithmic signatures using weighted
    keyword and regular expression scoring.
    """

    def __init__(
        self,
        engine: FeistbackEngine,
        config_path: str = "config/intents.json",
    ) -> None:
        self.engine = engine
        self.config_path = config_path
        self.intents: List[Dict[str, Any]] = []
        self._load_config()

    def _load_config(self) -> None:
        """Loads and precompiles intent patterns from JSON configuration."""
        resolved_path = Path(self.config_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).resolve().parent.parent / self.config_path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Intent configuration file not found at: {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.intents = data.get("intents", [])
        for intent in self.intents:
            compiled = []
            for pattern_str in intent.get("regex_patterns", []):
                try:
                    compiled.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error as err:
                    print(f"Warning: Failed to compile regex '{pattern_str}': {err}")
            intent["_compiled_regex"] = compiled

    def is_synthesis_query(self, text: str) -> bool:
        """Determines whether a query requests pure code/script synthesis without immediate execution."""
        return is_synthesis_query(text)

    def extract_numbers(self, text: str) -> List[Union[int, float]]:
        """Extracts integer and float arguments from query string."""
        return extract_numbers(text)

    def match_intent(self, text: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Matches free-form text against registered intents using weighted scoring.
        """
        clean_text = text.lower().strip()
        best_intent: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for intent in self.intents:
            score = 0.0

            # 1. Regex scoring (high confidence: +10.0 points per match)
            for regex in intent.get("_compiled_regex", []):
                if regex.search(clean_text):
                    score += 10.0

            # 2. Keyword scoring (+2.0 points per keyword; +3.0 if multi-word)
            for kw in intent.get("keywords", []):
                kw_clean = kw.lower().strip()
                if kw_clean in clean_text:
                    score += 3.0 if " " in kw_clean else 2.0

            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score < 1.0:
            return None, 0.0

        return best_intent, best_score

    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[Union[int, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Resolves query intent, extracts numeric parameters, executes
        PureFeistelLM generation, and runs deterministic verification on MiniStackVM.
        """
        # Case 1: Active pending intent awaiting missing arguments
        if pending_intent is not None:
            intent = pending_intent
            new_numbers = self.extract_numbers(user_query)
            args = (pending_args or []) + new_numbers
            expected_count = intent.get("expected_args_count", 0)

            if len(args) < expected_count:
                return {
                    "status": "NEED_INPUT",
                    "mode": "execute",
                    "intent": intent,
                    "intent_id": intent["id"],
                    "intent_name": intent.get("name", intent["id"]),
                    "provided_args": args,
                    "expected_count": expected_count,
                    "message": (
                        f"Tuvastasin kavatsuse '{intent.get('name', intent['id'])}'. "
                        f"Vajatakse {expected_count} arvu, kuid leiti {len(args)}. Milliseid arve soovid kasutada?"
                    ),
                    "query": user_query,
                }

            # Sufficient arguments collected: execute immediately
            target_args = args[:expected_count]
            signature = intent["signature_template"]
            engine_res: EngineResult = self.engine.generate_and_verify(
                prompt=signature,
                test_inputs=[target_args],
            )
            vm_result = None
            if engine_res.test_outcomes:
                vm_result = engine_res.test_outcomes[0].vm_result

            status = "SUCCESS" if engine_res.verified else "VERIFICATION_FAILED"
            return {
                "status": status,
                "mode": "execute",
                "intent_id": intent["id"],
                "intent_name": intent.get("name", intent["id"]),
                "args": target_args,
                "result": vm_result,
                "verified": engine_res.verified,
                "generated_code": engine_res.code,
                "bytecode": engine_res.bytecode,
                "disassembly": engine_res.disassembly,
                "vm_latency_us": engine_res.latency_vm_avg_us,
                "generation_latency_s": engine_res.latency_gen_s,
                "compilation_latency_ms": engine_res.latency_comp_ms,
                "query": user_query,
            }

        # Case 2: New incoming query
        intent, score = self.match_intent(user_query)
        if intent is None:
            return {
                "status": "UNRECOGNIZED_INTENT",
                "error": f"Ei suutnud tuvastada kavatsust päringule: '{user_query}'",
                "query": user_query,
                "score": score,
            }

        # Sub-case 2A: Pure code synthesis requested
        if self.is_synthesis_query(user_query):
            signature = intent["signature_template"]
            canonical_sig, inv_map = canonicalize_signature(signature)
            t0_gen = time.perf_counter()
            gen_raw, _ = self.engine.model.generate(
                prompt=canonical_sig if inv_map else signature,
                max_new_tokens=64,
                temperature=0.0,
                early_stopping=True,
            )
            gen_lat_s = time.perf_counter() - t0_gen

            body = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
            indented = "\n".join("    " + line.strip() for line in body.split("\n") if line.strip())
            full_code = signature.strip() + "\n" + indented

            # Validate syntax through ASTCompiler
            try:
                self.engine.compiler.compile(full_code)
                is_valid = True
            except Exception:
                is_valid = False

            return {
                "status": "SUCCESS",
                "mode": "synthesize",
                "intent_id": intent["id"],
                "intent_name": intent.get("name", intent["id"]),
                "signature": signature,
                "generated_code": full_code,
                "generation_latency_s": gen_lat_s,
                "message": f"Genereeritud puhas funktsioonikood kavatsusele '{intent.get('name', intent['id'])}'.",
                "query": user_query,
            }

        # Sub-case 2B: Standard execution mode
        args = self.extract_numbers(user_query)
        expected_count = intent.get("expected_args_count", 0)

        # Check for missing arguments (Slot-filling required)
        if len(args) < expected_count:
            return {
                "status": "NEED_INPUT",
                "mode": "execute",
                "intent": intent,
                "intent_id": intent["id"],
                "intent_name": intent.get("name", intent["id"]),
                "provided_args": args,
                "expected_count": expected_count,
                "message": (
                    f"Kavatsus '{intent.get('name', intent['id'])}' vajab {expected_count} arvu "
                    f"(leiti {len(args)}). Palun sisesta puuduvad {expected_count - len(args)} arvu."
                ),
                "query": user_query,
            }

        target_args = args[:expected_count]
        signature = intent["signature_template"]

        engine_res = self.engine.generate_and_verify(
            prompt=signature,
            test_inputs=[target_args],
        )

        vm_result = None
        if engine_res.test_outcomes:
            vm_result = engine_res.test_outcomes[0].vm_result

        status = "SUCCESS" if engine_res.verified else "VERIFICATION_FAILED"

        return {
            "status": status,
            "mode": "execute",
            "intent_id": intent["id"],
            "intent_name": intent.get("name", intent["id"]),
            "args": target_args,
            "result": vm_result,
            "verified": engine_res.verified,
            "generated_code": engine_res.code,
            "bytecode": engine_res.bytecode,
            "disassembly": engine_res.disassembly,
            "vm_latency_us": engine_res.latency_vm_avg_us,
            "generation_latency_s": engine_res.latency_gen_s,
            "compilation_latency_ms": engine_res.latency_comp_ms,
            "query": user_query,
        }
