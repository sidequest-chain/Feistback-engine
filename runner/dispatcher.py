"""
Feistback Intent Dispatcher & Routing Layer
===========================================
Defines the BaseDispatcher abstract contract and ConfigDispatcher implementation
for mapping free-form natural language and 'estengrish' user queries into
formal function signatures and executing them via FeistbackEngine on MiniStackVM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from runner.engine import FeistbackEngine, EngineResult
from scripts.run_pure_feistel import canonicalize_signature, decanonicalize_code


class BaseDispatcher(ABC):
    """
    Abstract Base Class defining the contract for intent dispatchers.
    """

    @abstractmethod
    def dispatch(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Parses a user query, identifies intent and arguments, and executes
        the corresponding neuro-symbolic generation and VM verification.

        Args:
            user_query: Free-form natural language or command string.
            pending_intent: Optional pending intent awaiting missing arguments.
            pending_args: Optional list of previously collected arguments.

        Returns:
            Dict[str, Any]: Execution status, intent metadata, generated code,
                            and deterministic MiniStackVM result.
        """
        pass


# Mapping of Estonian number words and common inflected forms to integers (0-10)
ESTONIAN_WORDS_TO_INT: Dict[str, int] = {
    "null": 0,
    "üks": 1, "ühe": 1,
    "kaks": 2, "kahe": 2,
    "kolm": 3, "kolme": 3,
    "neli": 4, "nelja": 4,
    "viis": 5, "viie": 5,
    "kuus": 6, "kuue": 6,
    "seitse": 7, "seitsme": 7,
    "kaheksa": 8,
    "üheksa": 9,
    "kümme": 10, "kümne": 10,
}


# Regular expressions and keywords identifying pure code synthesis requests (no VM execution needed)
SYNTHESIS_PATTERNS = [
    r"\bloo\b.*\bskript\b",
    r"\btee\b.*\bskript\b",
    r"\bkirjuta\b.*\bkood\b",
    r"\bkirjuta\b.*\bskript\b",
    r"\bnäita\b.*\bkoodi?\b",
    r"\bloo\b.*\bfunktsioon\b",
    r"\btee\b.*\bfunktsioon\b",
    r"\bkirjuta\b.*\bfunktsioon\b",
    r"\bkuva\b.*\bkoodi?\b",
    r"\bgenerate\b.*\bcode\b",
    r"\bwrite\b.*\bscript\b",
    r"\bshow\b.*\bcode\b",
    r"\bcreate\b.*\bfunction\b",
    r"\bpythoni\s+skript\b",
    r"\bpythoni\s+kood\b",
    r"\bkoodi\s+süntees\b",
    r"\bsünteesi\b",
]


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

        # Build unified scanner regex for hex, signed decimals, and Estonian number words
        words_regex = "|".join(re.escape(w) for w in sorted(ESTONIAN_WORDS_TO_INT.keys(), key=len, reverse=True))
        self._number_pattern = re.compile(rf"(?:0x[0-9a-fA-F]+|-?\d+|\b(?:{words_regex})\b)", re.IGNORECASE)

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
        clean = text.lower().strip()
        return any(re.search(pat, clean) for pat in SYNTHESIS_PATTERNS)

    def extract_numbers(self, text: str) -> List[int]:
        """
        Extracts integer values (including negative numbers, hex format, and
        Estonian number words) from a raw query string in strict left-to-right order.

        Args:
            text: Input text containing potential numeric arguments.

        Returns:
            List[int]: Ordered list of extracted integer values.
        """
        matches = self._number_pattern.findall(text)
        numbers: List[int] = []
        for m in matches:
            m_lower = m.lower()
            if m_lower in ESTONIAN_WORDS_TO_INT:
                numbers.append(ESTONIAN_WORDS_TO_INT[m_lower])
            elif m.startswith(("0x", "0X")):
                numbers.append(int(m, 16))
            else:
                numbers.append(int(m, 10))
        return numbers

    def match_intent(self, text: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Matches free-form text against registered intents using weighted scoring.

        Args:
            text: User input query.

        Returns:
            Tuple[Optional[Dict[str, Any]], float]: Best matching intent dictionary and score.
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
        pending_args: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Resolves query intent, extracts numeric parameters, executes
        PureFeistelLM generation, and runs deterministic verification on MiniStackVM.
        Supports code synthesis mode and multi-turn pending intent clarification.
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
                    "message": f"Tuvastasin kavatsuse '{intent.get('name', intent['id'])}'. Vajatakse {expected_count} arvu, kuid leiti {len(args)}. Milliseid arve soovid kasutada?",
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
                "query": user_query,
                "intent": intent,
                "intent_id": intent["id"],
                "intent_name": intent.get("name", intent["id"]),
                "signature": signature,
                "args": target_args,
                "result": vm_result,
                "verified": engine_res.verified,
                "generated_code": engine_res.code,
                "disassembly": engine_res.disassembly,
                "generation_latency_s": engine_res.latency_gen_s,
                "compilation_latency_ms": engine_res.latency_comp_ms,
                "vm_latency_us": engine_res.latency_vm_avg_us,
            }

        # Case 2: Fresh query - check mode (synthesis vs execution)
        is_synthesis = self.is_synthesis_query(user_query)
        intent, score = self.match_intent(user_query)

        if intent is None:
            return {
                "status": "ERROR",
                "error": "No matching intent recognized for the input query.",
                "query": user_query,
                "score": score,
            }

        # Mode A: Pure Code Synthesis (mode="synthesize")
        if is_synthesis:
            prompt_clean = intent["signature_template"]
            canonical_prompt, inv_map = canonicalize_signature(prompt_clean)
            prompt_to_gen = canonical_prompt if inv_map else prompt_clean

            t0_gen = time.perf_counter()
            gen_raw, token_ids = self.engine.model.generate(
                prompt=prompt_to_gen,
                max_new_tokens=64,
                temperature=0.0,
                top_p=0.85,
                repetition_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            gen_lat_s = time.perf_counter() - t0_gen

            body_code = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
            indented_body = "\n".join("    " + line.strip() for line in body_code.split("\n") if line.strip())
            full_code = prompt_clean.strip() + "\n" + indented_body

            # Verify syntax with ASTCompiler without VM execution
            compile_err = None
            try:
                bytecode = self.engine.compiler.compile(full_code)
                syntax_valid = True
            except Exception as e:
                syntax_valid = False
                compile_err = str(e)

            if syntax_valid:
                return {
                    "status": "SUCCESS",
                    "mode": "synthesize",
                    "query": user_query,
                    "intent": intent,
                    "intent_id": intent["id"],
                    "intent_name": intent.get("name", intent["id"]),
                    "signature": intent["signature_template"],
                    "generated_code": full_code,
                    "message": f"Genereeritud puhas funktsioonikood kavatsusele '{intent.get('name', intent['id'])}'.",
                    "generation_latency_s": gen_lat_s,
                    "syntax_valid": True,
                }
            else:
                return {
                    "status": "COMPILATION_ERROR",
                    "mode": "synthesize",
                    "query": user_query,
                    "intent": intent,
                    "intent_id": intent["id"],
                    "intent_name": intent.get("name", intent["id"]),
                    "signature": intent["signature_template"],
                    "generated_code": full_code,
                    "error": compile_err,
                    "message": f"Koodi süntaksi verifitseerimine ebaõnnestus: {compile_err}",
                    "generation_latency_s": gen_lat_s,
                    "syntax_valid": False,
                }

        # Mode B: Standard Execution (mode="execute")
        args = self.extract_numbers(user_query)
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
                "message": f"Tuvastasin kavatsuse '{intent.get('name', intent['id'])}'. Vajatakse {expected_count} arvu, kuid leiti {len(args)}. Milliseid arve soovid kasutada?",
                "query": user_query,
            }

        # Take the exact number of required arguments in order
        target_args = args[:expected_count]
        signature = intent["signature_template"]

        # Run neuro-symbolic pipeline
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
            "query": user_query,
            "intent": intent,
            "intent_id": intent["id"],
            "intent_name": intent.get("name", intent["id"]),
            "signature": signature,
            "args": target_args,
            "result": vm_result,
            "verified": engine_res.verified,
            "generated_code": engine_res.code,
            "disassembly": engine_res.disassembly,
            "generation_latency_s": engine_res.latency_gen_s,
            "compilation_latency_ms": engine_res.latency_comp_ms,
            "vm_latency_us": engine_res.latency_vm_avg_us,
        }
