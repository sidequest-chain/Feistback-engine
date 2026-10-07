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
from typing import Any, Dict, List, Optional, Tuple

from runner.engine import FeistbackEngine, EngineResult


class BaseDispatcher(ABC):
    """
    Abstract Base Class defining the contract for intent dispatchers.
    """

    @abstractmethod
    def dispatch(self, user_query: str) -> Dict[str, Any]:
        """
        Parses a user query, identifies intent and arguments, and executes
        the corresponding neuro-symbolic generation and VM verification.

        Args:
            user_query: Free-form natural language or command string.

        Returns:
            Dict[str, Any]: Execution status, intent metadata, generated code,
                            and deterministic MiniStackVM result.
        """
        pass


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

    def extract_numbers(self, text: str) -> List[int]:
        """
        Extracts integer values (including negative numbers and hex format)
        from a raw query string.

        Args:
            text: Input text containing potential numeric arguments.

        Returns:
            List[int]: Ordered list of extracted integer values.
        """
        # Match hexadecimal numbers (0x...) or decimal integers (with optional negative sign)
        pattern = r"(?:0x[0-9a-fA-F]+|-?\d+)"
        matches = re.findall(pattern, text)
        numbers: List[int] = []
        for m in matches:
            if m.startswith(("0x", "0X")):
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

    def dispatch(self, user_query: str) -> Dict[str, Any]:
        """
        Resolves query intent, extracts numeric parameters, executes
        PureFeistelLM generation, and runs deterministic verification on MiniStackVM.
        """
        args = self.extract_numbers(user_query)
        intent, score = self.match_intent(user_query)

        if intent is None:
            return {
                "status": "ERROR",
                "error": "No matching intent recognized for the input query.",
                "query": user_query,
                "score": score,
            }

        expected_count = intent.get("expected_args_count", 0)
        if len(args) < expected_count:
            return {
                "status": "ERROR",
                "error": f"Insufficient arguments: intent '{intent['id']}' expects {expected_count} arguments, but found {len(args)}: {args}",
                "query": user_query,
                "intent_id": intent["id"],
                "args_found": args,
                "score": score,
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
            "query": user_query,
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
