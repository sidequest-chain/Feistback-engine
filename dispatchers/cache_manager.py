"""
Layer A Semantic Intent Cache Manager (RFC-002 Tier 1)
======================================================
Provides an LRU-bounded cache (`cache/learned_intents.json`) for storing natural
language query patterns resolved by the Tier 2 Language Expert (`eesti_keele_sl2ng`).

Protections against Cache Poisoning & Ecosystem Drift:
1. Model Hash Binding (`model_hash`): Automatically invalidates cached patterns
   whenever the specialist `.pt`/`.feist` model weights are updated.
2. Confidence Margin Gate (`min_confidence_margin`): Refuses to cache predictions
   where the neural model hesitated between top-1 and top-2 token choices.
3. VM Verification Gate (`verified_by_vm`): Refuses to cache unless `MiniStackVM`
   confirmed deterministic execution parity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional


_NUM_TOKEN_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?|0x[0-9a-fA-F]+", re.IGNORECASE)


def normalize_query_pattern(query: str) -> str:
    """
    Normalizes a user query into a canonical structural pattern by replacing
    numeric literals with `<num>` placeholders and collapsing whitespace.
    """
    clean = query.lower().strip()
    masked = _NUM_TOKEN_PATTERN.sub("<num>", clean)
    return " ".join(masked.split())


def compute_pattern_hash(pattern: str, args_count: int) -> str:
    """Computes a compact 10-character SHA-256 hex digest for a structural query pattern."""
    payload = f"{pattern}::{args_count}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


class LearnedIntentCache:
    """
    LRU-bounded Layer A intent cache backed by `cache/learned_intents.json`.
    Enforces both the Confidence Margin Gate (`confidence_margin >= min_confidence_margin`)
    and the VM Verification Gate (`verified_by_vm=True`), bound to `model_hash`.
    """

    def __init__(
        self,
        cache_path: str = "cache/learned_intents.json",
        max_entries: int = 500,
        model_hash: str = "default",
        min_confidence_margin: float = 0.85,
    ) -> None:
        p = Path(cache_path)
        if not p.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            p = repo_root / cache_path
        self.cache_path = p
        self.max_entries = max_entries
        self.model_hash = model_hash
        self.min_confidence_margin = min_confidence_margin
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Loads cached intent entries from disk if present and matching `model_hash`."""
        self._entries.clear()
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stored_hash = data.get("model_hash", "default")
            if self.model_hash != "default" and stored_hash != "default" and stored_hash != self.model_hash:
                # Specialist model changed; invalidate stale learned intents
                return
            self.max_entries = int(data.get("max_entries", self.max_entries))
            for entry in data.get("entries", []):
                q_hash = entry.get("query_hash")
                margin = float(entry.get("confidence_margin", 1.0))
                if q_hash and entry.get("verified_by_vm") is True and margin >= self.min_confidence_margin:
                    self._entries[q_hash] = entry
        except Exception:
            self._entries.clear()

    def _save(self) -> None:
        """Persists cache entries to disk atomically."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            sorted_entries = sorted(
                self._entries.values(),
                key=lambda e: e.get("last_accessed", 0),
                reverse=True,
            )[: self.max_entries]
            payload = {
                "model_hash": self.model_hash,
                "min_confidence_margin": self.min_confidence_margin,
                "max_entries": self.max_entries,
                "entries": sorted_entries,
            }
            tmp_path = self.cache_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self.cache_path)
        except Exception:
            pass

    def lookup(self, query: str, args_count: int) -> Optional[Dict[str, Any]]:
        """
        Looks up a user query in the L1 learned cache.
        Updates `last_accessed` and `hit_count` on hit.
        """
        pattern = normalize_query_pattern(query)
        q_hash = compute_pattern_hash(pattern, args_count)
        entry = self._entries.get(q_hash)
        if entry is None or not entry.get("verified_by_vm", False):
            return None

        entry["last_accessed"] = int(time.time())
        entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
        self._save()
        return entry

    def record_verified(
        self,
        query: str,
        intent_name: str,
        target_signature: str,
        args_count: int,
        verified_by_vm: bool,
        confidence_margin: float = 1.0,
        execution_strategy: str = "single_pass",
    ) -> bool:
        """
        Persists a newly resolved intent pattern into the L1 cache strictly if:
        1. `verified_by_vm` is True, AND
        2. `confidence_margin >= self.min_confidence_margin` (Confidence Gate).
        """
        if not verified_by_vm:
            return False
        if confidence_margin < self.min_confidence_margin:
            return False

        pattern = normalize_query_pattern(query)
        q_hash = compute_pattern_hash(pattern, args_count)
        now_ts = int(time.time())

        if q_hash in self._entries:
            existing = self._entries[q_hash]
            existing["last_accessed"] = now_ts
            existing["hit_count"] = int(existing.get("hit_count", 0)) + 1
            existing["confidence_margin"] = round(float(confidence_margin), 4)
        else:
            if len(self._entries) >= self.max_entries:
                oldest_key = min(
                    self._entries.keys(),
                    key=lambda k: self._entries[k].get("last_accessed", 0),
                )
                del self._entries[oldest_key]

            self._entries[q_hash] = {
                "query_hash": q_hash,
                "matched_pattern": pattern,
                "intent_name": intent_name,
                "target_signature": target_signature,
                "execution_strategy": execution_strategy,
                "args_count": args_count,
                "confidence_margin": round(float(confidence_margin), 4),
                "last_accessed": now_ts,
                "hit_count": 1,
                "verified_by_vm": True,
            }

        self._save()
        return True
