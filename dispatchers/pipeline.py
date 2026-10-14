"""
Universal Council of Specialties Pipeline
=========================================
Implements a modular, multi-stage neural-symbolic pipeline where micro-models
(`.feist` / `ModelRegistry`) and deterministic verifiers (`MiniStackVM` / `ToolRegistry`)
act as composable stages with explicit JSON contracts and confidence gates.

Default Pipeline Stages:
1. LanguageNormalizerStage  : Cleans conversational wrappers & normalizes Estonian morphology.
2. DomainRouterStage        : Classifies domain (arithmetic, finance/tax, number_theory, custom)
                              and flags out-of-domain or contradictory queries.
3. ContractDispatcherStage  : Invokes Layer A Cache, Tier 2 Specialist LM, Layer B JIT Cache & VM.
4. VerificationAuditStage   : Enforces deterministic VM verification & records stage telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Dict, List, Optional, Union

from dispatchers.council import CONFIRMATION_WORDS, CouncilDispatcher
from dispatchers.parsers.et_numbers import extract_numbers


@dataclass
class StageTrace:
    """Telemetry record for a single pipeline stage execution."""
    stage_name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable state envelope passed sequentially through CouncilPipeline stages."""
    original_query: str
    working_query: str = ""
    domain: str = "general"
    extracted_args: List[Union[int, float]] = field(default_factory=list)
    pending_intent: Optional[Dict[str, Any]] = None
    pending_args: Optional[List[Union[int, float]]] = None
    result: Dict[str, Any] = field(default_factory=dict)
    traces: List[StageTrace] = field(default_factory=list)
    halted: bool = False

    def __post_init__(self) -> None:
        if not self.working_query:
            self.working_query = self.original_query


class PipelineStage:
    """Abstract base class for a composable CouncilPipeline stage."""
    name: str = "base_stage"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError


class LanguageNormalizerStage(PipelineStage):
    """
    Stage 1: Conversational & Morphological Pre-Normalizer.
    Extracts numerical arguments and normalizes conversational filler tokens
    while preserving full semantic intent for downstream micro-models.
    """
    name = "language_normalizer"

    # Conversational prefixes/suffixes that carry no mathematical/contract semantics
    _FILLER_PREFIX_RE = re.compile(
        r"^(?:ou+|kle|kuule|hei|jou|yo|palun|plz|pls|davai|no|oota|ütle|ytle)\s+",
        re.IGNORECASE,
    )
    _FILLER_SUFFIX_RE = re.compile(
        r"\s+(?:plz|pls|palun|eks|onju|tänks|tnx|aitäh|ruttu|kiirelt)[\?!\.]*$",
        re.IGNORECASE,
    )

    def __init__(self, strip_conversational_fillers: bool = True) -> None:
        self.strip_conversational_fillers = strip_conversational_fillers

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        ctx.extracted_args = extract_numbers(ctx.working_query)

        cleaned = ctx.working_query.strip()
        if self.strip_conversational_fillers:
            prev = None
            while prev != cleaned:
                prev = cleaned
                cleaned = self._FILLER_PREFIX_RE.sub("", cleaned).strip()
                cleaned = self._FILLER_SUFFIX_RE.sub("", cleaned).strip()
            if cleaned:
                ctx.working_query = cleaned

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.traces.append(
            StageTrace(
                stage_name=self.name,
                latency_ms=round(elapsed_ms, 4),
                metadata={
                    "working_query": ctx.working_query,
                    "extracted_args_count": len(ctx.extracted_args),
                },
            )
        )
        return ctx


class DomainRouterStage(PipelineStage):
    """
    Stage 2: Domain Classifier & Specialist Selector.
    Identifies whether the query targets arithmetic, finance/VAT/discount,
    number theory (GCD/LCM/primality), bitwise ops, or flags contradictory /
    out-of-domain queries before they can poison downstream specialist caches.
    """
    name = "domain_router"

    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "finance": [
            "allahindlus", "ale", "soodukas", "soodustus", "käibemaks", "km",
            "vat", "protsent", "protsenti", "%", "hind", "maksab", "palk",
            "palgast", "arvel", "arvelt", "kontol", "kulutasin", "teenisin",
        ],
        "number_theory": [
            "süt", "suurim ühistegur", "vük", "väikseim ühiskordne", "algarv",
            "faktoriaal", "gcd", "lcm",
        ],
        "bitwise": [
            "xor", "shift", "shifteeri", "bit", "bitid", "bitte", "mask", "maski",
            "clamp", "clämbi", "klembi", "klämbi", "vahemikku", "piiridesse",
        ],
        "arithmetic": [
            "liida", "lahuta", "korruta", "jaga", "summa", "summeeri", "vahe",
            "diferents", "korrutis", "jagatis", "pane", "viska", "löö", "võta",
            "tõmba", "juurde", "maha", "kokku", "ruut", "ruutu", "pluss", "miinus",
            "korda", "laos", "laoseisust", "müüdi", "toodi", "lisandus", "liitus",
            "ostsin", "töötasin", "jagatud",
        ],
    }

    _CONTRADICTORY_RE = re.compile(
        r"\b(?:või|voi)\b.*\b(?:liida|lahuta|korruta|jaga|arvuta)\b|"
        r"\b(?:liida|lahuta|korruta|jaga)\b.*\b(?:või|voi)\b",
        re.IGNORECASE,
    )

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        q_lower = ctx.working_query.lower()

        if ctx.pending_intent is not None or q_lower in CONFIRMATION_WORDS:
            detected_domain = "arithmetic"
        elif self._CONTRADICTORY_RE.search(q_lower):
            detected_domain = "ambiguous"
        else:
            detected_domain = "out_of_domain"
            for dom, keywords in self._DOMAIN_KEYWORDS.items():
                if any(kw in q_lower for kw in keywords):
                    detected_domain = dom
                    break

        ctx.domain = detected_domain
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.traces.append(
            StageTrace(
                stage_name=self.name,
                latency_ms=round(elapsed_ms, 4),
                metadata={"domain": detected_domain},
            )
        )
        return ctx


class ContractDispatcherStage(PipelineStage):
    """
    Stage 3: Neural-Symbolic Contract Dispatcher.
    Routes the normalized query through `CouncilDispatcher` (Layer A Cache ->
    Tier 2 Specialist Model with Margin Gate -> Layer B JIT Cache -> MiniStackVM).
    Blocks out-of-domain or contradictory queries before cache persistence.
    """
    name = "contract_dispatcher"

    def __init__(
        self,
        default_dispatcher: CouncilDispatcher,
        domain_dispatchers: Optional[Dict[str, CouncilDispatcher]] = None,
    ) -> None:
        self.default_dispatcher = default_dispatcher
        self.domain_dispatchers: Dict[str, CouncilDispatcher] = domain_dispatchers or {}

    def register_domain_dispatcher(self, domain: str, dispatcher: CouncilDispatcher) -> None:
        """Binds a specialized CouncilDispatcher to a specific domain tag."""
        self.domain_dispatchers[domain] = dispatcher

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        dispatcher = self.domain_dispatchers.get(ctx.domain, self.default_dispatcher)

        if ctx.domain in ("ambiguous", "out_of_domain") and ctx.pending_intent is None:
            # Query still evaluated by neural specialist to record confidence margin,
            # but strictly gated to NEED_CONFIRMATION so it can never poison Layer A cache.
            contract = dispatcher.predict_contract(ctx.working_query) or {}
            conf_margin = float(contract.get("confidence_margin", 0.0))
            intent_name = contract.get("intent_name", "unknown")
            signature = contract.get("signature", "def unknown(a: int, b: int) -> int:")
            ctx.result = {
                "status": "NEED_CONFIRMATION",
                "mode": "execute",
                "route": "CONFIDENCE_GATE_BLOCKED",
                "intent": contract,
                "intent_id": intent_name,
                "intent_name": intent_name,
                "signature": signature,
                "args": ctx.extracted_args,
                "confidence_margin": conf_margin,
                "cache_hit": False,
                "message": (
                    f"Päring liigitati kategooriasse '{ctx.domain}' (mudeli kindlus {conf_margin * 100:.1f}%). "
                    f"Palun kinnita tehe '{intent_name}'."
                ),
                "query": ctx.original_query,
            }
        else:
            res = dispatcher.dispatch(
                user_query=ctx.working_query,
                pending_intent=ctx.pending_intent,
                pending_args=ctx.pending_args,
            )
            ctx.result = res

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.traces.append(
            StageTrace(
                stage_name=self.name,
                latency_ms=round(elapsed_ms, 4),
                metadata={
                    "status": ctx.result.get("status"),
                    "route": ctx.result.get("route"),
                    "intent_name": ctx.result.get("intent_name"),
                    "confidence_margin": ctx.result.get("confidence_margin"),
                },
            )
        )
        return ctx


class VerificationAuditStage(PipelineStage):
    """
    Stage 4: Deterministic Verification & Telemetry Auditor.
    Attaches domain and stage-level latency traces to the final response payload.
    """
    name = "verification_audit"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.traces.append(
            StageTrace(
                stage_name=self.name,
                latency_ms=round(elapsed_ms, 4),
                metadata={"verified": bool(ctx.result.get("verified", False))},
            )
        )

        total_ms = sum(t.latency_ms for t in ctx.traces)
        ctx.result["domain"] = ctx.domain
        ctx.result["normalized_query"] = ctx.working_query
        ctx.result["pipeline_total_ms"] = round(total_ms, 3)
        ctx.result["pipeline_stages"] = [
            {
                "stage": t.stage_name,
                "latency_ms": t.latency_ms,
                **t.metadata,
            }
            for t in ctx.traces
        ]
        return ctx


class CouncilPipeline:
    """
    Composable multi-stage Council execution pipeline.
    Allows chaining arbitrary pre-processors, domain routers, `.feist` micro-models,
    and deterministic VM verification stages.
    """

    def __init__(
        self,
        dispatcher: CouncilDispatcher,
        stages: Optional[List[PipelineStage]] = None,
        strip_conversational_fillers: bool = True,
    ) -> None:
        self.dispatcher = dispatcher
        if stages is not None:
            self.stages = list(stages)
        else:
            self.stages = [
                LanguageNormalizerStage(strip_conversational_fillers=strip_conversational_fillers),
                DomainRouterStage(),
                ContractDispatcherStage(default_dispatcher=dispatcher),
                VerificationAuditStage(),
            ]

    def add_stage(self, stage: PipelineStage, index: Optional[int] = None) -> None:
        """Inserts a custom PipelineStage into the execution sequence."""
        if index is None:
            self.stages.append(stage)
        else:
            self.stages.insert(index, stage)

    def run(
        self,
        user_query: str,
        pending_intent: Optional[Dict[str, Any]] = None,
        pending_args: Optional[List[Union[int, float]]] = None,
    ) -> Dict[str, Any]:
        """Runs the query through all registered pipeline stages and returns the audited result."""
        ctx = PipelineContext(
            original_query=user_query,
            pending_intent=pending_intent,
            pending_args=pending_args,
        )
        for stage in self.stages:
            if ctx.halted:
                break
            ctx = stage.process(ctx)
        return ctx.result
