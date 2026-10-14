"""
Feistback Production Engine Interface
=====================================
Unified neuro-symbolic engine coupling PureFeistelLM code generation,
AST canonicalization adapter, ASTCompiler bytecode generation,
Layer B Universal JIT Bytecode Cache, and deterministic MiniStackVM
execution with sub-microsecond verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch

from runtime.bytecode_cache import BytecodeCache, compute_file_sha256
from runtime.canonicalizer import canonicalize_signature, decanonicalize_code
from runtime.compiler import ASTCompiler
from runtime.models.pure_feistel_lm import PureFeistelLM
from runtime.registry import ModelRegistry
from runtime.vm.ministack_vm import MiniStackVM


# Safe restricted builtins for Python reference verification
_SAFE_VERIFY_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


@dataclass
class TestOutcome:
    """Outcome of a single test input execution on MiniStackVM vs Python reference."""
    inputs: List[Any]
    vm_result: Any
    py_result: Any
    matched: bool
    latency_us: float
    error: Optional[str] = None


@dataclass
class EngineResult:
    """Comprehensive structured outcome of a full neuro-symbolic generation and execution cycle."""
    prompt: str
    canonical_prompt: str
    inv_map: Dict[str, str]
    generated_raw: str
    code: str
    bytecode: List[Any]
    disassembly: str
    test_outcomes: List[TestOutcome]
    latency_gen_s: float
    latency_comp_ms: float
    latency_vm_avg_us: float
    verified: bool
    arg_names: List[str] = field(default_factory=list)
    bytecode_cache_hit: bool = False


class FeistbackEngine:
    """
    Production-ready neuro-symbolic reasoner and verification engine.
    """

    def __init__(
        self,
        model: PureFeistelLM,
        compiler: Optional[ASTCompiler] = None,
        vm: Optional[MiniStackVM] = None,
        device: str = "cuda",
        bytecode_cache: Optional[BytecodeCache] = None,
        model_hash: str = "default",
    ) -> None:
        self.model = model
        self.compiler = compiler or ASTCompiler()
        self.vm = vm or MiniStackVM()
        self.device = device
        self.model_hash = model_hash
        self.bytecode_cache = bytecode_cache or BytecodeCache(model_hash=model_hash)

    @classmethod
    def from_registry(
        cls,
        model_name: str = "logic_feistel_clean",
        device: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> "FeistbackEngine":
        """
        Instantiates FeistbackEngine using declarative model configuration from ModelRegistry.
        """
        reg = registry or ModelRegistry()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg = reg.load_model(model_name, device=dev)
        weights_file = Path(cfg.get("_model_dir", "")) / cfg.get("weights", "")
        m_hash = compute_file_sha256(weights_file)
        compiler = ASTCompiler()
        vm = MiniStackVM()
        return cls(
            model=model,
            compiler=compiler,
            vm=vm,
            device=dev,
            model_hash=m_hash,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = "models/logic_feistel_clean/pure_feistel_raf_clean.pt",
        device: Optional[str] = None,
        vocab_map_path: str = "models/logic_feistel_clean/vocab_16k/vocab_map.pt",
        pruned_weights_path: str = "models/logic_feistel_clean/vocab_16k/pruned_weights.pt",
        registry: Optional[ModelRegistry] = None,
    ) -> "FeistbackEngine":
        """
        Instantiates FeistbackEngine from trained weights, resolving via ModelRegistry when possible.
        """
        reg = registry or ModelRegistry()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve via registry if matching a known registered model
        for m in reg.list_models():
            cfg = reg.get_config(m["name"])
            m_weights = os.path.basename(cfg.get("weights", ""))
            if os.path.basename(checkpoint_path) == m_weights or m["name"] in checkpoint_path:
                return cls.from_registry(m["name"], device=dev, registry=reg)

        # Canonical path resolution within models/
        if not os.path.exists(checkpoint_path):
            alt_path = os.path.join("models", "logic_feistel_clean", "pure_feistel_raf_clean.pt")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path

        if not os.path.exists(pruned_weights_path):
            alt_pw = os.path.join("models", "logic_feistel_clean", "vocab_16k", "pruned_weights.pt")
            if os.path.exists(alt_pw):
                pruned_weights_path = alt_pw

        if not os.path.exists(vocab_map_path):
            alt_vm = os.path.join("models", "logic_feistel_clean", "vocab_16k", "vocab_map.pt")
            if os.path.exists(alt_vm):
                vocab_map_path = alt_vm

        model = PureFeistelLM(
            vocab_size=16384,
            d_model=896,
            num_layers=12,
            state_dim=128,
            memory_type="raf",
            torch_dtype=torch.bfloat16,
            device=dev,
            from_pretrained_qwen=False,
            pruned_weights_path=pruned_weights_path,
            vocab_map_path=vocab_map_path,
        )

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=dev)
            model.load_backbone(ckpt)
        else:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist!")

        m_hash = compute_file_sha256(checkpoint_path)
        compiler = ASTCompiler()
        vm = MiniStackVM()
        return cls(
            model=model,
            compiler=compiler,
            vm=vm,
            device=dev,
            model_hash=m_hash,
        )

    def generate_and_verify(
        self,
        prompt: str,
        test_inputs: Optional[List[Any]] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        no_repeat_ngram_size: int = 3,
        use_bytecode_cache: bool = True,
    ) -> EngineResult:
        """
        Executes end-to-end neuro-symbolic code generation and verification:
        1. Canonicalizes prompt signature to high-affinity orthogonal registers.
        2. Checks Layer B JIT Bytecode Cache (`canonical_signature -> bytecode`).
        3. On cache miss, generates code via PureFeistelLM (0 bytes KV cache) and compiles AST.
        4. Executes MiniStackVM and verifies mathematical equivalence.
        5. Stores verified bytecode into Layer B JIT cache.
        """
        # 1. Normalize prompt
        prompt_raw = prompt.replace("\\n", "\n").replace("\\t", "\t").strip()
        if prompt_raw.endswith(":"):
            prompt_clean = prompt_raw + "\n"
        else:
            prompt_clean = prompt_raw

        # 2. Canonicalize signature
        canonical_prompt, inv_map = canonicalize_signature(prompt_clean)
        prompt_to_gen = canonical_prompt
        canonical_sig_key = prompt_to_gen.strip()

        # 3. Check Layer B Universal JIT Bytecode Cache
        cached_bc = self.bytecode_cache.lookup(canonical_sig_key) if use_bytecode_cache else None
        if cached_bc is not None:
            gen_raw = cached_bc["canonical_body"]
            bytecode = list(cached_bc["bytecode"])
            arg_names = list(cached_bc.get("arg_names", []))
            disassembly = self.vm.disassemble(bytecode)
            t_gen_s = 0.0
            t_comp_ms = 0.0
            bc_cache_hit = True
            body_code = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
            indented_body = "\n".join("    " + line.strip() for line in body_code.split("\n") if line.strip())
            full_code = prompt_clean.strip() + "\n" + indented_body
        else:
            bc_cache_hit = False
            # Neural Generation
            t0_gen = time.perf_counter()
            gen_raw, _ = self.model.generate(
                prompt=prompt_to_gen,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.85,
                repetition_penalty=1.0,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
            t_gen_s = time.perf_counter() - t0_gen

            # Decanonicalize and construct full code
            body_code = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
            indented_body = "\n".join("    " + line.strip() for line in body_code.split("\n") if line.strip())
            full_code = prompt_clean.strip() + "\n" + indented_body

            # AST -> Bytecode Compilation
            t0_comp = time.perf_counter()
            compilation_error = None
            try:
                bytecode = self.compiler.compile(full_code)
                disassembly = self.vm.disassemble(bytecode)
                arg_names = self.compiler.arg_names
            except Exception as e:
                bytecode = []
                disassembly = f"Compilation Error: {e}"
                arg_names = []
                compilation_error = str(e)
            t_comp_ms = (time.perf_counter() - t0_comp) * 1000.0

            if compilation_error is not None:
                return EngineResult(
                    prompt=prompt_clean.strip(),
                    canonical_prompt=canonical_prompt.strip(),
                    inv_map=inv_map,
                    generated_raw=gen_raw,
                    code=full_code,
                    bytecode=[],
                    disassembly=disassembly,
                    test_outcomes=[
                        TestOutcome(
                            inputs=test_inputs or [],
                            vm_result=None,
                            py_result=None,
                            matched=False,
                            latency_us=0.0,
                            error=compilation_error,
                        )
                    ],
                    latency_gen_s=t_gen_s,
                    latency_comp_ms=t_comp_ms,
                    latency_vm_avg_us=0.0,
                    verified=False,
                    arg_names=[],
                    bytecode_cache_hit=False,
                )

        # 6. Resolve / Auto-generate test inputs
        cases = self._resolve_test_inputs(test_inputs, arg_names)

        # 7. Prepare Python reference execution in restricted sandbox scope
        local_scope: Dict[str, Any] = {}
        safe_globals: Dict[str, Any] = {"__builtins__": _SAFE_VERIFY_BUILTINS}
        py_callable = None
        try:
            exec(full_code, safe_globals, local_scope)
            py_callable = next(v for v in local_scope.values() if callable(v))
        except Exception:
            py_callable = None

        # 8. MiniStackVM execution and verification
        test_outcomes: List[TestOutcome] = []
        vm_latencies: List[float] = []

        for case in cases:
            t0_vm = time.perf_counter()
            err = None
            try:
                vm_res = self.vm.execute(bytecode, case)
                t_vm_us = (time.perf_counter() - t0_vm) * 1_000_000.0
                vm_latencies.append(t_vm_us)
            except Exception as e:
                vm_res = None
                t_vm_us = (time.perf_counter() - t0_vm) * 1_000_000.0
                err = str(e)

            if py_callable is not None:
                try:
                    py_res = py_callable(*case)
                    if isinstance(py_res, bool):
                        py_res = 1 if py_res else 0
                except Exception as e:
                    py_res = f"PyError: {e}"
            else:
                py_res = "ExecError"

            matched = (vm_res == py_res) if err is None else False
            test_outcomes.append(
                TestOutcome(
                    inputs=case,
                    vm_result=vm_res,
                    py_result=py_res,
                    matched=matched,
                    latency_us=t_vm_us,
                    error=err,
                )
            )

        avg_vm_us = sum(vm_latencies) / max(1, len(vm_latencies))
        verified = len(test_outcomes) > 0 and all(t.matched for t in test_outcomes)

        # 9. Persist verified canonical signature -> bytecode into Layer B JIT Cache
        if verified and use_bytecode_cache and not bc_cache_hit:
            self.bytecode_cache.store(
                canonical_signature=canonical_sig_key,
                canonical_body=gen_raw,
                bytecode=bytecode,
                arg_names=arg_names,
                verified=True,
            )

        return EngineResult(
            prompt=prompt_clean.strip(),
            canonical_prompt=canonical_prompt.strip(),
            inv_map=inv_map,
            generated_raw=gen_raw,
            code=full_code,
            bytecode=bytecode,
            disassembly=disassembly,
            test_outcomes=test_outcomes,
            latency_gen_s=t_gen_s,
            latency_comp_ms=t_comp_ms,
            latency_vm_avg_us=avg_vm_us,
            verified=verified,
            arg_names=arg_names,
            bytecode_cache_hit=bc_cache_hit,
        )

    def _resolve_test_inputs(
        self,
        test_inputs: Optional[List[Any]],
        arg_names: List[str],
    ) -> List[List[Any]]:
        """Normalizes user test inputs or synthesizes canonical test suites."""
        num_args = len(arg_names)

        if test_inputs is not None:
            # Handle user supplied JSON / list structure
            if not isinstance(test_inputs, list):
                test_inputs = [test_inputs]

            # If list of lists
            if test_inputs and all(isinstance(x, list) for x in test_inputs):
                return test_inputs

            # If flat list of arguments for 1 test case: [10, 20]
            if len(test_inputs) == num_args and num_args > 1:
                return [test_inputs]

            # If 1-argument function and passed flat list of test values: [4, 7, -2]
            if num_args == 1:
                return [[x] if not isinstance(x, list) else x for x in test_inputs]

            return [test_inputs]

        # Auto-generate representative test cases
        if num_args == 1:
            return [[4], [-5], [0], [7], [12]]
        elif num_args == 2:
            return [[10, 25], [0, 5], [-3, 8], [12, 4], [100, 200]]
        elif num_args == 3:
            return [[2, 3, 5], [1, 10, 2], [4, 4, 4], [-2, 3, 7]]
        else:
            return [
                [i + 1 for i in range(num_args)],
                [-(i + 1) for i in range(num_args)],
            ]
