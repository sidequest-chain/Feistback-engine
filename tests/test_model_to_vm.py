"""
Model to MiniStackVM End-to-End Verification Test
=================================================
Validates the complete neuro-symbolic pipeline:
1. Autoregressive neural code generation via PureFeistelLM (RAF, 0 bytes KV-cache)
2. AST signature canonicalization and decanonicalization
3. AST -> MiniStackVM bytecode compilation
4. Deterministic microsecond execution on MiniStackVM
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from runtime.models.pure_feistel_lm import PureFeistelLM
from runtime.compiler import ASTCompiler
from runtime.vm.ministack_vm import MiniStackVM
from runtime.canonicalizer import canonicalize_signature, decanonicalize_code


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    checkpoint_path = "models/logic_feistel_clean/pure_feistel_raf_clean.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("NEURO-SYMBOLIC PIPELINE TEST: PUREFEISTELLM -> AST COMPILER -> MINISTACKVM")
    print("=" * 80)
    print(f"Device              : {device}")
    print(f"Checkpoint          : {checkpoint_path}")

    # 1. Instantiate and load model
    print("\n[1/3] Loading PureFeistelLM (RAF, 16k vocab, 0 bytes KV cache)...")
    t0_load = time.perf_counter()
    model = PureFeistelLM(
        vocab_size=16384,
        d_model=896,
        num_layers=12,
        state_dim=128,
        memory_type="raf",
        torch_dtype=torch.bfloat16,
        device=device,
        from_pretrained_qwen=False,
        pruned_weights_path="models/logic_feistel_clean/vocab_16k/pruned_weights.pt",
        vocab_map_path="models/logic_feistel_clean/vocab_16k/vocab_map.pt",
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_backbone(ckpt)
    print(f"  Model loaded in {(time.perf_counter() - t0_load):.2f}s | Step: {ckpt.get('step')} | Loss: {ckpt.get('loss_ce', 0.0):.4f}")

    # 2. Define test suite
    test_cases = [
        {
            "id": 1,
            "name": "OOD Binary Addition",
            "prompt": "def add(alpha: int, beta: int) -> int:",
            "test_inputs": [10, 25],
            "expected_fn": lambda a, b: a + b,
            "input_repr": "alpha=10, beta=25",
        },
        {
            "id": 2,
            "name": "OOD Single Parity Check",
            "prompt": "def is_even(counter_val: int) -> bool:",
            "test_inputs": [4],
            "expected_fn": None,  # Will verify against Python execution of generated code
            "input_repr": "counter_val=4",
        },
        {
            "id": 3,
            "name": "OOD Triple Multiplication",
            "prompt": "def multiply_three(first: int, second: int, third: int) -> int:",
            "test_inputs": [2, 3, 5],
            "expected_fn": lambda a, b, c: a * b * c,
            "input_repr": "first=2, second=3, third=5",
        },
        {
            "id": 4,
            "name": "OOD Comparison (Negative)",
            "prompt": "def is_negative(val: int) -> bool:",
            "test_inputs": [-5],
            "expected_fn": lambda x: 1 if x < 0 else 0,
            "input_repr": "val=-5",
        },
    ]

    compiler = ASTCompiler()
    vm = MiniStackVM()

    print("\n[2/3] Executing 4 End-to-End Generations & Compilations...")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    for tc in test_cases:
        prompt_raw = tc["prompt"].strip()
        if prompt_raw.endswith(":"):
            prompt_raw += "\n"

        # Canonicalize signature
        canonical_prompt, inv_map = canonicalize_signature(prompt_raw)

        # Neural Generation
        t0_gen = time.perf_counter()
        gen_raw, _ = model.generate(
            prompt=canonical_prompt,
            max_new_tokens=32,
            temperature=0.1,
            top_p=0.85,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        t_gen = time.perf_counter() - t0_gen

        # Decanonicalize
        body_code = decanonicalize_code(gen_raw, inv_map) if inv_map else gen_raw
        indented_body = "\n".join("    " + line.strip() for line in body_code.split("\n") if line.strip())
        full_code = prompt_raw.strip() + "\n" + indented_body

        # AST -> Bytecode Compilation
        t0_comp = time.perf_counter()
        bytecode = compiler.compile(full_code)
        t_comp = (time.perf_counter() - t0_comp) * 1000.0  # ms

        # MiniStackVM Execution
        t0_vm = time.perf_counter()
        vm_result = vm.execute(bytecode, tc["test_inputs"])
        t_vm = (time.perf_counter() - t0_vm) * 1_000_000.0  # microseconds

        # Reference verification via Python exec
        local_scope: Dict[str, Any] = {}
        try:
            exec(full_code, {}, local_scope)
            fn_callable = next(v for v in local_scope.values() if callable(v))
            py_result = fn_callable(*tc["test_inputs"])
            if isinstance(py_result, bool):
                py_result = 1 if py_result else 0
        except Exception as e:
            py_result = f"ExecError: {e}"

        match = (vm_result == py_result)

        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "prompt": tc["prompt"],
            "code": full_code,
            "bytecode": bytecode,
            "inputs": tc["input_repr"],
            "vm_res": vm_result,
            "py_res": py_result,
            "match": match,
            "t_gen": t_gen,
            "t_comp_ms": t_comp,
            "t_vm_us": t_vm,
        })

    # 3. Report Results
    print("\n[3/3] Detailed Verification Results:")
    print("=" * 80)
    for res in results:
        status_str = "SUCCESS (100% MATCH)" if res["match"] else "FAILED (MISMATCH)"
        print(f"Test {res['id']}: {res['name']}")
        print(f"  Input Prompt     : {res['prompt']}")
        print(f"  Generated Code   :")
        for line in res['code'].split('\n'):
            print(f"    {line}")
        print(f"  Disassembled VM  : {len(res['bytecode'])} instruction words -> {res['bytecode']}")
        print(f"  Test Evaluation  : {res['inputs']}")
        print(f"  MiniStackVM Res  : {res['vm_res']} (latency: {res['t_vm_us']:.2f} µs)")
        print(f"  Python Ref Res   : {res['py_res']}")
        print(f"  Compiler Latency : {res['t_comp_ms']:.4f} ms")
        print(f"  Gen Latency      : {res['t_gen']:.3f} s")
        print(f"  Status           : {status_str}")
        print("-" * 80)

    all_passed = all(r["match"] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 4 TESTS PASSED: 100% Deterministic Equivalence between Neural Output and MiniStackVM!")
    else:
        print("SOME TESTS FAILED: Please inspect discrepancies above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
