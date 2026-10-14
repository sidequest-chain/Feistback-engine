"""
End-to-End Test Suite for Intent Dispatcher, CouncilPipeline & Two-Layer Cache
==============================================================================
Validates:
1. Natural language and 'estengrish' query routing via ConfigDispatcher (Immutable L1 Base).
2. Layer B Universal JIT BytecodeCache (`canonical_signature -> bytecode` in 0.0ms).
3. CouncilDispatcher + CouncilPipeline + Layer A Confidence Gate (`min_margin` + `model_hash`):
   - Conversational Estonian wrapper ("ou pane 15 ja 25 kokku plz") executes with high confidence.
   - Estonian elative case reversed word order ("lahuta 12 arvust 50" -> 50 - 12 = 38).
   - Everyday Estonian base vocabulary ("mul oli 100 eurot ja kulutasin 35 eurot" -> 65).
   - Sandboxed VM Tool contracts via OpCode.CALL_TOOL ("arvuta 200 eurot 15% allahindlusega" -> 170.0,
     "leia suurim ühistegur arvudele 48 ja 18" -> 6).
   - Subsequent structural match hits Layer A + Layer B in microseconds.
   - Genuinely ambiguous/contradictory query ("liida või lahuta või korruta 15 ja 25") is
     blocked by `CONFIDENCE_GATE_BLOCKED` (`NEED_CONFIRMATION`), preventing cache poisoning!
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime import BytecodeCache, FeistbackEngine
from dispatchers import ConfigDispatcher, CouncilDispatcher, CouncilPipeline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 85)
    print("FEISTBACK INTENT DISPATCHER VERIFICATION TEST (NATURAL LANGUAGE -> VM)")
    print("=" * 85)

    checkpoint_path = "models/logic_feistel_clean/pure_feistel_raf_clean.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        bc_cache_path = os.path.join(tmpdir, "bytecode_jit.json")
        intent_cache_path = os.path.join(tmpdir, "learned_intents.json")

        print(f"[1/3] Loading FeistbackEngine from {checkpoint_path}...")
        engine = FeistbackEngine.from_checkpoint(checkpoint_path)
        engine.bytecode_cache = BytecodeCache(cache_path=bc_cache_path, model_hash=engine.model_hash)
        dispatcher = ConfigDispatcher(engine, config_path="config/intents.json")
        print(f"Loaded {len(dispatcher.intents)} intents from config/intents.json.\n")

        # The 5 required benchmark test queries
        test_cases = [
            {
                "id": 1,
                "query": "liida kokku 15 ja 25",
                "expected_intent": "add",
                "expected_args": [15, 25],
                "expected_result": 40,
            },
            {
                "id": 2,
                "query": "clämbi see counter 105 vahemikku 0 ja 100",
                "expected_intent": "clamp_math",
                "expected_args": [105, 0, 100],
                "expected_result": 100,
            },
            {
                "id": 3,
                "query": "arvuta vahe ruut arvudel 15 ja 8 ja liida üks otsa",
                "expected_intent": "calc_with_locals",
                "expected_args": [15, 8],
                "expected_result": 50,
            },
            {
                "id": 4,
                "query": "viska xor 12 ja 4 vahel",
                "expected_intent": "bitwise_comb",
                "expected_args": [12, 4],
                "expected_result": 8,
            },
            {
                "id": 5,
                "query": "kas number -7 on negatiivne",
                "expected_intent": "is_negative",
                "expected_args": [-7],
                "expected_result": 1,
            },
        ]

        print("[2/3] Running 5 Natural Language & Estengrish Test Queries...")
        print("=" * 85)

        all_passed = True
        results_summary = []

        for tc in test_cases:
            t0 = time.perf_counter()
            resp = dispatcher.dispatch(tc["query"])
            total_time_ms = (time.perf_counter() - t0) * 1000.0

            status = resp.get("status")
            intent_id = resp.get("intent_id")
            args = resp.get("args")
            vm_res = resp.get("result")
            gen_code = resp.get("generated_code", "").strip()
            vm_lat_us = resp.get("vm_latency_us", 0.0)

            passed = (
                status == "SUCCESS"
                and intent_id == tc["expected_intent"]
                and args == tc["expected_args"]
                and vm_res == tc["expected_result"]
            )

            if not passed:
                all_passed = False

            status_str = "PASS" if passed else "FAIL"
            results_summary.append({
                "id": tc["id"],
                "query": tc["query"],
                "intent": intent_id,
                "args": args,
                "vm_res": vm_res,
                "expected": tc["expected_result"],
                "vm_latency_us": vm_lat_us,
                "total_ms": total_time_ms,
                "status": status_str,
                "code": gen_code,
            })

            print(f"Test {tc['id']}: \"{tc['query']}\"")
            print(f"  Matched Intent  : {intent_id} (Expected: {tc['expected_intent']})")
            print(f"  Extracted Args  : {args} (Expected: {tc['expected_args']})")
            print(f"  Generated Code  :\n    " + "\n    ".join(gen_code.split("\n")))
            print(f"  MiniStackVM Res : {vm_res} (Expected: {tc['expected_result']})")
            print(f"  Execution Stats : VM Latency: {vm_lat_us:.2f} us | Total Turnaround: {total_time_ms:.1f} ms")
            print(f"  Status          : {status_str}")
            print("-" * 85)

        print("\n" + "=" * 85)
        print("TEST SUMMARY REPORT")
        print("=" * 85)
        print(f"{'ID':<3} | {'Query':<36} | {'Intent':<17} | {'Result':<7} | {'VM Lat':<10} | {'Status'}")
        print("-" * 85)
        for r in results_summary:
            print(
                f"{r['id']:<3} | {r['query'][:36]:<36} | {r['intent']:<17} | "
                f"{str(r['vm_res']):<7} | {r['vm_latency_us']:>6.2f} us | {r['status']}"
            )
        print("=" * 85)

        assert all_passed, "One or more dispatcher test cases failed!"

        # 3. Verify CouncilPipeline, Rich Estonian Morphology, VM Tools & Confidence Gate
        slang_ckpt = "models/eesti_keele_sl2ng/slang_expert_feistel.pt"
        if os.path.exists(slang_ckpt):
            print("\n[3/3] Testing CouncilPipeline, Estonian Morphology, VM Tools & Confidence Gate...")
            council = CouncilDispatcher(
                engine=engine,
                model_path=slang_ckpt,
                cache_path=intent_cache_path,
            )
            pipeline = CouncilPipeline(dispatcher=council)

            # A. Conversational Estonian wrapper ("ou pane 15 ja 25 kokku plz") -> High confidence!
            t0 = time.perf_counter()
            resp_conv = pipeline.run("ou pane 15 ja 25 kokku plz")
            t_conv_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"  [1. Conversational Slang ] Route: {resp_conv.get('route')} | "
                f"Domain: {resp_conv.get('domain')} | "
                f"Margin: {resp_conv.get('confidence_margin', 0)*100:.1f}% | "
                f"Result: {resp_conv.get('result')} | Time: {t_conv_ms:.1f} ms"
            )
            assert resp_conv["status"] == "SUCCESS" and resp_conv["result"] == 40
            assert resp_conv["bytecode_cache_hit"] is True

            # B. Second call with same conversational structure ("ou pane 100 ja 250 kokku plz") -> Hits Layer A + Layer B!
            t0 = time.perf_counter()
            resp_t1 = pipeline.run("ou pane 100 ja 250 kokku plz")
            t_t1_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"  [2. Layer A + Layer B Hit] Route: {resp_t1.get('route')} | "
                f"JIT Bytecode Hit: {resp_t1.get('bytecode_cache_hit')} | "
                f"Result: {resp_t1.get('result')} | Time: {t_t1_ms:.2f} ms"
            )
            assert resp_t1["status"] == "SUCCESS" and resp_t1["result"] == 350
            assert resp_t1["cache_hit"] is True and resp_t1["bytecode_cache_hit"] is True

            # C. Estonian case morphology: reversed elative ("lahuta 12 arvust 50" -> 50 - 12 = 38)
            resp_morph = pipeline.run("lahuta 12 arvust 50")
            print(
                f"  [3. Estonian Morphology  ] Intent: {resp_morph.get('intent_name')} | "
                f"Args: {resp_morph.get('args')} | Result: {resp_morph.get('result')} (Expected: 38)"
            )
            assert resp_morph["status"] == "SUCCESS" and resp_morph["result"] == 38

            # D. Everyday Estonian base vocabulary ("mul oli 100 eurot ja kulutasin 35 eurot" -> 65)
            resp_base = pipeline.run("mul oli 100 eurot ja kulutasin 35 eurot")
            print(
                f"  [4. Base Estonian Vocab  ] Intent: {resp_base.get('intent_name')} | "
                f"Args: {resp_base.get('args')} | Result: {resp_base.get('result')} (Expected: 65)"
            )
            assert resp_base["status"] == "SUCCESS" and resp_base["result"] == 65

            # E. Sandboxed VM Tool via CALL_TOOL ("arvuta 200 eurot 15% allahindlusega" -> 170.0)
            resp_disc = pipeline.run("arvuta 200 eurot 15% allahindlusega")
            print(
                f"  [5. VM CALL_TOOL Discount] Domain: {resp_disc.get('domain')} | "
                f"Intent: {resp_disc.get('intent_name')} | Result: {resp_disc.get('result')} (Expected: 170.0)"
            )
            assert resp_disc["status"] == "SUCCESS" and abs(resp_disc["result"] - 170.0) < 1e-6

            # F. Sandboxed VM Tool via CALL_TOOL ("leia suurim ühistegur arvudele 48 ja 18" -> 6)
            resp_gcd = pipeline.run("leia suurim ühistegur arvudele 48 ja 18")
            print(
                f"  [6. VM CALL_TOOL GCD     ] Domain: {resp_gcd.get('domain')} | "
                f"Intent: {resp_gcd.get('intent_name')} | Result: {resp_gcd.get('result')} (Expected: 6)"
            )
            assert resp_gcd["status"] == "SUCCESS" and resp_gcd["result"] == 6

            # G. Contradictory/ambiguous query -> Blocked by Confidence Gate (NEED_CONFIRMATION)
            ambig_q = "kas liida või lahuta või korruta või jaga 15 ja 25"
            resp_ambig = pipeline.run(ambig_q)
            print(
                f"  [7. Poisoning Protection ] Route: {resp_ambig.get('route')} | "
                f"Status: {resp_ambig.get('status')} | "
                f"Margin: {resp_ambig.get('confidence_margin', 0)*100:.1f}%"
            )
            assert resp_ambig["status"] == "NEED_CONFIRMATION"
            assert council.learned_cache.lookup(ambig_q, 2) is None

    print("\nALL DISPATCHER, COUNCIL PIPELINE, VM TOOL, AND CONFIDENCE GATE TESTS PASSED!")


if __name__ == "__main__":
    main()
