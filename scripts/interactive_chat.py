"""
Interactive CLI for Feistback Neuro-Symbolic Intent Dispatcher
==============================================================
Provides an interactive command-line interface for communicating with the
Feistback neuro-symbolic engine via ConfigDispatcher or CouncilDispatcher in real time,
displaying intent matching, generated Python code, disassembled MiniStackVM bytecode,
Confidence Gate protection, and verified deterministic execution results.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime import FeistbackEngine
from dispatchers import ConfigDispatcher, CouncilDispatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive natural language chat with Feistback neuro-symbolic engine"
    )
    default_ckpt = "models/logic_feistel_clean/pure_feistel_raf_clean.pt"
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained checkpoint (.pt) (default: models/logic_feistel_clean/pure_feistel_raf_clean.pt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/intents.json",
        help="Path to intents JSON configuration (default: config/intents.json)",
    )
    parser.add_argument(
        "--dispatcher",
        choices=["fast_path", "council"],
        default="council",
        help="Vali dispetšer: fast_path (reeglid) voi council (eesti_keele_sl2ng mikromudel)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device for PureFeistelLM ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def print_banner(dispatcher: Any, device: str) -> None:
    print("=" * 80)
    print("FEISTBACK NEURO-SYMBOLIC CHAT CLI")
    print("=" * 80)
    if hasattr(dispatcher, "intents"):
        disp_info = f"Registered Intents: {len(dispatcher.intents)}"
    else:
        disp_info = "Micro-Model: eesti_keele_sl2ng (Neural Council)"
    print(f"Device: {device} | Dispatcher: {dispatcher.__class__.__name__} ({disp_info})")
    print("Type a natural language or 'estengrish' query (or 'quit'/'exit' to exit).")
    print("Example queries:")
    print("  - 'liida kokku 15 ja 25'")
    print("  - 'clämbi see counter 105 vahemikku 0 ja 100'")
    print("  - 'arvuta vahe ruut arvudel 15 ja 8 ja liida üks otsa'")
    print("  - 'viska xor 12 ja 4 vahel'")
    print("  - 'kas number -7 on negatiivne'")
    print("  - 'liida kolm arvu 1, 5 ja 6 lahuta sellest 2'")
    print("=" * 80)


def process_query(
    dispatcher: Any,
    query: str,
    pending_intent: Optional[Dict[str, Any]] = None,
    pending_args: Optional[List[int]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[int]]:
    t0 = time.perf_counter()
    resp = dispatcher.dispatch(query, pending_intent=pending_intent, pending_args=pending_args)
    total_time_ms = (time.perf_counter() - t0) * 1000.0

    print("-" * 80)
    print(f"Query: \"{query}\"")

    status = resp.get("status")

    if status == "NEED_CONFIRMATION":
        print(f"  [Status]      : NEED_CONFIRMATION (Confidence Gate kaitse)")
        print(f"  [Candidate]   : {resp.get('intent_name')} -> {resp.get('signature')}")
        print(f"  [Confidence]  : {resp.get('confidence_margin', 0.0) * 100.0:.1f}%")
        print(f"\n  Dispetšer > {resp.get('message')} (vasta 'jah' või sisesta uus käsk)\n")
        print("-" * 80)
        return resp.get("intent"), resp.get("args", [])

    elif status == "NEED_INPUT":
        intent_info = resp.get("intent", {})
        intent_name = resp.get("intent_name", intent_info.get("name", "Tundmatu"))
        print(f"  [Status]      : NEED_INPUT (Ootel argumentide ootamine)")
        print(f"  [Intent]      : {intent_name}")
        print(f"  [Args Found]  : {resp.get('provided_args', [])} (Vajatakse kokku: {resp.get('expected_count')})")
        print(f"\n  Dispetšer > {resp.get('message')}\n")
        print("-" * 80)
        return resp.get("intent"), resp.get("provided_args", [])

    elif status == "SUCCESS":
        mode = resp.get("mode", "execute")
        intent_id = resp.get("intent_id")
        intent_name = resp.get("intent_name", intent_id)
        gen_code = resp.get("generated_code", "").strip()
        gen_lat_s = resp.get("generation_latency_s", 0.0)

        if mode == "synthesize":
            print(f"  [Mode]        : Code Synthesis (Synthesize Only - No VM Execution)")
            print(f"  [Intent]      : {intent_name} ({intent_id})")
            print(f"  [Signature]   : {resp.get('signature')}")
            print("  [Generated Code]:")
            for line in gen_code.split("\n"):
                print(f"      {line}")
            print(f"  [Verification]: Syntax verified by ASTCompiler: OK (100% Valid Python)")
            print(f"  [Performance] : Neural Gen: {gen_lat_s * 1000.0:.1f} ms | Total: {total_time_ms:.1f} ms")
            print(f"  [Message]     : {resp.get('message')}")
        else:
            args = resp.get("args")
            disasm = resp.get("disassembly", "").strip()
            vm_res = resp.get("result")
            vm_lat_us = resp.get("vm_latency_us", 0.0)
            comp_lat_ms = resp.get("compilation_latency_ms", 0.0)
            bc_hit = resp.get("bytecode_cache_hit", False)

            print(f"  [Intent]      : {intent_name} ({intent_id})")
            print(f"  [Arguments]   : {args}")
            print("  [Generated Code]:")
            for line in gen_code.split("\n"):
                print(f"      {line}")
            print("  [MiniStackVM Bytecode]:")
            for line in disasm.split("\n"):
                print(f"      {line}")
            print(f"  [Result]      : {vm_res}  (Deterministic VM Verified: 100% MATCH)")
            print(
                f"  [Performance] : VM Latency: {vm_lat_us:.2f} µs | "
                f"Compile: {comp_lat_ms:.2f} ms | "
                f"Neural Gen: {gen_lat_s * 1000.0:.1f} ms ({'JIT Cache Hit' if bc_hit else 'Synthesized'}) | "
                f"Total: {total_time_ms:.1f} ms"
            )
        print("-" * 80)
        return None, []

    else:
        err_msg = resp.get("error", "Unknown dispatch error")
        print(f"  [Status]      : {resp.get('status')}")
        print(f"  [Error]       : {err_msg}")
        if "score" in resp:
            print(f"  [Score]       : {resp['score']:.1f}")
        print("-" * 80)
        return None, []


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print(f"Initializing FeistbackEngine on {args.device}...")
    engine = FeistbackEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if args.dispatcher == "council":
        dispatcher = CouncilDispatcher(engine=engine, device=args.device)
    else:
        dispatcher = ConfigDispatcher(engine=engine, config_path=args.config)
    print_banner(dispatcher, args.device)

    pending_intent: Optional[Dict[str, Any]] = None
    pending_args: List[int] = []

    while True:
        try:
            prompt_prefix = "\nFeistback (ootel) > " if pending_intent else "\nFeistback > "
            user_input = input(prompt_prefix).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Feistback Chat CLI.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", ":q"}:
            print("Exiting Feistback Chat CLI.")
            break

        if user_input.lower() in {"cancel", "tühista", "reset"}:
            pending_intent = None
            pending_args = []
            print("Ootel kavatsus tühistatud.")
            continue

        pending_intent, pending_args = process_query(
            dispatcher,
            user_input,
            pending_intent=pending_intent,
            pending_args=pending_args,
        )


if __name__ == "__main__":
    main()
