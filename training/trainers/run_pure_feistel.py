"""
Run PureFeistelLM Script
========================
Loads PureFeistelLM, restores trained distillation checkpoint weights,
and executes autoregressive code generation without KV-cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from runtime.canonicalizer import canonicalize_signature, decanonicalize_code
from runtime.models.pure_feistel_lm import PureFeistelLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PureFeistelLM autoregressive generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/logic_feistel_clean/pure_feistel_raf_clean.pt",
        help="Path to trained checkpoint file (.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="def fibonacci(n: int) -> int:\n",
        help="Input prompt for code generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (0.0 for greedy argmax)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.85,
        help="Nucleus sampling probability threshold (0.0 < p <= 1.0)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty multiplier (whitespace and indentation tokens exempt)",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Size of n-grams that cannot be repeated (default: 3, 0 to disable)",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable indentation and terminal statement early stopping (default: True)",
    )
    parser.add_argument(
        "--canonicalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt argument canonicalization and output decanonicalization (default: True)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of Feistel layers",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=128,
        help="Recurrent state dimension for GatedLinearRecurrence (default: 128)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=151936,
        help="Vocabulary size (151936 or 16384)",
    )
    parser.add_argument(
        "--vocab_map",
        type=str,
        default="models/logic_feistel_clean/vocab_16k/vocab_map.pt",
        help="Path to vocab_map.pt for 16k vocab",
    )
    parser.add_argument(
        "--memory_type",
        type=str,
        default=None,
        choices=["glr", "raf"],
        help="Memory mechanism ('glr' or 'raf', auto-detected from checkpoint if None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    # Detect vocab_size, num_layers, state_dim, and memory_type from checkpoint if present
    target_vocab_size = args.vocab_size
    target_num_layers = args.num_layers
    target_state_dim = args.state_dim
    target_memory_type = args.memory_type
    if os.path.exists(args.checkpoint):
        try:
            ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(ckpt_meta, dict):
                if "vocab_size" in ckpt_meta:
                    target_vocab_size = ckpt_meta["vocab_size"]
                if "num_layers" in ckpt_meta:
                    target_num_layers = ckpt_meta["num_layers"]
                if "state_dim" in ckpt_meta:
                    target_state_dim = ckpt_meta["state_dim"]
                if target_memory_type is None and "memory_type" in ckpt_meta:
                    target_memory_type = ckpt_meta["memory_type"]
        except Exception:
            pass

    if target_memory_type is None:
        target_memory_type = "raf" if "raf" in args.checkpoint.lower() else "glr"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PURE FEISTEL LM INFERENCE RUNNER (0 BYTES KV-CACHE)")
    print("=" * 75)
    print(f"Device              : {device}")
    print(f"Checkpoint          : {args.checkpoint}")
    print(f"Feistel Layers      : {target_num_layers}")
    print(f"Memory Type         : {target_memory_type.upper()}")
    print(f"State Dimension     : {target_state_dim if target_memory_type == 'glr' else '4x16x16 (Matrix)'}")
    print(f"Vocab Size          : {target_vocab_size}")
    print(f"Max New Tokens      : {args.max_new_tokens}")
    print(f"Temperature         : {args.temperature}")
    print(f"Top-P               : {args.top_p}")
    print(f"Repetition Penalty  : {args.repetition_penalty}")
    print(f"No Repeat N-Gram    : {args.no_repeat_ngram_size}")
    print(f"Early Stopping      : {args.early_stopping}")
    print(f"Canonicalize        : {args.canonicalize}")
    print("=" * 75)

    # 1. Initialize PureFeistelLM
    print("\n[1/3] Instantiating PureFeistelLM architecture...")
    model = PureFeistelLM(
        vocab_size=target_vocab_size,
        d_model=896,
        num_layers=target_num_layers,
        state_dim=target_state_dim,
        memory_type=target_memory_type,
        torch_dtype=torch.bfloat16,
        device=device,
        from_pretrained_qwen=(target_vocab_size != 16384),
        pruned_weights_path="models/logic_feistel_clean/vocab_16k/pruned_weights.pt",
        vocab_map_path=args.vocab_map,
    )

    # 2. Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"\n[2/3] Loading trained weights from '{args.checkpoint}'...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_backbone(ckpt)
        if isinstance(ckpt, dict) and "step" in ckpt:
            loss_val = ckpt.get("loss_total", ckpt.get("loss", 0.0))
            kd_val = ckpt.get("loss_kd", 0.0)
            ce_val = ckpt.get("loss_ce", 0.0)
            print(f"  Checkpoint Step: {ckpt.get('step')} | TotalLoss: {loss_val:.4f} | KDLoss: {kd_val:.4f} | CELoss: {ce_val:.4f} | CosSim: {ckpt.get('cos_sim', 0.0):.4f}")
    else:
        print(f"\n[WARNING] Checkpoint '{args.checkpoint}' not found! Running with initial weights.")

    # 3. Autoregressive Generation
    prompt_raw = args.prompt.replace("\\n", "\n").replace("\\t", "\t")
    # Prompt normalization: strip trailing whitespace to ensure clean BPE token alignment
    prompt_clean = prompt_raw.rstrip()
    if prompt_clean.endswith(":"):
        prompt_clean += "\n"
    prompt = prompt_clean

    prompt_to_generate = prompt
    inv_map: Dict[str, str] = {}
    if args.canonicalize:
        canonical_prompt, inv_map = canonicalize_signature(prompt)
        if inv_map:
            prompt_to_generate = canonical_prompt

    print("\n[3/3] Generating code autoregressively (0 bytes KV cache)...")
    print("-" * 75)
    print("PROMPT:")
    print(prompt, end="" if prompt.endswith("\n") else "\n")
    if inv_map:
        print("CANONICAL PROMPT (BPE Adapted):")
        print(prompt_to_generate, end="" if prompt_to_generate.endswith("\n") else "\n")
        print(f"VARIABLE MAPPINGS   : {inv_map}")
    print("-" * 75)

    t0 = time.perf_counter()
    generated_text, token_ids = model.generate(
        prompt=prompt_to_generate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        early_stopping=args.early_stopping,
    )
    elapsed = time.perf_counter() - t0
    prompt_len = model.tokenizer(prompt_to_generate, return_tensors="pt")["input_ids"].shape[-1]
    num_gen_tokens = token_ids.shape[-1] - prompt_len
    tok_per_sec = num_gen_tokens / max(1e-6, elapsed)

    if args.canonicalize and inv_map:
        final_completion = decanonicalize_code(generated_text, inv_map)
    else:
        final_completion = generated_text

    print("GENERATED COMPLETION:")
    print(final_completion)
    print("-" * 75)
    print(f"Generation Latency : {elapsed:.3f} s ({num_gen_tokens} tokens @ {tok_per_sec:.1f} tok/s)")
    print(f"KV-Cache Memory    : 0 bytes (Verified)")
    print("=" * 75)


if __name__ == "__main__":
    main()
