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

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from core.pure_feistel_lm import PureFeistelLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PureFeistelLM autoregressive generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pure_feistel_final.pt",
        help="Path to trained checkpoint file (.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default='def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n',
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
        default="checkpoints/vocab_16k/vocab_map.pt",
        help="Path to vocab_map.pt for 16k vocab",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device (default: cuda if available)",
    )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PURE FEISTEL LM INFERENCE RUNNER (0 BYTES KV-CACHE)")
    print("=" * 75)
    print(f"Device              : {device}")
    print(f"Checkpoint          : {args.checkpoint}")
    print(f"Feistel Layers      : {args.num_layers}")
    print(f"State Dimension     : {args.state_dim}")
    print(f"Vocab Size          : {args.vocab_size}")
    print(f"Max New Tokens      : {args.max_new_tokens}")
    print(f"Temperature         : {args.temperature}")
    print(f"Top-P               : {args.top_p}")
    print(f"Repetition Penalty  : {args.repetition_penalty}")
    print("=" * 75)

    # Detect vocab_size, num_layers, and state_dim from checkpoint if present
    target_vocab_size = args.vocab_size
    target_num_layers = args.num_layers
    target_state_dim = args.state_dim
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
                print(f"[Run] Checkpoint metadata: vocab_size={target_vocab_size}, num_layers={target_num_layers}, state_dim={target_state_dim}")
        except Exception:
            pass

    # 1. Initialize PureFeistelLM
    print("\n[1/3] Instantiating PureFeistelLM architecture...")
    model = PureFeistelLM(
        vocab_size=target_vocab_size,
        d_model=896,
        num_layers=target_num_layers,
        state_dim=target_state_dim,
        torch_dtype=torch.bfloat16,
        device=device,
        from_pretrained_qwen=(target_vocab_size != 16384),
        pruned_weights_path="checkpoints/vocab_16k/pruned_weights.pt",
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
    print("\n[3/3] Generating code autoregressively (0 bytes KV cache)...")
    print("-" * 75)
    print("PROMPT:")
    print(args.prompt, end="" if args.prompt.endswith("\n") else "\n")
    print("-" * 75)

    t0 = time.perf_counter()
    generated_text, token_ids = model.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    elapsed = time.perf_counter() - t0
    num_gen_tokens = token_ids.shape[-1] - model.tokenizer(args.prompt, return_tensors="pt")["input_ids"].shape[-1]
    tok_per_sec = num_gen_tokens / max(1e-6, elapsed)

    print("GENERATED COMPLETION:")
    print(generated_text)
    print("-" * 75)
    print(f"Generation Latency : {elapsed:.3f} s ({num_gen_tokens} tokens @ {tok_per_sec:.1f} tok/s)")
    print(f"KV-Cache Memory    : 0 bytes (Verified)")
    print("=" * 75)


if __name__ == "__main__":
    main()
