"""
Vocabulary Analysis & Pruning Feasibility Tool for PureFeistelLM
================================================================
Analyzes token frequencies, coverage thresholds, base character representations,
and parameter/VRAM savings for pruning Qwen2.5-Coder vocabulary (151k -> 16k / 32k)
on the train_code.jsonl dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoTokenizer


def format_token_display(tok_str: str) -> str:
    """Formats whitespace and control characters for human-readable display."""
    return repr(tok_str)


def analyze_vocabulary(
    data_path: str = "data/train_code.jsonl",
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    d_model: int = 896,
    batch_size: int = 8,
    seq_len: int = 256,
) -> None:
    print("=" * 85)
    print("QWEN2.5-CODER VOCABULARY FREQUENCY & PRUNING ANALYSIS")
    print("=" * 85)
    print(f"Dataset Path         : {data_path}")
    print(f"Base Tokenizer       : {model_name}")
    print(f"Model Hidden Dim     : {d_model}")
    print(f"Training Batch/SeqLen: {batch_size} / {seq_len} tokens")
    print("=" * 85)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found!")

    # 1. Load Tokenizer
    print("\n[1/5] Loading Qwen2.5-Coder tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw_vocab_size = tokenizer.vocab_size
    len_vocab = len(tokenizer)
    model_vocab_size = 151936  # Qwen2.5-Coder model configuration vocab size
    print(f"  tokenizer.vocab_size : {raw_vocab_size:,}")
    print(f"  len(tokenizer)       : {len_vocab:,}")
    print(f"  Model LM Head vocab  : {model_vocab_size:,}")

    # 2. Read and Tokenize Dataset
    print(f"\n[2/5] Tokenizing dataset from '{data_path}'...")
    token_counter: Counter[int] = Counter()
    sample_count = 0
    total_tokens_no_eos = 0

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151643

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            code = entry.get("code") or entry.get("text", "")
            if not code:
                continue

            sample_count += 1
            tokens = tokenizer.encode(code, add_special_tokens=False)
            total_tokens_no_eos += len(tokens)
            token_counter.update(tokens)
            # Include EOS token appended to each packed sample
            token_counter[eos_id] += 1

    total_tokens = sum(token_counter.values())
    unique_tokens = len(token_counter)

    print(f"  Total code samples   : {sample_count:,}")
    print(f"  Total tokens counted : {total_tokens:,} (incl. EOS; {total_tokens_no_eos:,} pure code tokens)")
    print(f"  Unique tokens present: {unique_tokens:,} / {model_vocab_size:,} ({unique_tokens / model_vocab_size * 100:.2f}% of full vocab)")
    print(f"  Unused tokens in data: {model_vocab_size - unique_tokens:,} ({ (model_vocab_size - unique_tokens) / model_vocab_size * 100:.2f}%)")

    # 3. Frequency Distribution & Cumulative Coverage
    print("\n[3/5] Cumulative Coverage & Distribution Thresholds:")
    print("-" * 85)
    print(f"{'Coverage Target':<18} | {'Tokens Needed':<15} | {'% of Full Vocab':<18} | {'Least Freq in Top':<18}")
    print("-" * 85)

    sorted_tokens = token_counter.most_common()
    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999, 1.00]
    thresh_idx = 0
    running_sum = 0
    coverage_results: Dict[float, int] = {}

    for rank, (tid, count) in enumerate(sorted_tokens, start=1):
        running_sum += count
        ratio = running_sum / total_tokens

        while thresh_idx < len(thresholds) and ratio >= thresholds[thresh_idx]:
            target = thresholds[thresh_idx]
            coverage_results[target] = rank
            pct_vocab = (rank / model_vocab_size) * 100
            print(f"{target * 100:>5.1f}% coverage     | {rank:<15,d} | {pct_vocab:>6.2f}%            | {count:<10,d} occurrences")
            thresh_idx += 1

    # In case floating point precision slightly misses 1.00
    if 1.00 not in coverage_results:
        coverage_results[1.00] = unique_tokens
        pct_vocab = (unique_tokens / model_vocab_size) * 100
        print(f"100.0% coverage     | {unique_tokens:<15,d} | {pct_vocab:>6.2f}%            | {sorted_tokens[-1][1]:<10,d} occurrences")

    print("-" * 85)

    # Top 20 Most Frequent Tokens
    print("\nTop 20 Most Frequent Tokens in Dataset:")
    print(f"{'Rank':<6} | {'Token ID':<10} | {'Count':<10} | {'Freq %':<8} | {'Decoded Representation':<30}")
    print("-" * 75)
    for rank, (tid, count) in enumerate(sorted_tokens[:20], start=1):
        freq_pct = (count / total_tokens) * 100
        dec = tokenizer.decode([tid])
        print(f"{rank:<6d} | {tid:<10d} | {count:<10,d} | {freq_pct:>5.2f}%  | {format_token_display(dec):<30}")

    # 4. Base Token Representation Check (Special Tokens & ASCII 0-127)
    print("\n[4/5] Base Token Integrity Check (ASCII 0-127 & Special Tokens):")
    print("-" * 85)

    special_token_map = {
        "EOS": tokenizer.eos_token_id,
        "PAD": tokenizer.pad_token_id,
        "BOS": tokenizer.bos_token_id,
        "UNK": tokenizer.unk_token_id,
    }
    for name, tid in special_token_map.items():
        if tid is not None:
            present = tid in token_counter
            occ = token_counter[tid]
            print(f"  Special token {name:<4} (ID: {tid:<7}): {'Present in data' if present else 'Missing from data'} ({occ:,} times)")
        else:
            print(f"  Special token {name:<4}: Not defined in tokenizer")

    # Extra Qwen special tokens
    extra_specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    for sym in extra_specials:
        tid = tokenizer.convert_tokens_to_ids(sym)
        if tid is not None and tid != tokenizer.unk_token_id:
            present = tid in token_counter
            occ = token_counter[tid]
            print(f"  Special token {sym:<15} (ID: {tid:<7}): {'Present in data' if present else 'Not used in data'} ({occ:,} times)")

    # Check 128 ASCII characters
    ascii_single_tokens: Dict[int, int] = {}
    ascii_missing_from_data: List[Tuple[int, str, int]] = []

    for c in range(128):
        char = chr(c)
        enc = tokenizer.encode(char, add_special_tokens=False)
        if len(enc) == 1:
            tid = enc[0]
            ascii_single_tokens[c] = tid
            if tid not in token_counter:
                ascii_missing_from_data.append((c, char, tid))
        else:
            print(f"  WARNING: ASCII char {c} ({repr(char)}) maps to {len(enc)} tokens: {enc}")

    print(f"\n  ASCII 0-127 Coverage : {len(ascii_single_tokens)}/128 characters have exact single-token IDs.")
    print(f"  ASCII Tokens in Data : {128 - len(ascii_missing_from_data)}/128 observed in training set.")
    if ascii_missing_from_data:
        print(f"  ASCII Tokens Absent  : {len(ascii_missing_from_data)} rare ASCII chars not in code dataset:")
        sample_missing = ascii_missing_from_data[:10]
        missing_str = ", ".join(f"{repr(ch)} (ID:{tid})" for _, ch, tid in sample_missing)
        print(f"    Sample: {missing_str}")
        print("    --> In pruned vocabulary, these MUST be explicitly preserved as base fallback bytes.")

    # 5. Parameter and VRAM Savings Calculation
    print("\n[5/5] Parameter & Memory Savings Analysis (Model Architecture & Activations):")
    print("=" * 85)

    vocab_candidates = [
        ("Current (Full Qwen2.5)", model_vocab_size),
        ("Candidate 32k (32,768)", 32768),
        ("Candidate 16k (16,384)", 16384),
        ("Candidate 8k  (8,192)", 8192),
    ]

    print(f"{'Configuration':<26} | {'Vocab Size':<12} | {'Untied Params':<15} | {'Tied Params':<14} | {'Logits (FP32)':<13}")
    print("-" * 85)

    for name, v in vocab_candidates:
        untied_p = 2 * v * d_model
        tied_p = 1 * v * d_model
        logits_bytes = batch_size * seq_len * v * 4  # FP32 logits
        logits_mb = logits_bytes / (1024**2)
        print(
            f"{name:<26} | {v:<12,d} | {untied_p / 1e6:>7.2f}M params  | "
            f"{tied_p / 1e6:>6.2f}M params | {logits_mb:>7.1f} MB"
        )
    print("-" * 85)

    # Detailed savings breakdown
    curr_untied = 2 * model_vocab_size * d_model
    curr_tied = 1 * model_vocab_size * d_model
    curr_logits_mb = (batch_size * seq_len * model_vocab_size * 4) / (1024**2)

    print("\nSAVINGS SUMMARY RELATIVE TO BASELINE (151,936 VOCAB):")
    for name, v in vocab_candidates[1:]:
        untied_p = 2 * v * d_model
        tied_p = 1 * v * d_model
        logits_mb = (batch_size * seq_len * v * 4) / (1024**2)

        saved_untied = curr_untied - untied_p
        saved_tied = curr_tied - tied_p
        saved_vram_weights_bf16 = (saved_untied * 2) / (1024**2)  # BF16 = 2 bytes
        saved_logits_mb = curr_logits_mb - logits_mb

        print(f"\n* {name}:")
        print(f"  - Untied parameters freed : {saved_untied / 1e6:.2f}M parameters ({saved_untied / curr_untied * 100:.1f}% reduction)")
        print(f"  - Tied parameters freed   : {saved_tied / 1e6:.2f}M parameters ({saved_tied / curr_tied * 100:.1f}% reduction)")
        print(f"  - Weight VRAM saved (BF16): {saved_vram_weights_bf16:.1f} MB (Untied) / {saved_vram_weights_bf16 / 2:.1f} MB (Tied)")
        print(f"  - Logits tensor VRAM saved: {saved_logits_mb:.1f} MB per forward/backward pass (from {curr_logits_mb:.1f} MB to {logits_mb:.1f} MB)")
        print(f"  - Softmax / CE Speedup    : ~{model_vocab_size / v:.1f}x faster vocabulary projection")

    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset vocabulary usage and pruning potential")
    parser.add_argument("--data_path", type=str, default="data/train_code.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-0.5B", help="Base model/tokenizer name")
    parser.add_argument("--d_model", type=int, default=896, help="Hidden embedding dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for logits memory calculation")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length for logits memory calculation")
    args = parser.parse_args()

    analyze_vocabulary(
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
