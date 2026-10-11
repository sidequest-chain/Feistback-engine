"""
Vocabulary Pruning Tool for PureFeistelLM (151k -> 16k)
=======================================================
Prunes Qwen2.5-Coder vocabulary from 151,936 down to 16,384 tokens:
  1. Preserves all special tokens (EOS, PAD, BOS, <|endoftext|>, <|im_start|>, <|im_end|>).
  2. Preserves all 256 base byte/ASCII tokens for byte-level BPE universality.
  3. Preserves all 11,485 unique tokens observed in data/train_code.jsonl by frequency.
  4. Fills the remaining slots up to 16,384 with Qwen's most frequent BPE merges (lowest IDs).
  5. Slices pre-trained Qwen2.5-Coder-0.5B embedding and lm_head weight matrices [16384, 896].
  6. Saves vocab_map.pt and pruned_weights.pt into checkpoints/vocab_16k/.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM


TARGET_VOCAB_SIZE = 16384
MODEL_VOCAB_SIZE = 151936
HIDDEN_DIM = 896


class PrunedVocabMap:
    """
    Lightweight bidirectional mapping helper between original Qwen token IDs (0..151935)
    and pruned 16k token IDs (0..16383).
    """

    def __init__(self, vocab_map_path: str = "models/logic_feistel_clean/vocab_16k/vocab_map.pt") -> None:
        data = torch.load(vocab_map_path, map_location="cpu")
        self.old_to_new: torch.Tensor = data["old_to_new"]
        self.new_to_old: torch.Tensor = data["new_to_old"]
        self.target_vocab_size: int = data.get("target_vocab_size", TARGET_VOCAB_SIZE)

    def encode_old_to_new(self, old_ids: List[int]) -> List[int]:
        return [self.old_to_new[oid].item() for oid in old_ids]

    def decode_new_to_old(self, new_ids: List[int]) -> List[int]:
        return [self.new_to_old[nid].item() for nid in new_ids]


def build_pruned_token_list(
    tokenizer: AutoTokenizer,
    data_path: str = "data/train_code.jsonl",
    target_vocab_size: int = TARGET_VOCAB_SIZE,
) -> Tuple[List[int], Dict[str, int]]:
    """
    Constructs the exact list of target_vocab_size tokens in priority order:
      a) Special tokens
      b) Base byte/ASCII fallback tokens (0..255)
      c) All unique tokens from training code by frequency
      d) Top frequent tokens from Qwen vocabulary (lowest token IDs)
    """
    selected_set: Set[int] = set()
    ordered_list: List[int] = []
    stats: Dict[str, int] = {}

    def add_token(tid: int) -> bool:
        if 0 <= tid < MODEL_VOCAB_SIZE and tid not in selected_set:
            selected_set.add(tid)
            ordered_list.append(tid)
            return True
        return False

    # A. Special tokens
    special_candidates = [
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
    ]
    for sym in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(sym)
            if tid is not None and tid != tokenizer.unk_token_id:
                special_candidates.append(tid)
        except Exception:
            pass

    count_specials = 0
    for tid in special_candidates:
        if tid is not None and add_token(tid):
            count_specials += 1
    stats["special_tokens"] = count_specials

    # B. Base byte/ASCII tokens (chr 0..255)
    count_base_bytes = 0
    for c in range(256):
        try:
            enc = tokenizer.encode(chr(c), add_special_tokens=False)
            for tid in enc:
                if add_token(tid):
                    count_base_bytes += 1
        except Exception:
            pass
    stats["base_byte_tokens"] = count_base_bytes

    # C. All unique tokens from train_code.jsonl in frequency order
    token_counter: Counter[int] = Counter()
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            code = entry.get("code") or entry.get("text", "")
            if code:
                tokens = tokenizer.encode(code, add_special_tokens=False)
                token_counter.update(tokens)

    # Ensure EOS token counted
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151643
    token_counter[eos_id] += 1

    count_data_tokens = 0
    for tid, _ in token_counter.most_common():
        if add_token(tid):
            count_data_tokens += 1
    stats["data_unique_tokens"] = count_data_tokens

    # D. Fill remaining slots with lowest Qwen BPE IDs (most frequent merges)
    count_filled = 0
    for tid in range(len(tokenizer)):
        if len(ordered_list) >= target_vocab_size:
            break
        if add_token(tid):
            count_filled += 1
    stats["filled_from_qwen"] = count_filled

    assert len(ordered_list) == target_vocab_size, (
        f"Expected {target_vocab_size} tokens, but collected {len(ordered_list)}!"
    )
    return ordered_list, stats


def prune_and_save_weights(
    selected_tokens: List[int],
    output_dir: str = "models/logic_feistel_clean/vocab_16k",
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    d_model: int = HIDDEN_DIM,
) -> None:
    """
    Slices embedding table and lm_head from Qwen2.5-Coder-0.5B using selected_tokens,
    and saves vocab_map.pt and pruned_weights.pt.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_vocab_size = len(selected_tokens)

    # 1. Build Index Tensors
    print("\n[1/3] Building bidirectional index mappings...")
    old_to_new = torch.full((MODEL_VOCAB_SIZE,), -1, dtype=torch.long)
    new_to_old = torch.tensor(selected_tokens, dtype=torch.long)

    for new_idx, old_idx in enumerate(selected_tokens):
        old_to_new[old_idx] = new_idx

    vocab_map_data = {
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "selected_tokens": selected_tokens,
        "target_vocab_size": target_vocab_size,
        "model_vocab_size": MODEL_VOCAB_SIZE,
    }
    vocab_map_path = os.path.join(output_dir, "vocab_map.pt")
    torch.save(vocab_map_data, vocab_map_path)
    print(f"  Saved vocab mapping to: {vocab_map_path} ({os.path.getsize(vocab_map_path) / 1024:.1f} KB)")

    # 2. Load Pretrained Qwen Weights & Slice
    print(f"\n[2/3] Loading pretrained weights from '{model_name}' to slice embedding & lm_head...")
    qwen = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    full_embed_weights = qwen.model.embed_tokens.weight.data  # [151936, 896]
    full_lm_head_weights = qwen.lm_head.weight.data          # [151936, 896]

    print(f"  Full embed_tokens shape: {list(full_embed_weights.shape)}")
    print(f"  Full lm_head shape     : {list(full_lm_head_weights.shape)}")

    # Slice using new_to_old indices
    pruned_embed_weights = full_embed_weights[new_to_old].clone()
    pruned_lm_head_weights = full_lm_head_weights[new_to_old].clone()

    print(f"  Pruned embed_tokens shape: {list(pruned_embed_weights.shape)}")
    print(f"  Pruned lm_head shape     : {list(pruned_lm_head_weights.shape)}")

    pruned_weights_data = {
        "embed_tokens": pruned_embed_weights,
        "lm_head": pruned_lm_head_weights,
        "vocab_size": target_vocab_size,
        "d_model": d_model,
        "dtype": str(pruned_embed_weights.dtype),
    }
    pruned_weights_path = os.path.join(output_dir, "pruned_weights.pt")
    torch.save(pruned_weights_data, pruned_weights_path)
    print(f"  Saved pruned weights to: {pruned_weights_path} ({os.path.getsize(pruned_weights_path) / (1024**2):.2f} MB)")

    # Explicitly clean up memory
    del qwen
    del full_embed_weights
    del full_lm_head_weights


def verify_pruned_vocabulary(
    vocab_map_path: str = "models/logic_feistel_clean/vocab_16k/vocab_map.pt",
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
) -> None:
    """
    Verifies that sample code tokenizes, maps to 16k tokens, and round-trips
    back to the exact identical original text without any loss of fidelity.
    """
    print("\n[3/3] Verifying bidirectional token conversion and reconstruction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_map = PrunedVocabMap(vocab_map_path)

    test_samples = [
        "def is_even(n: int) -> bool:\n    return n % 2 == 0",
        "def fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    return 1",
        "for i in range(10):\n    print(f'val: {i}')",
    ]

    for idx, sample_text in enumerate(test_samples, start=1):
        print(f"\n  --- Test Case {idx} ---")
        print(f"  Input Text: {repr(sample_text)}")

        # 1. Encode with original tokenizer
        orig_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        print(f"  Original Qwen IDs ({len(orig_ids)} tokens): {orig_ids}")

        # 2. Map to pruned 16k IDs
        new_ids = vocab_map.encode_old_to_new(orig_ids)
        print(f"  Pruned 16k IDs    ({len(new_ids)} tokens): {new_ids}")

        # Verify all tokens are within [0, 16383] and mapped correctly
        assert all(0 <= nid < TARGET_VOCAB_SIZE for nid in new_ids), (
            f"Found unmapped token (-1) in new_ids: {new_ids}"
        )

        # 3. Reconstruct back to old IDs
        reconstructed_orig_ids = vocab_map.decode_new_to_old(new_ids)
        assert reconstructed_orig_ids == orig_ids, (
            f"Reconstructed IDs {reconstructed_orig_ids} did not match original IDs {orig_ids}!"
        )

        # 4. Decode reconstructed IDs
        decoded_text = tokenizer.decode(reconstructed_orig_ids)
        assert decoded_text == sample_text, (
            f"Decoded text {repr(decoded_text)} does not match original {repr(sample_text)}!"
        )

        print("  Reconstruction: EXACT BIT-ACCURATE MATCH (100% Fidelity)")

    print("\n" + "=" * 85)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 85)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune Qwen vocabulary to 16,384 tokens")
    parser.add_argument("--data_path", type=str, default="data/train_code.jsonl", help="Dataset path")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-0.5B", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="models/logic_feistel_clean/vocab_16k", help="Output directory")
    parser.add_argument("--target_vocab_size", type=int, default=TARGET_VOCAB_SIZE, help="Target vocabulary size")
    args = parser.parse_args()

    print("=" * 85)
    print("QWEN2.5-CODER VOCABULARY PRUNING SUITE (151k -> 16k)")
    print("=" * 85)
    print(f"Dataset Path        : {args.data_path}")
    print(f"Base Model          : {args.model_name}")
    print(f"Target Vocab Size   : {args.target_vocab_size:,}")
    print(f"Output Directory    : {args.output_dir}")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    selected_tokens, stats = build_pruned_token_list(
        tokenizer=tokenizer,
        data_path=args.data_path,
        target_vocab_size=args.target_vocab_size,
    )

    print("\nVocabulary Composition Breakdown:")
    for category, count in stats.items():
        print(f"  - {category:<22}: {count:>6,d} tokens")
    print(f"  Total Selected Tokens : {len(selected_tokens):>6,d} tokens")

    prune_and_save_weights(
        selected_tokens=selected_tokens,
        output_dir=args.output_dir,
        model_name=args.model_name,
        d_model=HIDDEN_DIM,
    )

    verify_pruned_vocabulary(
        vocab_map_path=os.path.join(args.output_dir, "vocab_map.pt"),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
