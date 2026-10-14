"""
Train Slang & Estengrish BPE Tokenizer (4,096 Vocabulary)
=========================================================
Trains an isolated 4,096-token Byte Pair Encoding (BPE) tokenizer on the
synthetic Estonian / morphology / IT-slang / tool dataset (data/slang_train_10k.jsonl).
This decouples the eesti_keele_sl2ng micro-model vocabulary from the 16k code vocabulary,
guaranteeing efficient byte-level token compression for Estonian morphosyntax and JSON contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Reconfigure stdout/stderr to UTF-8 for console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "slang_train_10k.jsonl"
OUTPUT_DIR = REPO_ROOT / "models" / "eesti_keele_sl2ng"
TOKENIZER_JSON = OUTPUT_DIR / "tokenizer.json"

VOCAB_SIZE = 4096
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>", "<|user|>", "<|contract|>"]


def dataset_text_iterator(file_path: Path) -> Iterator[str]:
    """
    Streams text records from the JSONL dataset.
    Each sample contains instruction and formatted contract to ensure
    the tokenizer learns both Estonian colloquial morphosyntax and contract JSON keys.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instruction = record.get("instruction", "")
            contract = record.get("contract", {})
            contract_str = json.dumps(contract, ensure_ascii=False)
            formatted_text = f"<|user|> {instruction} <|contract|> {contract_str} <|endoftext|>\n"
            yield formatted_text


def train_tokenizer() -> Tokenizer:
    """Configures and trains the BPE tokenizer on the training stream."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at: {DATA_PATH}")

    print("=" * 80)
    print("TRAINING SLANG & ESTENGRISH BPE TOKENIZER (4,096 VOCABULARY)")
    print("=" * 80)
    print(f"Dataset path   : {DATA_PATH}")
    print(f"Target vocab   : {VOCAB_SIZE}")
    print(f"Min frequency  : {MIN_FREQUENCY}")
    print(f"Special tokens : {SPECIAL_TOKENS}")
    print("=" * 80)

    # Initialize BPE model with unknown token fallback
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(dataset_text_iterator(DATA_PATH), trainer=trainer)
    train_duration = time.perf_counter() - t0

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\nTraining completed in {train_duration:.2f} seconds.")
    print(f"Actual vocabulary size: {actual_vocab_size:,} tokens.")

    # Save tokenizer artifacts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_JSON))
    print(f"Saved tokenizer to: {TOKENIZER_JSON}")

    # Save vocabulary dictionary for inspection
    vocab_file = OUTPUT_DIR / "vocab.json"
    vocab = tokenizer.get_vocab()
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary dump to: {vocab_file}")

    # Save training metadata
    metadata_file = OUTPUT_DIR / "metadata.json"
    metadata = {
        "model_type": "BPE",
        "actual_vocab_size": actual_vocab_size,
        "target_vocab_size": VOCAB_SIZE,
        "min_frequency": MIN_FREQUENCY,
        "special_tokens": SPECIAL_TOKENS,
        "source_dataset": str(DATA_PATH.relative_to(REPO_ROOT)),
        "train_duration_s": round(train_duration, 4),
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {metadata_file}\n")

    return tokenizer


def verify_tokenizer(tokenizer_path: Path) -> None:
    """Loads saved tokenizer and executes verification suite."""
    print("=" * 80)
    print("VERIFYING TRAINED TOKENIZER ON BENCHMARK SENTENCES")
    print("=" * 80)

    loaded_tok = Tokenizer.from_file(str(tokenizer_path))

    test_sentences = [
        "ou pane 15 ja 25 kokku plz",
        "lahuta 12 arvust 50",
        "arvuta 200 eurot 15% allahindlusega",
        "mul oli 100 eurot ja kulutasin 35 eurot",
        "leia suurim ühistegur arvudele 48 ja 18",
    ]

    all_matched = True

    for idx, sentence in enumerate(test_sentences, 1):
        encoded = loaded_tok.encode(sentence)
        tokens = encoded.tokens
        ids = encoded.ids
        decoded = loaded_tok.decode(ids)
        roundtrip_match = (decoded == sentence)

        if not roundtrip_match:
            all_matched = False

        status_str = "MATCH (100% OK)" if roundtrip_match else "MISMATCH"

        print(f"Test {idx}: \"{sentence}\"")
        print(f"  Token count  : {len(tokens)}")
        print(f"  Token IDs    : {ids}")
        print(f"  Tokens       : {tokens}")
        print(f"  Decoded      : \"{decoded}\"")
        print(f"  Roundtrip    : {status_str}")
        print("-" * 80)

    assert all_matched, "One or more roundtrip decodings failed!"
    print("ALL 5 BENCHMARK SENTENCES PASSED 100% ROUNDTRIP VERIFICATION!\n")


def main() -> None:
    train_tokenizer()
    verify_tokenizer(TOKENIZER_JSON)


if __name__ == "__main__":
    main()
