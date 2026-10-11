"""
Train Estonian & IT-Slang Specialist Micro-Model (eesti_keele_sl2ng)
=====================================================================
Trains an isolated ~2.9M parameter PureFeistelLM model with Reversible Associative Feistel (RAF)
recurrent memory (24 KB state, 0 bytes KV-cache) to translate colloquial Estonian, IT-slang,
and compound operations into formal neuro-symbolic execution contracts (RFC-002 Tier 2).

Target Loss Masking:
  Computes CrossEntropyLoss (ignore_index=-100) strictly on the contract JSON tokens
  and <|endoftext|>. All user instruction tokens up to and including <|contract|> are masked.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.models.pure_feistel_lm import PureFeistelLM

# Reproducibility
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# File paths
DATA_PATH = REPO_ROOT / "data" / "slang_train_10k.jsonl"
TOKENIZER_PATH = REPO_ROOT / "models" / "eesti_keele_sl2ng" / "tokenizer.json"
OUTPUT_CHECKPOINT = REPO_ROOT / "models" / "eesti_keele_sl2ng" / "slang_expert_feistel.pt"


# ==============================================================================
# Dataset and Collate with Target Loss Masking
# ==============================================================================

class SlangContractDataset(Dataset):
    """
    Dataset streaming structured instruction-to-contract sequences.
    Format: <|user|> {instruction} <|contract|> {contract_json} <|endoftext|>
    """

    def __init__(self, data_path: Path, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.contract_token_id = tokenizer.token_to_id("<|contract|>")
        self.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = tokenizer.token_to_id("<|pad|>")

        self.samples: List[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                inst = record["instruction"]
                contract_json = json.dumps(record["contract"], ensure_ascii=False)
                formatted = f"<|user|> {inst} <|contract|> {contract_json} <|endoftext|>"
                self.samples.append(formatted)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


def create_collate_fn(tokenizer: Tokenizer, contract_id: int, pad_id: int):
    """
    Collate function creating causal training pairs with Target Loss Masking.
    Masks all positions up to <|contract|> with -100 so the loss is computed
    exclusively on the contract JSON payload and <|endoftext|>.
    """
    def collate_fn(batch_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_ids = [tokenizer.encode(text).ids for text in batch_texts]
        max_len = max(len(ids) for ids in batch_ids)
        B = len(batch_ids)

        # Sequences are length (max_len - 1) for causal prediction
        seq_len = max_len - 1
        input_ids = torch.full((B, seq_len), pad_id, dtype=torch.long)
        targets = torch.full((B, seq_len), -100, dtype=torch.long)

        for i, ids in enumerate(batch_ids):
            inp = ids[:-1]
            tgt = ids[1:]

            # Locate contract token position
            c_idx = inp.index(contract_id) if contract_id in inp else 0

            input_ids[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            target_t = torch.tensor(tgt, dtype=torch.long)

            # Mask everything up to the contract token
            target_t[:c_idx] = -100
            targets[i, :len(tgt)] = target_t

        return input_ids, targets

    return collate_fn


# ==============================================================================
# Model Factory
# ==============================================================================

def build_slang_expert_model(
    vocab_size: int,
    d_model: int = 512,
    num_layers: int = 12,
    max_seq_len: int = 128,
    device: str = "cuda",
) -> PureFeistelLM:
    """
    Initializes the PureFeistelLM model for the eesti_keele_sl2ng micro-model.
    Uses RAF (Reversible Associative Feistel) memory with O(1) state and 0 bytes KV-cache.
    Enables gradient flow across all embedding and head layers.
    """
    model = PureFeistelLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        memory_type="raf",
        num_heads=4,
        key_dim=16,
        value_dim=16,
        torch_dtype=torch.float32,
        device=device,
        from_pretrained_qwen=False,
    )

    # Enable full gradient flow for training from scratch
    model.embed_tokens.requires_grad_(True)
    model.norm.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    model.feistel_backbone.requires_grad_(True)

    return model


# ==============================================================================
# Autoregressive Greedy Inference Evaluation
# ==============================================================================

def generate_contract_greedy(
    model: PureFeistelLM,
    tokenizer: Tokenizer,
    query: str,
    device: str = "cuda",
    max_new_tokens: int = 96,
) -> Tuple[str, Optional[Dict[str, Any]], bool]:
    """
    Performs deterministic greedy autoregressive generation with O(1) RAF state tracking.
    """
    model.eval()
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    # Prompt with user query and contract trigger (no trailing space to align with token 107 ' {"')
    prompt_str = f"<|user|> {query} <|contract|>"
    prompt_ids = tokenizer.encode(prompt_str).ids
    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    curr_tensor = prompt_tensor.clone()
    gen_token_ids: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = next_token.item()
            if tok_id == eos_id:
                break
            gen_token_ids.append(tok_id)
            curr_tensor = torch.cat([curr_tensor, next_token], dim=-1)

    raw_text = tokenizer.decode(gen_token_ids).strip()

    # Attempt JSON extraction
    parsed_json: Optional[Dict[str, Any]] = None
    is_valid_json = False

    # Extract clean JSON payload bounded by outermost curly braces
    start_brace = raw_text.find("{")
    end_brace = raw_text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_candidate = raw_text[start_brace : end_brace + 1]
        try:
            parsed_json = json.loads(json_candidate)
            is_valid_json = True
        except json.JSONDecodeError:
            is_valid_json = False
    else:
        try:
            parsed_json = json.loads(raw_text)
            is_valid_json = True
        except json.JSONDecodeError:
            is_valid_json = False

    return raw_text, parsed_json, is_valid_json


# ==============================================================================
# Main Training Loop
# ==============================================================================

def train_and_evaluate(export_package: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("TRAINING EESTI_KEELE_SL2NG SPECIALIST MICRO-MODEL (PUREFEISTELLM RAF)")
    print("=" * 80)
    print(f"Device        : {device}")
    if device == "cuda":
        print(f"GPU Model     : {torch.cuda.get_device_name(0)}")
    print(f"Dataset       : {DATA_PATH}")
    print(f"Tokenizer     : {TOKENIZER_PATH}")

    # Load tokenizer
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    contract_id = tokenizer.token_to_id("<|contract|>")
    pad_id = tokenizer.token_to_id("<|pad|>")

    print(f"Vocab size    : {vocab_size} tokens")

    # Hyperparameters
    d_model = 512
    num_layers = 12
    max_seq_len = 128
    batch_size = 32
    num_epochs = 5
    lr = 6e-4
    min_lr = 1e-5
    weight_decay = 0.01

    # Instantiate model
    print("\n[1/3] Initializing PureFeistelLM (~2.9M parameters, 12 layers, 24 KB RAF state)...")
    model = build_slang_expert_model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")

    # Dataset & DataLoader
    dataset = SlangContractDataset(DATA_PATH, tokenizer)
    collate_fn = create_collate_fn(tokenizer, contract_id, pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    total_steps = len(dataloader) * num_epochs
    warmup_steps = min(150, total_steps // 10)

    # Optimizer & LR Scheduler (Linear Warmup + Cosine Annealing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr / lr + 0.5 * (1.0 - min_lr / lr) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n[2/3] Training for {num_epochs} epochs ({total_steps} steps, batch_size={batch_size})...")
    print("-" * 80)

    start_time = time.perf_counter()
    global_step = 0
    epoch_losses: List[float] = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss_accum = 0.0
        step_count = 0
        epoch_start = time.perf_counter()

        for step, (input_ids, targets) in enumerate(dataloader, 1):
            global_step += 1
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, targets=targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss_accum += loss_val
            step_count += 1

            if global_step % 75 == 0 or global_step == total_steps:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - start_time
                print(
                    f"Epoch {epoch}/{num_epochs} | Step {global_step:4d}/{total_steps} | "
                    f"Loss: {loss_val:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}s"
                )

        mean_epoch_loss = epoch_loss_accum / max(1, step_count)
        epoch_losses.append(mean_epoch_loss)
        epoch_duration = time.perf_counter() - epoch_start
        print(f"--> Epoch {epoch} Completed in {epoch_duration:.1f}s | Mean Loss: {mean_epoch_loss:.4f}")
        print("-" * 80)

    total_duration = time.perf_counter() - start_time
    final_loss = epoch_losses[-1]
    print(f"\nTraining finished in {total_duration:.1f}s ({total_duration / 60:.2f} min).")
    print(f"Final Epoch Loss: {final_loss:.4f} (Target < 0.05 met: {final_loss < 0.05})")

    # Save model checkpoint
    print(f"\nSaving model checkpoint to: {OUTPUT_CHECKPOINT}")
    OUTPUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "memory_type": "raf",
            "num_heads": 4,
            "key_dim": 16,
            "value_dim": 16,
            "torch_dtype": "float32",
        },
        "tokenizer_path": str(TOKENIZER_PATH.relative_to(REPO_ROOT)),
        "epochs": num_epochs,
        "final_loss": final_loss,
        "epoch_losses": epoch_losses,
        "total_parameters": total_params,
        "train_duration_s": round(total_duration, 2),
    }
    torch.save(checkpoint_payload, OUTPUT_CHECKPOINT)
    print("Checkpoint saved successfully.")

    # Ensure declarative config.json is generated/updated with actual hyperparameters
    model_dir = OUTPUT_CHECKPOINT.parent
    config_path = model_dir / "config.json"
    config_data = {
        "name": "eesti_keele_sl2ng",
        "role": "language_expert",
        "weights": OUTPUT_CHECKPOINT.name,
        "tokenizer": TOKENIZER_PATH.name,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
        "memory_type": "raf",
        "num_heads": 4,
        "key_dim": 16,
        "value_dim": 16,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"Generated/updated declarative model config at: {config_path}")

    # Package export if requested
    if export_package:
        from runtime.registry import ModelRegistry
        dist_dir = REPO_ROOT / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        export_target = dist_dir / "eesti_keele_sl2ng.feist"
        print(f"\n[Export Package] Packaging model into {export_target}...")
        reg = ModelRegistry()
        pkg_res = reg.package_model("eesti_keele_sl2ng", output_path=export_target)
        print(f"[Export Package] Successfully packaged {pkg_res['model_name']} ({pkg_res['archive_size_mb']:.2f} MB) -> {pkg_res['archive_path']}")

    # ==========================================================================
    # Post-Training Benchmark Evaluation
    # ==========================================================================
    print("\n" + "=" * 80)
    print("[3/3] POST-TRAINING REAL-TIME CONTRACT INFERENCE EVALUATION")
    print("=" * 80)

    eval_queries = [
        "löö kokku 15 ja 25",
        "liida kolm arvu 1, 5 ja 6 lahuta sellest 2",
        "clämbi counter 105 vahemikku 0 ja 100",
        "viska xor 12 ja 4 vahel",
        "palun liida",
    ]

    all_passed = True
    for idx, query in enumerate(eval_queries, 1):
        raw_text, parsed_json, is_valid = generate_contract_greedy(
            model=model,
            tokenizer=tokenizer,
            query=query,
            device=device,
            max_new_tokens=96,
        )

        status_str = "SUCCESS (Valid JSON)" if is_valid else "FAILED (JSON Parse Error)"
        if not is_valid:
            all_passed = False

        print(f"Test {idx}: \"{query}\"")
        print(f"  Raw Output     : {raw_text}")
        print(f"  JSON Validity  : {status_str}")
        if is_valid and parsed_json:
            print(f"  Intent Name    : {parsed_json.get('intent_name')}")
            print(f"  Strategy       : {parsed_json.get('strategy')}")
            print(f"  Signature      : {parsed_json.get('signature')}")
            print(f"  Extracted Args : {parsed_json.get('args')}")
        print("-" * 80)

    if all_passed:
        print("\nALL 5 BENCHMARK INFERENCE EVALUATIONS GENERATED VALID CONTRACTS!")
    else:
        print("\nOne or more benchmark evaluations encountered a parsing failure.")
def evaluate_checkpoint(checkpoint_path: Path, device: str = "cuda") -> None:
    """Loads an existing checkpoint and runs the 5 benchmark tests."""
    print("=" * 80)
    print("EVALUATING EXISTING CHECKPOINT")
    print("=" * 80)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    model = PureFeistelLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        memory_type=cfg["memory_type"],
        num_heads=cfg["num_heads"],
        key_dim=cfg["key_dim"],
        value_dim=cfg["value_dim"],
        torch_dtype=torch.float32,
        device=device,
        from_pretrained_qwen=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    eval_queries = [
        "löö kokku 15 ja 25",
        "liida kolm arvu 1, 5 ja 6 lahuta sellest 2",
        "clämbi counter 105 vahemikku 0 ja 100",
        "viska xor 12 ja 4 vahel",
        "palun liida",
    ]

    all_passed = True
    for idx, query in enumerate(eval_queries, 1):
        raw_text, parsed_json, is_valid = generate_contract_greedy(
            model=model,
            tokenizer=tokenizer,
            query=query,
            device=device,
            max_new_tokens=96,
        )

        status_str = "SUCCESS (Valid JSON)" if is_valid else "FAILED (JSON Parse Error)"
        if not is_valid:
            all_passed = False

        print(f"Test {idx}: \"{query}\"")
        print(f"  Raw Output     : {raw_text}")
        print(f"  JSON Validity  : {status_str}")
        if is_valid and parsed_json:
            print(f"  Intent Name    : {parsed_json.get('intent_name')}")
            print(f"  Strategy       : {parsed_json.get('strategy')}")
            print(f"  Signature      : {parsed_json.get('signature')}")
            print(f"  Extracted Args : {parsed_json.get('args')}")
        print("-" * 80)

    if all_passed:
        print("\nALL 5 BENCHMARK INFERENCE EVALUATIONS GENERATED VALID CONTRACTS!")
    else:
        print("\nOne or more benchmark evaluations encountered a parsing failure.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Estonian & IT-Slang Specialist Micro-Model")
    parser.add_argument("--eval-only", action="store_true", help="Run benchmark evaluation on existing checkpoint")
    parser.add_argument(
        "--export_package",
        action="store_true",
        help="Automatically generate config.json and package model into dist/<name>.feist after training",
    )
    args = parser.parse_args()

    if args.eval_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluate_checkpoint(OUTPUT_CHECKPOINT, device=device)
    else:
        train_and_evaluate(export_package=args.export_package)

