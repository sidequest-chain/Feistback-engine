"""
Train Dynamic FeistelHybrid Model (LOAD_ARG Register Synthesis)
==============================================================
Trains a 16-layer FeistelHolographicLM (d_model=128) on dynamic mathematical
operations with abstract register lookups (<ARG:0>, <ARG:1>).

Saves model checkpoint to:
  checkpoints/feistel_dynamic_hybrid.pt
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import FeistelHolographicLM
from runner import HybridTokenizer
from trainer.dynamic_dataset import generate_dynamic_dataset


def prepare_tensors(
    dataset: List[Dict[str, Any]],
    tokenizer: HybridTokenizer,
    max_seq_len: int = 112,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes dataset sequences with causal prompt masking (ignore_index=0).
    The prompt includes the <EXEC_VM> trigger so target begins cleanly at <OP:LOAD_ARG>.
    """
    total_len = max_seq_len + 1
    pad_id = tokenizer.vocab[tokenizer.pad_token]

    inputs_list: List[List[int]] = []
    targets_list: List[List[int]] = []

    for item in dataset:
        full_text = item["full_sequence"]
        prompt_prefix = f"{item['prompt']}<EXEC_VM> "

        enc_full = tokenizer.encode(full_text, add_special=True)
        enc_prompt = tokenizer.encode(prompt_prefix, add_special=True)
        
        # enc_prompt has <s> and </s>, so real prompt token count is len(enc_prompt) - 2.
        num_mask = len(enc_prompt) - 2

        if len(enc_full) > total_len:
            enc_full = enc_full[:total_len]
        padded = enc_full + [pad_id] * (total_len - len(enc_full))

        inp = padded[:-1]
        tgt = list(padded[1:])
        for i in range(min(num_mask, len(tgt))):
            tgt[i] = 0

        inputs_list.append(inp)
        targets_list.append(tgt)

    inp_tensor = torch.tensor(inputs_list, dtype=torch.long, device=device)
    tgt_tensor = torch.tensor(targets_list, dtype=torch.long, device=device)
    return inp_tensor, tgt_tensor


def train_dynamic_hybrid(
    epochs: int = 200,
    batch_size: int = 128,
    num_samples: int = 2000,
    save_path: Optional[str] = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  TRAINING DYNAMIC FEISTELHYBRID WITH LOAD_ARG REGISTERS")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    tokenizer = HybridTokenizer(num_registers=4)
    print(f"Tokenizer Vocab Size: {tokenizer.vocab_size} (Strictly abstract registers)")

    # 1. Generate dynamic training samples
    print(f"\nGenerating {num_samples} dynamic mathematical training samples...")
    train_data = generate_dynamic_dataset(num_samples=num_samples, seed=42, min_val=1, max_val=500)
    print(f"Generated {len(train_data)} samples.")

    max_seq_len = 112
    inp_tensor, tgt_tensor = prepare_tensors(train_data, tokenizer, max_seq_len=max_seq_len, device=device)
    print(f"Training Tensor Shape: {inp_tensor.shape}")

    # 2. Initialize 16-layer FeistelHolographicLM
    model = FeistelHolographicLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_layers=16,
        max_seq_len=max_seq_len,
        activation="silu",
        layer_scale=1.0
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count:,} weights (16 layers, d_model=128)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_samples = inp_tensor.shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nStarting {epochs}-epoch training (Batch size: {batch_size}, Batches/epoch: {num_batches})...")
    start_time = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(n_samples, device=device)
        total_loss = 0.0

        for b in range(num_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            b_inp = inp_tensor[b_idx]
            b_tgt = tgt_tensor[b_idx]

            optimizer.zero_grad()
            _, loss = model(b_inp, b_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(b_idx)

        scheduler.step()
        epoch_loss = total_loss / n_samples

        if ep % 20 == 0 or ep == epochs:
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0.0
            print(f"  Epoch {ep:3d}/{epochs} | Loss: {epoch_loss:8.6f} | VRAM: {vram_mb:.1f} MB")

    total_time = time.perf_counter() - start_time
    print(f"\nTraining Complete in {total_time:.2f}s ({(total_time/epochs)*1000.0:.2f} ms/epoch) | Final Loss: {epoch_loss:.6f}")

    # 3. Save Checkpoint
    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "checkpoints", "feistel_dynamic_hybrid.pt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "d_model": 128,
        "num_layers": 16,
        "max_seq_len": max_seq_len,
        "epochs": epochs,
        "final_loss": epoch_loss
    }, save_path)
    print(f"Saved Checkpoint to: {save_path}")


if __name__ == "__main__":
    train_dynamic_hybrid()
