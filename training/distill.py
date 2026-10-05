"""
Feistel Distillation Trainer (PureFeistelLM Backbone Distillation)
=================================================================
Trains the 12-layer ReversibleFeistelNetwork backbone of PureFeistelLM to mimic
the intermediate representations of frozen Qwen2.5-Coder-0.5B teacher model.

Key Architectural Invariants:
  - Teacher is frozen in bfloat16 eval mode.
  - Student embed_tokens, norm, and lm_head are frozen.
  - ONLY Student backbone weights (2.42M parameters) are trained.
  - Loss operates directly on hidden states (MSE + Cosine Similarity) without
    computing massive 151k vocabulary cross-entropy during representation alignment.
  - Exactly 0 bytes KV-cache throughout training and inference.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2ForCausalLM

from core.pure_feistel_lm import PureFeistelLM
from training.dataset import CodeDistillationDataset, generate_default_code_dataset


class FeistelDistillationTrainer:
    """
    Automated trainer for representation distillation from Qwen2.5-Coder to PureFeistelLM.
    """

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
        student_layers: int = 12,
        lr: float = 8e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        seq_len: int = 128,
        steps: int = 2000,
        data_path: str = "data/train_code.jsonl",
        save_dir: str = "checkpoints",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.teacher_model_name = teacher_model_name
        self.student_layers = student_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps = steps
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        print("=" * 75)
        print("INITIALIZING FEISTEL HYBRID DISTILLATION TRAINER")
        print("=" * 75)
        print(f"Target Device        : {self.device}")
        print(f"Teacher Model        : {self.teacher_model_name}")
        print(f"Student Layers       : {self.student_layers} Feistel layers")
        print(f"Distillation Steps   : {self.steps}")
        print(f"Batch Size / SeqLen  : {self.batch_size} / {self.seq_len} tokens")
        print(f"Learning Rate        : {self.lr} (CosineAnnealingLR eta_min=5e-5)")
        print(f"Loss Formulation     : 0.2 * loss_rep (MSE + 0.5*CosDist) + 1.0 * loss_ce")
        print(f"Dataset Path         : {self.data_path}")
        print("=" * 75)

        # 1. Load Tokenizer
        print("\n[1/4] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)

        # 2. Ensure Dataset and DataLoader
        print("\n[2/4] Preparing dataset and continuous token packing...")
        if not os.path.exists(self.data_path):
            generate_default_code_dataset(self.data_path, target_count=150)
        
        self.dataset = CodeDistillationDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(f"Loaded {len(self.dataset)} packed chunks of {self.seq_len} tokens each.")

        # 3. Load Teacher Model (Frozen)
        print("\n[3/4] Loading frozen Teacher model (Qwen2.5-Coder-0.5B)...")
        self.teacher = Qwen2ForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 4. Initialize Student Model (PureFeistelLM) sharing embeddings/norm/lm_head
        print("\n[4/4] Initializing Student model (PureFeistelLM)...")
        self.student = PureFeistelLM(
            vocab_size=self.teacher.config.vocab_size,
            d_model=self.teacher.config.hidden_size,
            num_layers=self.student_layers,
            max_seq_len=max(512, self.seq_len),
            layer_scale=0.05,
            torch_dtype=torch.bfloat16,
            device=str(self.device),
            from_pretrained_qwen=False,
            embed_tokens=self.teacher.model.embed_tokens,
            norm=self.teacher.model.norm,
            lm_head=self.teacher.lm_head,
            tokenizer=self.tokenizer,
        )

        # Strictly freeze everything except student backbone
        self.student.embed_tokens.requires_grad_(False)
        self.student.norm.requires_grad_(False)
        self.student.lm_head.requires_grad_(False)
        self.student.backbone.requires_grad_(True)

        trainable_params = sum(p.numel() for p in self.student.backbone.parameters() if p.requires_grad)
        total_student_params = sum(p.numel() for p in self.student.parameters())
        print(f"Student Total Params    : {total_student_params / 1e6:.2f}M")
        print(f"Trainable Backbone Params: {trainable_params / 1e6:.2f}M ({trainable_params:,} weights)")

        # Optimizer and Cosine Scheduler
        self.optimizer = torch.optim.AdamW(
            self.student.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.steps,
            eta_min=5e-5,
        )

    def train(self) -> Dict[str, Any]:
        """
        Executes the hybrid representation & cross-entropy distillation loop.
        """
        print("\n" + "=" * 85)
        print("STARTING HYBRID DISTILLATION TRAINING LOOP")
        print("=" * 85)
        print(
            f"{'Step':<8} | {'TotalLoss':<10} | {'RepLoss':<9} | {'CELoss':<9} | "
            f"{'CosSim':<8} | {'LR':<9} | {'Throughput':<12}"
        )
        print("-" * 85)

        data_iter = itertools.cycle(self.dataloader)
        history: Dict[str, list] = {
            "steps": [],
            "loss_total": [],
            "loss_rep": [],
            "loss_ce": [],
            "cos_sim": [],
            "lr": [],
            "tokens_per_sec": [],
            "vram_mb": [],
        }

        initial_metrics: Optional[Tuple[float, float, float, float]] = None
        final_metrics: Optional[Tuple[float, float, float, float]] = None

        t_total_start = time.perf_counter()

        for step in range(1, self.steps + 1):
            t_step_start = time.perf_counter()
            batch = next(data_iter).to(self.device)

            # 1. Teacher forward (frozen)
            with torch.no_grad():
                teacher_out = self.teacher.model(input_ids=batch, return_dict=True)
                teacher_hs = teacher_out.last_hidden_state  # (B, L, 896)

            # 2. Student forward through backbone & norm
            student_hs = self.student.forward_backbone(batch, with_norm=True)

            # A. Geometric representation loss
            loss_mse = F.mse_loss(student_hs, teacher_hs)
            cos_sim = F.cosine_similarity(student_hs, teacher_hs, dim=-1).mean()
            loss_cos = 1.0 - cos_sim
            loss_rep = loss_mse + 0.5 * loss_cos

            # B. Next-token syntax cross-entropy loss via frozen lm_head
            student_logits = self.student.lm_head(student_hs)  # (B, L, V)
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Combined total loss
            loss_total = 0.2 * loss_rep + 1.0 * loss_ce

            # 4. Backward & Step
            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.student.backbone.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Timing & Memory
            step_duration = max(1e-6, time.perf_counter() - t_step_start)
            tok_per_sec = (self.batch_size * self.seq_len) / step_duration
            vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

            loss_total_val = loss_total.item()
            loss_rep_val = loss_rep.item()
            loss_ce_val = loss_ce.item()
            cos_val = cos_sim.item()
            curr_lr = self.optimizer.param_groups[0]["lr"]

            if initial_metrics is None:
                initial_metrics = (loss_total_val, loss_rep_val, loss_ce_val, cos_val)
            final_metrics = (loss_total_val, loss_rep_val, loss_ce_val, cos_val)

            # Logging every 100 steps
            if step == 1 or step % 100 == 0 or step == self.steps:
                print(
                    f"{step:<8d} | {loss_total_val:<10.4f} | {loss_rep_val:<9.4f} | "
                    f"{loss_ce_val:<9.4f} | {cos_val:<8.4f} | {curr_lr:<9.2e} | {tok_per_sec:<7.0f} tok/s"
                )

            history["steps"].append(step)
            history["loss_total"].append(loss_total_val)
            history["loss_rep"].append(loss_rep_val)
            history["loss_ce"].append(loss_ce_val)
            history["cos_sim"].append(cos_val)
            history["lr"].append(curr_lr)
            history["tokens_per_sec"].append(tok_per_sec)
            history["vram_mb"].append(vram_mb)

            # Checkpoint every 500 steps
            if step % 500 == 0:
                ckpt_path = os.path.join(self.save_dir, f"pure_feistel_step_{step}.pt")
                self.save_checkpoint(ckpt_path, step, loss_total_val, loss_rep_val, loss_ce_val, cos_val)

        total_elapsed = time.perf_counter() - t_total_start
        mean_tok_sec = sum(history["tokens_per_sec"]) / max(1, len(history["tokens_per_sec"]))

        # Save final checkpoint
        final_ckpt_path = os.path.join(self.save_dir, "pure_feistel_final.pt")
        self.save_checkpoint(final_ckpt_path, self.steps, final_metrics[0], final_metrics[1], final_metrics[2], final_metrics[3])

        print("-" * 85)
        print("HYBRID DISTILLATION TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 85)
        print(f"Total Duration       : {total_elapsed:.2f} s")
        print(f"Mean Throughput      : {mean_tok_sec:.0f} tokens/s")
        print(f"Initial Metrics      : Total={initial_metrics[0]:.4f} | Rep={initial_metrics[1]:.4f} | CE={initial_metrics[2]:.4f} | CosSim={initial_metrics[3]:.4f}")
        print(f"Final Metrics (2000) : Total={final_metrics[0]:.4f} | Rep={final_metrics[1]:.4f} | CE={final_metrics[2]:.4f} | CosSim={final_metrics[3]:.4f}")
        print(f"Final Checkpoint     : {final_ckpt_path}")
        print("=" * 85 + "\n")

        return {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "total_elapsed_sec": total_elapsed,
            "mean_tokens_per_sec": mean_tok_sec,
            "final_checkpoint": final_ckpt_path,
            "history": history,
        }

    def save_checkpoint(
        self,
        path: str,
        step: int,
        loss_total: float,
        loss_rep: float,
        loss_ce: float,
        cos_sim: float,
    ) -> None:
        """
        Saves student backbone parameters and training metadata.
        """
        checkpoint_data = {
            "step": step,
            "backbone_state_dict": self.student.backbone.state_dict(),
            "num_layers": self.student_layers,
            "d_model": self.teacher.config.hidden_size,
            "loss_total": loss_total,
            "loss_rep": loss_rep,
            "loss_ce": loss_ce,
            "cos_sim": cos_sim,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint_data, path)
        print(
            f"  [Checkpoint] Saved step {step} to {path} "
            f"(TotalLoss: {loss_total:.4f}, CELoss: {loss_ce:.4f}, CosSim: {cos_sim:.4f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PureFeistelLM Hybrid Distillation Trainer")
    parser.add_argument("--data_path", type=str, default="data/train_code.jsonl", help="Path to JSONL code dataset")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length in tokens")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--student_layers", type=int, default=12, help="Number of Feistel layers in student")
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    trainer = FeistelDistillationTrainer(
        teacher_model_name="Qwen/Qwen2.5-Coder-0.5B",
        student_layers=args.student_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        data_path=args.data_path,
        save_dir=args.save_dir,
    )
    trainer.train()
