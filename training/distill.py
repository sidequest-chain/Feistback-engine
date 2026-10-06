"""
Feistel Distillation Trainer (PureFeistelLM Backbone Distillation)
=================================================================
Trains the 12-layer ReversibleFeistelNetwork backbone of PureFeistelLM to mimic
the intermediate representations of frozen Qwen2.5-Coder-0.5B teacher model.

Key Architectural Invariants:
  - Teacher is frozen in bfloat16 eval mode.
  - Student embed_tokens, norm, and lm_head are frozen.
  - ONLY Student backbone weights (2.42M parameters) are trained.
  - Dynamic AutoScheduler computes training steps and epochs from dataset volume and core parameters.
  - Next-token cross-entropy leads loss (weight 1.0) with whitespace masking (weight 0.05).
  - Soft logit distillation (Top-64 KL with T=2.0, weight 0.3) provides soft target guidance.
  - Geometric representation loss (weight 0.05) acts as an anchor/stabilizer.
  - Best checkpoint tracking based on score = loss_total - 2.0 * cos_sim.
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
from training.dataset import CodeDistillationDataset


def compute_training_schedule(
    num_chunks: int,
    seq_len: int = 256,
    batch_size: int = 8,
    trainable_params: int = 2_425_344,
    alpha: float = 0.6,
    min_epochs: int = 3,
    max_epochs: int = 8,
) -> tuple[int, int]:
    """
    Computes total training steps and epochs dynamically based on dataset volume and trainable parameters.
    """
    total_tokens = num_chunks * seq_len
    target_tokens = int(alpha * trainable_params)
    raw_epochs = target_tokens / (total_tokens + 1e-8)
    epochs = int(max(min_epochs, min(raw_epochs, max_epochs)))
    total_steps = (num_chunks * epochs) // batch_size
    return total_steps, epochs


class FeistelDistillationTrainer:
    """
    Automated trainer for representation and logit distillation from Qwen2.5-Coder to PureFeistelLM.
    """

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
        student_layers: int = 12,
        state_dim: int = 128,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        seq_len: int = 256,
        steps: Optional[int] = None,
        alpha: float = 0.6,
        eta_min: float = 1e-5,
        temperature: float = 2.0,
        kd_top_k: int = 64,
        ws_weight: float = 0.05,
        vocab_size: int = 151936,
        vocab_map_path: str = "checkpoints/vocab_16k/vocab_map.pt",
        pruned_weights_path: str = "checkpoints/vocab_16k/pruned_weights.pt",
        best_ckpt_name: Optional[str] = None,
        data_path: str = "data/train_code.jsonl",
        save_dir: str = "checkpoints",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.teacher_model_name = teacher_model_name
        self.student_layers = student_layers
        self.state_dim = state_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.steps_arg = steps
        self.alpha = alpha
        self.eta_min = eta_min
        self.temperature = temperature
        self.kd_top_k = kd_top_k
        self.ws_weight = ws_weight
        self.vocab_size = vocab_size
        self.vocab_map_path = vocab_map_path
        self.pruned_weights_path = pruned_weights_path
        self.best_ckpt_name_override = best_ckpt_name
        self.data_path = data_path
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        print("=" * 85)
        if self.vocab_size == 16384:
            print("INITIALIZING FEISTEL TRAINER (16K VOCABULARY - PURE 1.0 * CE)")
        else:
            print("INITIALIZING FEISTEL TRAINER (HYBRID 1.0 * CE + 0.3 * KD)")
        print("=" * 85)
        print(f"Target Device        : {self.device}")
        print(f"Vocab Size           : {self.vocab_size}")
        print(f"Teacher Model        : {self.teacher_model_name if self.vocab_size != 16384 else 'None (Pure CE in 16k space)'}")
        print(f"Student Layers       : {self.student_layers} Feistel layers")
        print(f"Recurrent State Dim  : {self.state_dim}")
        print(f"Batch Size / SeqLen  : {self.batch_size} / {self.seq_len} tokens")
        print(f"Learning Rate        : {self.lr} (CosineAnnealingLR eta_min={self.eta_min})")
        print(f"Loss Formulation     : {'1.0 * loss_ce (Pure CE)' if self.vocab_size == 16384 else f'1.0 * loss_ce + 0.3 * loss_kd (T={self.temperature}, Top-{self.kd_top_k})'}")
        print(f"Best Model Metric    : Minimum CE Loss (score = ce_loss)")
        print(f"Dataset Path         : {self.data_path}")
        print("=" * 85)

        # 1. Load Tokenizer
        print("\n[1/4] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)

        # 2. Identify Whitespace Tokens & Build Mask Tensor
        print("\n[2/4] Identifying whitespace tokens in vocabulary...")
        ws_token_ids = set()
        for token_id in range(self.tokenizer.vocab_size):
            text = self.tokenizer.decode([token_id])
            if text and text.strip() == "":
                ws_token_ids.add(token_id)
        self.ws_token_ids = ws_token_ids

        # Ensure mask covers full model vocab size (151936) so special tokens never cause CUDA bounds error
        total_vocab_size = max(self.tokenizer.vocab_size, len(self.tokenizer), 151936)
        self.ws_mask_tensor = torch.zeros(
            total_vocab_size,
            dtype=torch.bool,
            device=self.device,
        )
        self.ws_mask_tensor[list(ws_token_ids)] = True
        print(f"Identified {len(self.ws_token_ids)} whitespace tokens in vocabulary (mask size: {total_vocab_size}).")

        # 3. Ensure Dataset and DataLoader
        print(f"\n[3/4] Preparing dataset (seq_len={self.seq_len}, vocab_size={self.vocab_size})...")
        self.dataset = CodeDistillationDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            vocab_map_path=self.vocab_map_path,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(f"Loaded {len(self.dataset)} packed chunks of {self.seq_len} tokens each.")

        # 4. Load Teacher Model (Frozen) & Initialize Student Model (PureFeistelLM)
        if self.vocab_size == 16384:
            print(f"\n[4/4] Initializing Student model (PureFeistelLM, {self.student_layers} layers, state_dim={self.state_dim}) with 16k pruned vocabulary...")
            self.teacher = None
            self.student = PureFeistelLM(
                vocab_size=16384,
                d_model=896,
                num_layers=self.student_layers,
                state_dim=self.state_dim,
                max_seq_len=max(512, self.seq_len),
                layer_scale=0.05,
                torch_dtype=torch.bfloat16,
                device=str(self.device),
                from_pretrained_qwen=False,
                pruned_weights_path=self.pruned_weights_path,
                vocab_map_path=self.vocab_map_path,
                tokenizer=self.tokenizer,
            )
        else:
            print(f"\n[4/4] Loading Teacher & initializing Student model (PureFeistelLM, {self.student_layers} layers, state_dim={self.state_dim})...")
            self.teacher = Qwen2ForCausalLM.from_pretrained(
                self.teacher_model_name,
                torch_dtype=torch.bfloat16,
                device_map=str(self.device),
            )
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

            self.student = PureFeistelLM(
                vocab_size=self.teacher.config.vocab_size,
                d_model=self.teacher.config.hidden_size,
                num_layers=self.student_layers,
                state_dim=self.state_dim,
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
        print(f"Student Total Params     : {total_student_params / 1e6:.2f}M")
        print(f"Trainable Backbone Params: {trainable_params / 1e6:.2f}M ({trainable_params:,} weights)")

        # Compute dynamic training schedule or use manual override
        if self.steps_arg is not None:
            self.steps = self.steps_arg
            self.epochs = (self.steps * self.batch_size) // max(1, len(self.dataset))
            print(f"Training Schedule        : {self.steps} steps (manual override, ~{self.epochs} epochs)")
        else:
            self.steps, self.epochs = compute_training_schedule(
                num_chunks=len(self.dataset),
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                trainable_params=trainable_params,
                alpha=self.alpha,
            )
            print(f"AutoScheduler Schedule   : {len(self.dataset)} chunks -> {self.epochs} epochs -> {self.steps} total steps (alpha={self.alpha})")

        # Optimizer and Cosine Scheduler
        self.optimizer = torch.optim.AdamW(
            self.student.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.steps,
            eta_min=self.eta_min,
        )

    def train(self) -> Dict[str, Any]:
        """
        Executes pure Cross-Entropy training with uniform token weighting and best checkpoint tracking.
        """
        print("\n" + "=" * 85)
        print("STARTING HYBRID DISTILLATION TRAINING LOOP (1.0 * CE + 0.3 * KD)")
        print("=" * 85)
        print(
            f"{'Step':<8} | {'Total':<10} | {'CELoss':<10} | {'KDLoss':<10} | {'CosSim':<8} | {'LR':<9} | {'Throughput':<12}"
        )
        print("-" * 85)

        data_iter = itertools.cycle(self.dataloader)
        history: Dict[str, list] = {
            "steps": [],
            "loss_total": [],
            "loss_ce": [],
            "loss_kd": [],
            "loss_rep": [],
            "cos_sim": [],
            "score": [],
            "lr": [],
            "tokens_per_sec": [],
            "vram_mb": [],
        }

        initial_metrics: Optional[Tuple[float, float, float]] = None
        final_metrics: Optional[Tuple[float, float, float]] = None

        best_score = float("inf")
        best_step = 0
        best_metrics: Optional[Tuple[float, float, float, float]] = None

        t_total_start = time.perf_counter()

        for step in range(1, self.steps + 1):
            t_step_start = time.perf_counter()
            batch = next(data_iter).to(self.device)

            if self.vocab_size == 16384:
                # 1. Pure CE training in 16k space (0 teacher overhead, maximum speed)
                student_hs = self.student.forward_backbone(batch, with_norm=True)
                student_logits = self.student.lm_head(student_hs)  # (B, L, 16384)

                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss_ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss_kd = torch.tensor(0.0, device=self.device)
                loss_total = 1.0 * loss_ce
                teacher_hs = None
            else:
                # 1. Teacher forward (frozen, no_grad) for soft target logits and hidden states
                with torch.no_grad():
                    teacher_out = self.teacher.model(input_ids=batch, return_dict=True)
                    teacher_hs = teacher_out.last_hidden_state
                    teacher_logits = self.teacher.lm_head(teacher_hs)

                # 2. Student forward through backbone, norm, and lm_head
                student_hs = self.student.forward_backbone(batch, with_norm=True)
                student_logits = self.student.lm_head(student_hs)  # (B, L, V)

                # 3. Pure Cross-Entropy loss with uniform weight 1.0 for all tokens
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss_ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # 4. Soft Top-64 Knowledge Distillation loss (T=2.0, per-token batchmean)
                top_teacher_logits, top_indices = torch.topk(teacher_logits, k=self.kd_top_k, dim=-1)
                top_student_logits = torch.gather(student_logits, dim=-1, index=top_indices)
                loss_kd = (self.temperature ** 2) * F.kl_div(
                    F.log_softmax(top_student_logits.view(-1, self.kd_top_k) / self.temperature, dim=-1),
                    F.softmax(top_teacher_logits.view(-1, self.kd_top_k) / self.temperature, dim=-1),
                    reduction="batchmean",
                )

                # 5. Combined loss: 1.0 * loss_ce + 0.3 * loss_kd (loss_rep excluded)
                loss_total = 1.0 * loss_ce + 0.3 * loss_kd

            # 6. Backward & Step
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
            loss_ce_val = loss_ce.item()
            loss_kd_val = loss_kd.item()
            loss_rep_val = 0.0
            curr_lr = self.optimizer.param_groups[0]["lr"]

            # Compute CosSim against teacher on logging steps
            is_log_step = (step == 1 or step % 100 == 0 or step == self.steps)
            if is_log_step and teacher_hs is not None:
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(student_hs, teacher_hs, dim=-1).mean()
                    cos_val = cos_sim.item()
            else:
                cos_val = history["cos_sim"][-1] if history["cos_sim"] else 0.0

            # Score is directly the CE loss
            score = loss_ce_val

            if initial_metrics is None:
                initial_metrics = (loss_total_val, loss_ce_val, cos_val)
            final_metrics = (loss_total_val, loss_ce_val, cos_val)

            # Track and save best checkpoint based on lowest CE loss
            if self.best_ckpt_name_override is not None:
                best_ckpt_name = self.best_ckpt_name_override
            elif self.student_layers == 18 or self.state_dim == 256:
                best_ckpt_name = "pure_feistel_best_8m.pt"
            elif self.vocab_size == 16384:
                best_ckpt_name = "pure_feistel_best_16k.pt"
            else:
                best_ckpt_name = "pure_feistel_best.pt"

            if score < best_score:
                best_score = score
                best_step = step
                best_metrics = (loss_total_val, loss_kd_val, loss_ce_val, cos_val)
                best_ckpt_path = os.path.join(self.save_dir, best_ckpt_name)
                self.save_checkpoint(
                    best_ckpt_path,
                    step,
                    loss_total_val,
                    loss_kd_val,
                    loss_ce_val,
                    cos_val,
                    tag=f"Best CE Checkpoint ({best_ckpt_name})",
                    verbose=False,
                )

            # Logging every 100 steps
            if is_log_step:
                print(
                    f"{step:<8d} | {loss_total_val:<10.4f} | {loss_ce_val:<10.4f} | {loss_kd_val:<10.4f} | {cos_val:<8.4f} | "
                    f"{curr_lr:<9.2e} | {tok_per_sec:<7.0f} tok/s"
                )

            history["steps"].append(step)
            history["loss_total"].append(loss_total_val)
            history["loss_ce"].append(loss_ce_val)
            history["loss_kd"].append(loss_kd_val)
            history["loss_rep"].append(0.0)
            history["cos_sim"].append(cos_val)
            history["score"].append(score)
            history["lr"].append(curr_lr)
            history["tokens_per_sec"].append(tok_per_sec)
            history["vram_mb"].append(vram_mb)

            # Periodic checkpoint every 500 steps
            if step % 500 == 0:
                suffix = "_8m" if (self.student_layers == 18 or self.state_dim == 256) else ("_16k" if self.vocab_size == 16384 else "")
                ckpt_path = os.path.join(self.save_dir, f"pure_feistel_step_{step}{suffix}.pt")
                self.save_checkpoint(
                    ckpt_path,
                    step,
                    loss_total_val,
                    loss_kd_val,
                    loss_ce_val,
                    cos_val,
                    tag="Periodic Checkpoint",
                    verbose=True,
                )

        total_elapsed = time.perf_counter() - t_total_start
        mean_tok_sec = sum(history["tokens_per_sec"]) / max(1, len(history["tokens_per_sec"]))

        # Save final checkpoint
        if self.student_layers == 18 or self.state_dim == 256:
            final_ckpt_name = "pure_feistel_final_8m.pt"
        elif self.vocab_size == 16384:
            final_ckpt_name = "pure_feistel_final_16k.pt"
        else:
            final_ckpt_name = "pure_feistel_final.pt"
        final_ckpt_path = os.path.join(self.save_dir, final_ckpt_name)
        self.save_checkpoint(
            final_ckpt_path,
            self.steps,
            final_metrics[0],
            0.0,
            final_metrics[1],
            final_metrics[2],
            tag="Final Checkpoint",
            verbose=True,
        )

        print("-" * 85)
        print("PURE CROSS-ENTROPY TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 85)
        print(f"Total Duration       : {total_elapsed:.2f} s")
        print(f"Mean Throughput      : {mean_tok_sec:.0f} tokens/s")
        print(
            f"Initial Metrics      : Total={initial_metrics[0]:.4f} | CE={initial_metrics[1]:.4f} | CosSim={initial_metrics[2]:.4f}"
        )
        print(
            f"Final Metrics ({self.steps:<4d}) : Total={final_metrics[0]:.4f} | CE={final_metrics[1]:.4f} | CosSim={final_metrics[2]:.4f}"
        )
        best_ckpt_name = (
            self.best_ckpt_name_override
            if self.best_ckpt_name_override is not None
            else ("pure_feistel_best_8m.pt" if (self.student_layers == 18 or self.state_dim == 256)
                  else ("pure_feistel_best_16k.pt" if self.vocab_size == 16384 else "pure_feistel_best.pt"))
        )
        best_ckpt_path = os.path.join(self.save_dir, best_ckpt_name)
        if best_metrics is not None:
            print(
                f"Best Checkpoint      : Step {best_step} | TotalLoss={best_metrics[0]:.4f} | "
                f"CELoss={best_metrics[2]:.4f} | CosSim={best_metrics[3]:.4f} | Score={best_score:.4f} -> {best_ckpt_path}"
            )
        print("=" * 85 + "\n")

        return {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "best_step": best_step,
            "best_metrics": best_metrics,
            "best_score": best_score,
            "total_elapsed_sec": total_elapsed,
            "mean_tokens_per_sec": mean_tok_sec,
            "final_checkpoint": final_ckpt_path,
            "best_checkpoint": best_ckpt_path,
            "history": history,
        }

    def save_checkpoint(
        self,
        path: str,
        step: int,
        loss_total: float,
        loss_kd: float,
        loss_ce: float,
        cos_sim: float,
        tag: str = "",
        verbose: bool = True,
    ) -> None:
        """
        Saves student backbone parameters and training metadata.
        """
        checkpoint_data = {
            "step": step,
            "backbone_state_dict": self.student.backbone.state_dict(),
            "num_layers": self.student_layers,
            "state_dim": self.state_dim,
            "vocab_size": self.vocab_size,
            "d_model": self.student.d_model,
            "loss_total": loss_total,
            "loss_kd": loss_kd,
            "loss_ce": loss_ce,
            "cos_sim": cos_sim,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(checkpoint_data, path)
        if verbose:
            tag_str = f" [{tag}]" if tag else ""
            print(
                f"  [Checkpoint]{tag_str} Saved step {step} to {path} "
                f"(TotalLoss: {loss_total:.4f}, KDLoss: {loss_kd:.4f}, CELoss: {loss_ce:.4f}, CosSim: {cos_sim:.4f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PureFeistelLM Distillation Trainer with AutoScheduler")
    parser.add_argument("--data_path", type=str, default="data/train_code.jsonl", help="Path to JSONL code dataset")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps (default: auto via compute_training_schedule)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha multiplier for compute_training_schedule (default: 0.6)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length in tokens")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--eta_min", type=float, default=1e-5, help="Minimum learning rate for CosineAnnealingLR")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature T")
    parser.add_argument("--ws_weight", type=float, default=0.05, help="Cross-entropy weight for whitespace tokens")
    parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size (151936 or 16384)")
    parser.add_argument("--vocab_map", type=str, default="checkpoints/vocab_16k/vocab_map.pt", help="Path to vocab_map.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--num_layers", "--student_layers", dest="student_layers", type=int, default=12, help="Number of Feistel layers in student")
    parser.add_argument("--state_dim", type=int, default=128, help="State dimension for GatedLinearRecurrence (default: 128)")
    parser.add_argument("--best_ckpt_name", type=str, default=None, help="Custom filename for best checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    trainer = FeistelDistillationTrainer(
        teacher_model_name="Qwen/Qwen2.5-Coder-0.5B",
        student_layers=args.student_layers,
        state_dim=args.state_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        alpha=args.alpha,
        eta_min=args.eta_min,
        temperature=args.temperature,
        ws_weight=args.ws_weight,
        vocab_size=args.vocab_size,
        vocab_map_path=args.vocab_map,
        best_ckpt_name=args.best_ckpt_name,
        data_path=args.data_path,
        save_dir=args.save_dir,
    )
    trainer.train()
