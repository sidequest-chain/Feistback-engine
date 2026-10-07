"""
Autonomous Feistel Agent Loop
=============================
Integrates proven architectural components (EXP-001 through EXP-007) into a unified
real-time perception-deliberation-action agent loop:
  1. Perceive: Encodes observation tokens into hidden states without KV-cache expansion.
  2. Deliberate: AdaptiveFeistelRunner (EXP-003) unrolls dynamic depth with early exit.
  3. Act & Overlap: AsyncHybridPipeline (EXP-007) overlaps GPU state prep with CPU MiniStackVM.
  4. State Update: Injects deterministic symbolic VM results directly into latent registers.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from mini_vm.ops import OpCode
from mini_vm.vm import MiniStackVM
from runner.adaptive_loop import AdaptiveFeistelRunner
from runner.async_pipeline import AsyncHybridPipeline


class AutonomousFeistelAgent:
    """
    Unified Autonomous Agent combining Reversible Feistel neural reasoner,
    adaptive deliberation depth, deterministic MiniStackVM, and asynchronous execution.
    """

    def __init__(
        self,
        d_model: int = 128,
        seq_len: int = 64,
        max_deliberation_steps: int = 8,
        epsilon: float = 0.01,
        device: Optional[str] = None,
        vocab_size: int = 124,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.d_model = d_model
        self.seq_len = seq_len
        self.max_deliberation_steps = max_deliberation_steps
        self.epsilon = epsilon

        # 1. Perception Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model).to(self.device)

        # 2. Latent Deliberation Runner (EXP-003)
        self.adaptive_runner = AdaptiveFeistelRunner(
            d_model=d_model,
            max_seq_len=seq_len,
            default_max_steps=max_deliberation_steps,
            default_epsilon=epsilon,
        ).to(self.device)
        self.adaptive_runner.eval()

        # 3. Deterministic MiniStackVM (EXP-004)
        self.vm = MiniStackVM(max_steps=5000)

        # 4. Asynchronous Pipeline Orchestrator (EXP-007)
        self.pipeline = AsyncHybridPipeline(
            runner=self.adaptive_runner,
            vm=self.vm,
            device=str(self.device),
        )

        # 5. Internal working registers
        self.registers: Dict[str, Any] = {}

    def perceive(
        self, observation: Union[torch.Tensor, List[int], str, None]
    ) -> torch.Tensor:
        """
        Encodes observation input into a continuous latent state tensor.
        """
        if observation is None:
            return torch.randn(1, self.seq_len, self.d_model, device=self.device)

        if isinstance(observation, torch.Tensor):
            if observation.ndim == 2:  # Token IDs (B, L)
                return self.token_embed(observation.to(self.device))
            elif observation.ndim == 3:  # Latent state (B, L, D)
                return observation.to(self.device)
            else:
                raise ValueError(f"Unsupported observation tensor shape: {observation.shape}")

        if isinstance(observation, list):
            input_ids = torch.tensor([observation], dtype=torch.long, device=self.device)
            return self.token_embed(input_ids)

        # Default random latent state
        return torch.randn(1, self.seq_len, self.d_model, device=self.device)

    def deliberate(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, int, List[float]]:
        """
        Executes dynamic deliberation with early-exit convergence check (delta < epsilon).
        """
        with torch.no_grad():
            return self.adaptive_runner.forward_adaptive(
                z, max_steps=self.max_deliberation_steps, epsilon=self.epsilon
            )

    def act_and_overlap(
        self,
        current_z: torch.Tensor,
        bytecode: List[Any],
        args: Optional[List[Any]] = None,
    ) -> Tuple[torch.Tensor, Any, float]:
        """
        Executes CPU VM bytecode in background thread while GPU concurrently
        prepares the next latent state in a dedicated CUDA stream.
        Returns:
            next_z: Updated latent state tensor from GPU.
            vm_res: Output value from MiniStackVM.
            vm_time_us: Execution latency of the VM task in microseconds.
        """
        vm_duration_box: List[float] = [0.0]

        def timed_vm_task():
            t0 = time.perf_counter()
            res = self.vm.execute(bytecode, args=args)
            vm_duration_box[0] = (time.perf_counter() - t0) * 1_000_000.0
            return res

        # Run overlapped cycle: CPU VM task runs concurrently with GPU deliberation
        next_z, vm_res = self.pipeline.run_async_cycle(
            current_z, timed_vm_task, steps=1
        )
        return next_z, vm_res, vm_duration_box[0]

    def update_state(
        self, z: torch.Tensor, vm_res: Any
    ) -> torch.Tensor:
        """
        Injects deterministic symbolic outputs directly into the latent state vector
        and internal agent registers without verbose text decoding.
        """
        self.registers["last_result"] = vm_res

        # If scalar numeric result, inject directly into first token embedding
        if isinstance(vm_res, (int, float)):
            with torch.no_grad():
                z_updated = z.clone()
                z_updated[:, 0, :1] += float(vm_res) * 0.001
                return z_updated

        return z

    def run_agent_task(
        self,
        initial_tokens: Union[torch.Tensor, List[int], str, None] = None,
        task_bytecode: Optional[List[Any]] = None,
        num_agent_steps: int = 10,
        bytecode_fn: Optional[Callable[[int], List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a complete multi-step autonomous agent reasoning loop.
        """
        start_time = time.perf_counter()
        history: List[Dict[str, Any]] = []

        # 1. Perception phase
        current_z = self.perceive(initial_tokens)

        for step in range(1, num_agent_steps + 1):
            step_start = time.perf_counter()

            # 2. Deliberation phase (Poincaré attractor early-exit)
            deliberated_z, delib_steps, deltas = self.deliberate(current_z)

            # Resolve bytecode and operation descriptor for this step
            op_desc = "EXEC_VM"
            if bytecode_fn is not None:
                resolved = bytecode_fn(step)
                if isinstance(resolved, tuple) and len(resolved) == 2:
                    current_bytecode, op_desc = resolved
                else:
                    current_bytecode = resolved
            elif task_bytecode is not None:
                current_bytecode = task_bytecode
            else:
                current_bytecode = [OpCode.PUSH_CONST, step, OpCode.RET]

            # 3. Action and Overlapped execution phase
            next_z, vm_res, vm_time_us = self.act_and_overlap(
                deliberated_z, current_bytecode
            )

            # 4. State Update phase (Direct latent memory injection)
            current_z = self.update_state(next_z, vm_res)

            step_time_ms = (time.perf_counter() - step_start) * 1000.0

            history.append({
                "step": step,
                "deliberation_steps": delib_steps,
                "vm_op": op_desc,
                "vm_result": vm_res,
                "step_time_ms": step_time_ms,
                "vm_time_us": vm_time_us,
                "status": "CONVERGED" if delib_steps < self.max_deliberation_steps else "MAX_STEPS",
            })

        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        mean_delib_steps = sum(h["deliberation_steps"] for h in history) / len(history)

        return {
            "step_history": history,
            "total_time_ms": total_time_ms,
            "mean_deliberation_steps": mean_delib_steps,
            "final_state": current_z,
            "registers": dict(self.registers),
        }

    def close(self) -> None:
        """Releases pipeline resources."""
        self.pipeline.close()

    def __enter__(self) -> AutonomousFeistelAgent:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
