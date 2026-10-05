"""
Heterogeneous Asynchronous CPU-VM and GPU Deliberation Pipeline.
Implements the 'two chefs' architecture (Strata principle) where GPU neural latent
deliberation and CPU MiniStackVM symbolic execution run concurrently without sequential stalling.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from mini_vm.vm import MiniStackVM
from runner.adaptive_loop import AdaptiveFeistelRunner


class AsyncHybridPipeline:
    """
    Orchestrates heterogeneous execution between GPU Feistel neural reasoner
    and CPU deterministic MiniStackVM.

    - GPU deliberation runs in a dedicated CUDA stream (torch.cuda.Stream()).
    - CPU VM operations execute in a background ThreadPoolExecutor.
    - Synchronizes output with minimal stall time right before next token projection.
    """

    def __init__(
        self,
        runner: Optional[AdaptiveFeistelRunner] = None,
        vm: Optional[MiniStackVM] = None,
        device: Optional[str] = None,
        max_workers: int = 1,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if runner is not None:
            self.runner = runner.to(self.device)
        else:
            self.runner = AdaptiveFeistelRunner(d_model=128, max_seq_len=128).to(self.device)
        self.runner.eval()

        self.vm = vm if vm is not None else MiniStackVM(max_steps=5000)

        # Dedicated CUDA stream for non-blocking GPU execution
        self.cuda_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

        # Background thread pool for non-blocking CPU execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def submit_vm_task(
        self, task_fn: Callable[[], Any]
    ) -> concurrent.futures.Future:
        """
        Submits an arbitrary CPU VM task or bytecode sequence to the thread pool.
        """
        return self.executor.submit(task_fn)

    def execute_gpu_deliberation(
        self, z: torch.Tensor, steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Executes neural latent deliberation in the dedicated CUDA stream.
        """
        if self.cuda_stream is not None:
            with torch.cuda.stream(self.cuda_stream):
                out = self.runner.forward_static(z, steps=steps)
            return out
        return self.runner.forward_static(z, steps=steps)

    def run_sequential_cycle(
        self,
        z: torch.Tensor,
        vm_task_fn: Callable[[], Any],
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Sequential execution: GPU deliberates, waits for completion,
        then CPU VM executes, stalling the pipeline.
        """
        # 1. GPU deliberation
        z_out = self.runner.forward_static(z, steps=steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 2. CPU VM execution
        vm_res = vm_task_fn()
        return z_out, vm_res

    def run_async_cycle(
        self,
        z: torch.Tensor,
        vm_task_fn: Callable[[], Any],
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Async overlapped execution: CPU VM task runs concurrently with GPU deliberation.
        """
        # 1. Dispatch CPU task to background thread pool
        cpu_future = self.submit_vm_task(vm_task_fn)

        # 2. Concurrently execute GPU deliberation in dedicated CUDA stream
        if self.cuda_stream is not None:
            with torch.cuda.stream(self.cuda_stream):
                z_out = self.runner.forward_static(z, steps=steps)
            self.cuda_stream.synchronize()
        else:
            z_out = self.runner.forward_static(z, steps=steps)

        # 3. Collect CPU VM result
        vm_res = cpu_future.result()
        return z_out, vm_res

    def close(self) -> None:
        """Shuts down the background thread pool executor."""
        self.executor.shutdown(wait=True)

    def __enter__(self) -> AsyncHybridPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
