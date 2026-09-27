"""
Hybrid Pipeline: Neural Reasoner + Deterministic VM Orchestrator
==============================================================
Manages end-to-end execution:
  1. Extracts numeric arguments from natural language prompts.
  2. Triggers neural model for VM bytecode emission up to </EXEC_VM>.
  3. Executes bytecode deterministically in MiniStackVM using extracted registers.
  4. Injects computed result into final response.
"""

import time
from typing import List, Dict, Any, Optional
import torch

from runner.tokenizer import HybridTokenizer
from runner.args_parser import extract_numeric_args
from mini_vm.vm import MiniStackVM
from core.model import FeistelHolographicLM


class HybridPipeline:
    """
    High-level orchestrator connecting Feistel neural model and MiniStackVM.
    """
    def __init__(
        self,
        model: FeistelHolographicLM,
        tokenizer: HybridTokenizer,
        vm: Optional[MiniStackVM] = None,
        device: str = "cuda",
        max_seq_len: int = 112
    ):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.vm = vm if vm is not None else MiniStackVM()
        self.max_seq_len = max_seq_len
        self.model.eval()

        self.pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]
        self.bos_id = self.tokenizer.vocab[self.tokenizer.bos_token]
        self.exec_vm_id = self.tokenizer.vocab[self.tokenizer.exec_vm_token]
        self.end_vm_id = self.tokenizer.vocab[self.tokenizer.end_vm_token]

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        """
        Generates full answer string with deterministic VM calculation.
        """
        res = self.run_task(prompt, max_tokens=max_tokens)
        return res["full_output"]

    def run_task(self, prompt: str, max_tokens: int = 64) -> Dict[str, Any]:
        """
        Detailed execution returning bytecode, extracted args, and VM result.
        """
        start_t = time.perf_counter()

        # 1. Dynamically extract integer arguments from prompt
        args = extract_numeric_args(prompt)

        # 2. Prepare prompt prefix with <EXEC_VM> cue
        prompt_trimmed = prompt.rstrip()
        if not prompt_trimmed.endswith(self.tokenizer.exec_vm_token):
            hybrid_prompt = f"{prompt_trimmed} {self.tokenizer.exec_vm_token} "
        else:
            hybrid_prompt = f"{prompt_trimmed} "

        input_ids = self.tokenizer.encode(hybrid_prompt, add_special=True)[:-1]
        current_ids = list(input_ids)
        gen_ids: List[int] = []

        # 3. Autoregressive generation until </EXEC_VM>
        for _ in range(max_tokens):
            curr_len = len(current_ids)
            if curr_len >= self.max_seq_len:
                break

            padded_seq = current_ids + [self.pad_id] * (self.max_seq_len - curr_len)
            input_tensor = torch.tensor([padded_seq], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits, _ = self.model(input_tensor)

            next_token_logits = logits[0, curr_len - 1, :].clone()
            next_token_logits[self.pad_id] = -float("inf")
            next_token_logits[self.bos_id] = -float("inf")
            next_id = torch.argmax(next_token_logits).item()

            gen_ids.append(next_id)
            current_ids.append(next_id)

            if next_id == self.end_vm_id:
                break

        gen_str = self.tokenizer.decode(gen_ids)

        # 4. Parse emitted VM bytecode
        vm_tokens = self.tokenizer.tokenize(gen_str)
        bytecode = self.tokenizer.tokens_to_bytecode(vm_tokens)

        # 5. Deterministic VM execution
        vm_start = time.perf_counter()
        vm_res = None
        vm_err = None
        try:
            vm_res = self.vm.execute(bytecode, args=args)
        except Exception as e:
            vm_err = str(e)
        vm_time_ms = (time.perf_counter() - vm_start) * 1000.0

        total_time_ms = (time.perf_counter() - start_t) * 1000.0

        # 6. Synthesize output response
        clean_bytecode_str = gen_str.strip()
        full_output = (
            f"{prompt_trimmed} {self.tokenizer.exec_vm_token} {clean_bytecode_str} "
            f"{self.tokenizer.res_token} Tulemuseks saame {vm_res}."
        )

        return {
            "prompt": prompt,
            "args": args,
            "bytecode": bytecode,
            "emitted_str": clean_bytecode_str,
            "vm_result": vm_res,
            "vm_error": vm_err,
            "full_output": full_output,
            "total_time_ms": total_time_ms,
            "vm_time_ms": vm_time_ms,
            "steps": len(gen_ids)
        }
