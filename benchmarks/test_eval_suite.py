"""
Smoke Test: Model Evaluation Suite Baseline
===========================================
Runs the ModelEvaluationSuite on the teacher ground-truth model
(Qwen/Qwen2.5-Coder-0.5B full 24-layer Transformer) to establish the
gold-standard performance baseline before training PureFeistelLM.
"""

from __future__ import annotations

import sys
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from benchmarks.eval_suite import ModelEvaluationSuite


def run_eval_suite_smoke_test(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
    report_name: str = "qwen_base_ground_truth",
) -> None:
    """
    Executes full evaluation on base teacher model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Ground Truth Model '{model_name}' on device '{device}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    print("Model loaded successfully. Initializing ModelEvaluationSuite...")
    suite = ModelEvaluationSuite(model=model, tokenizer=tokenizer, device=device)

    # Run full evaluation
    summary = suite.run_full_evaluation(report_name=report_name, save_to_file=True)
    print("Smoke test evaluation completed successfully.")
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_eval_suite_smoke_test()
