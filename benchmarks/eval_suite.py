"""
Model Evaluation Suite
======================
Automated evaluation framework for code models (PureFeistelLM, GraftedQwenCoder,
and baseline Transformer models) to assess output quality before and after training.

Evaluation Modules:
  A. AST Validity (test_ast_validity): Python syntax correctness via ast.parse().
  B. Functional Correctness (test_functional_pass): pass@1 execution of 8 standard programming tasks.
  C. VM Protocol Adherence (test_vm_protocol_adherence): Agentic <EXEC_VM> bytecode protocol compliance.
  D. Repetition & Diversity (test_repetition_and_diversity): N-gram diversity and repetition collapse detection.
  E. Full Suite Runner (run_full_evaluation): Orchestrates all modules and saves reports.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from mini_vm.ops import OPCODE_NAMES, NAME_TO_OPCODE


class ModelEvaluationSuite:
    """
    Automated evaluation suite assessing syntactic validity, functional execution,
    agent protocol compliance, and output diversity across neural architectures.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(model, "device"):
            self.device = model.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard prompts for AST validity (12 prompts)
        self.ast_prompts = [
            'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
            'def is_even(n: int) -> bool:\n    """Check if number is even."""\n',
            'def reverse_string(s: str) -> str:\n    """Reverse the given string."""\n',
            'def find_max(numbers: list[int]) -> int:\n    """Return the maximum element."""\n',
            'def count_vowels(text: str) -> int:\n    """Count vowels in the string."""\n',
            'def is_palindrome(word: str) -> bool:\n    """Check if word is a palindrome."""\n',
            'def square_elements(nums: list[int]) -> list[int]:\n    """Return list of squared numbers."""\n',
            'def get_first_item(items: list) -> any:\n    """Return the first item or None."""\n',
            'def compute_average(values: list[float]) -> float:\n    """Compute average of float values."""\n',
            'def filter_positive(nums: list[int]) -> list[int]:\n    """Filter only positive numbers."""\n',
            'def repeat_string(s: str, n: int) -> str:\n    """Repeat string n times."""\n',
            'def is_empty(container: list) -> bool:\n    """Check if list is empty."""\n',
        ]

        # 8 functional coding tasks with rigorous test assertions
        self.functional_tasks = [
            {
                "name": "add",
                "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
                "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n",
            },
            {
                "name": "is_even",
                "prompt": 'def is_even(n: int) -> bool:\n    """Return True if n is even, False otherwise."""\n',
                "test": "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True\n",
            },
            {
                "name": "reverse_str",
                "prompt": 'def reverse_str(s: str) -> str:\n    """Return the reversed string."""\n',
                "test": "assert reverse_str('hello') == 'olleh'\nassert reverse_str('') == ''\n",
            },
            {
                "name": "max_num",
                "prompt": 'def max_num(lst: list) -> int:\n    """Return the maximum number in the list."""\n',
                "test": "assert max_num([1, 5, 3, 9, 2]) == 9\nassert max_num([-5, -1]) == -1\n",
            },
            {
                "name": "count_vowels",
                "prompt": 'def count_vowels(s: str) -> int:\n    """Return the number of vowels in the string."""\n',
                "test": "assert count_vowels('hello world') == 3\nassert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5\n",
            },
            {
                "name": "is_palindrome",
                "prompt": 'def is_palindrome(s: str) -> bool:\n    """Return True if s is a palindrome, False otherwise."""\n',
                "test": "assert is_palindrome('radar') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('a') is True\n",
            },
            {
                "name": "square_list",
                "prompt": 'def square_list(lst: list) -> list:\n    """Return a new list with each element squared."""\n',
                "test": "assert square_list([1, 2, 3]) == [1, 4, 9]\nassert square_list([]) == []\n",
            },
            {
                "name": "fibonacci",
                "prompt": 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n',
                "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8\n",
            },
        ]

        # 5 Agent prompts designed to elicit symbolic VM execution bytecode
        self.vm_protocol_prompts = [
            "Calculate total memory: 1024 + 2048 using VM bytecode:\n<EXEC_VM>",
            "Task: Check if latency 420 exceeds threshold 300 using MiniStackVM bytecode:\n<EXEC_VM>",
            "Task: Compute batch dimension 16 * 64 via deterministic stack operations:\n<EXEC_VM>",
            "Task: Verify whether HTTP status code equals 200 using VM comparison bytecode:\n<EXEC_VM>",
            "Task: Subtract baseline offset 512 from 2048 using stack VM instructions:\n<EXEC_VM>",
        ]

    def generate_text(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Unified generation interface supporting HuggingFace, GraftedQwenCoder,
        and PureFeistelLM models.
        """
        # 1. Custom generate() interface (PureFeistelLM, GraftedQwenCoder)
        if hasattr(self.model, "generate") and not isinstance(self.model, PreTrainedModel):
            text, _ = self.model.generate(prompt, max_new_tokens=max_new_tokens)
            return text

        # 2. HuggingFace PreTrainedModel interface
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def extract_function_body(completion: str) -> str:
        """
        Extracts the target function body from completion, stopping at any
        subsequent top-level function, class, or script boundary (standard practice
        in code generation benchmarks such as HumanEval).
        """
        lines = completion.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if any(line.startswith(prefix) for prefix in ("def ", "class ", "@", "if __name__")):
                    break
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    # =========================================================================
    # A. AST Validity Module
    # =========================================================================
    def test_ast_validity(self, max_tokens: int = 64) -> Dict[str, Any]:
        """
        Generates code for standard function prompts and checks if the combined
        code can be successfully parsed into an Abstract Syntax Tree (AST).
        """
        valid_count = 0
        total_count = len(self.ast_prompts)
        details: List[Dict[str, Any]] = []

        for prompt in self.ast_prompts:
            completion = self.generate_text(prompt, max_new_tokens=max_tokens)
            cleaned_body = self.extract_function_body(completion)
            full_code = prompt + cleaned_body

            # Try parsing full code, prompt + completion, or formatted block
            is_valid = False
            parse_error = ""

            try:
                ast.parse(full_code)
                is_valid = True
            except SyntaxError as e:
                # Also test if raw prompt + completion or completion itself is a valid Python statement or block
                try:
                    ast.parse(prompt + completion)
                    is_valid = True
                except SyntaxError:
                    try:
                        ast.parse(completion)
                        is_valid = True
                    except SyntaxError:
                        parse_error = f"{type(e).__name__}: {str(e)}"

            if is_valid:
                valid_count += 1

            details.append({
                "prompt": prompt.strip().split("\n")[0],
                "valid": is_valid,
                "error": parse_error,
                "completion_snippet": completion.strip()[:60],
            })

        ast_pass_pct = (valid_count / total_count) * 100.0 if total_count > 0 else 0.0
        return {
            "valid_count": valid_count,
            "total_count": total_count,
            "ast_pass_pct": round(ast_pass_pct, 1),
            "details": details,
        }

    # =========================================================================
    # B. Functional Correctness (pass@1) Module
    # =========================================================================
    def test_functional_pass(
        self,
        max_tokens: int = 96,
        timeout_sec: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Evaluates pass@1 accuracy by executing the generated functions against
        deterministic unit tests in an isolated Python interpreter subprocess.
        """
        passed_tasks = 0
        total_tasks = len(self.functional_tasks)
        task_results: List[Dict[str, Any]] = []

        for task in self.functional_tasks:
            prompt = task["prompt"]
            test_assertions = task["test"]
            completion = self.generate_text(prompt, max_new_tokens=max_tokens)
            cleaned_body = self.extract_function_body(completion)

            # Assemble test script with clean function body
            script_code = prompt + cleaned_body + "\n\n" + test_assertions

            # Run in isolated subprocess with strict timeout
            status = "FAILED"
            error_msg = ""
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script_code],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                if proc.returncode == 0:
                    status = "PASSED"
                    passed_tasks += 1
                else:
                    error_msg = proc.stderr.strip()[:120] if proc.stderr else f"Exit code {proc.returncode}"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                error_msg = f"Execution exceeded {timeout_sec}s timeout limit"
            except Exception as ex:
                status = "ERROR"
                error_msg = f"{type(ex).__name__}: {str(ex)}"

            task_results.append({
                "task": task["name"],
                "status": status,
                "error": error_msg,
                "completion_snippet": completion.strip().replace("\n", " ")[:60],
            })

        pass_at_1_pct = (passed_tasks / total_tasks) * 100.0 if total_tasks > 0 else 0.0
        return {
            "passed_tasks": passed_tasks,
            "total_tasks": total_tasks,
            "pass_at_1_pct": round(pass_at_1_pct, 1),
            "task_results": task_results,
        }

    # =========================================================================
    # C. VM Protocol Adherence Module
    # =========================================================================
    def test_vm_protocol_adherence(self, max_tokens: int = 64) -> Dict[str, Any]:
        """
        Checks whether the model obeys the agentic <EXEC_VM> ... </EXEC_VM> protocol
        and outputs valid MiniStackVM opcodes.
        """
        valid_opcodes = set(OPCODE_NAMES.values()) | {"PUSH", "HALT", "POP"}
        adherent_count = 0
        total_prompts = len(self.vm_protocol_prompts)
        details: List[Dict[str, Any]] = []

        for prompt in self.vm_protocol_prompts:
            output = self.generate_text(prompt, max_new_tokens=max_tokens)
            full_response = prompt + output

            # 1. Regex check for <EXEC_VM> tags
            # Supports unclosed or closed tag matching
            vm_block_match = re.search(r"<EXEC_VM>([\s\S]*?)(?:</EXEC_VM>|$)", full_response, re.IGNORECASE)
            has_tag = vm_block_match is not None

            # 2. Check for presence of valid MiniStackVM opcodes
            found_opcodes: List[str] = []
            if vm_block_match:
                block_content = vm_block_match.group(1).upper()
                for op in valid_opcodes:
                    if re.search(r"\b" + op + r"\b", block_content):
                        found_opcodes.append(op)

            is_adherent = has_tag and len(found_opcodes) > 0
            if is_adherent:
                adherent_count += 1

            details.append({
                "prompt_snippet": prompt.strip().split("\n")[0][:45],
                "has_exec_vm_tag": has_tag,
                "found_opcodes": found_opcodes,
                "is_adherent": is_adherent,
            })

        adherence_pct = (adherent_count / total_prompts) * 100.0 if total_prompts > 0 else 0.0
        return {
            "adherent_count": adherent_count,
            "total_prompts": total_prompts,
            "protocol_adherence_pct": round(adherence_pct, 1),
            "details": details,
        }

    # =========================================================================
    # D. Repetition & Diversity Module
    # =========================================================================
    def test_repetition_and_diversity(
        self,
        prompts: Optional[List[str]] = None,
        max_tokens: int = 64,
    ) -> Dict[str, Any]:
        """
        Measures unique 2-gram and 3-gram ratios across generations and detects
        pathological repetition loop collapses.
        """
        test_prompts = prompts or self.ast_prompts[:5]
        per_prompt_stats: List[Dict[str, Any]] = []
        any_collapse = False

        for prompt in test_prompts:
            text = self.generate_text(prompt, max_new_tokens=max_tokens)
            words = re.findall(r"\w+|[^\w\s]", text)

            if len(words) < 2:
                # Trivial or empty output
                div_2g = 1.0
                div_3g = 1.0
                collapsed = False
            else:
                two_grams = list(zip(words[:-1], words[1:]))
                div_2g = len(set(two_grams)) / max(1, len(two_grams))

                if len(words) >= 3:
                    three_grams = list(zip(words[:-2], words[1:-1], words[2:]))
                    div_3g = len(set(three_grams)) / max(1, len(three_grams))
                else:
                    div_3g = 1.0

                # Collapse detected if diversity is extremely low or single token repeated >= 5 times
                collapsed = (div_2g < 0.20) or any(
                    text.count(token * 5) > 0 for token in ["False", "None", ".", "()", " return"]
                )

            if collapsed:
                any_collapse = True

            per_prompt_stats.append({
                "prompt": prompt.strip().split("\n")[0][:40],
                "div_2gram": round(div_2g, 3),
                "div_3gram": round(div_3g, 3),
                "collapsed": collapsed,
                "snippet": text.strip().replace("\n", " ")[:40],
            })

        mean_2g = sum(p["div_2gram"] for p in per_prompt_stats) / max(1, len(per_prompt_stats))
        mean_3g = sum(p["div_3gram"] for p in per_prompt_stats) / max(1, len(per_prompt_stats))

        return {
            "mean_2gram_diversity": round(mean_2g, 3),
            "mean_3gram_diversity": round(mean_3g, 3),
            "repetition_collapse_detected": any_collapse,
            "per_prompt_stats": per_prompt_stats,
        }

    # =========================================================================
    # E. Full Evaluation Suite Runner
    # =========================================================================
    def run_full_evaluation(
        self,
        report_name: str = "eval_report",
        save_to_file: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes all evaluation modules, prints a structured summary table,
        and saves the complete report as JSON in docs/.
        """
        print("\n" + "=" * 75)
        print(f"STARTING COMPREHENSIVE MODEL EVALUATION SUITE: {report_name}")
        print("=" * 75)

        t0 = time.perf_counter()

        print("[1/4] Running AST Syntax Validity Evaluation...")
        ast_res = self.test_ast_validity()

        print("[2/4] Running Functional Correctness (pass@1) Unit Tests...")
        func_res = self.test_functional_pass()

        print("[3/4] Running Agentic VM Protocol Adherence Evaluation...")
        vm_res = self.test_vm_protocol_adherence()

        print("[4/4] Running N-gram Diversity & Repetition Collapse Check...")
        div_res = self.test_repetition_and_diversity()

        total_duration = time.perf_counter() - t0

        summary = {
            "report_name": report_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(total_duration, 2),
            "ast_pass_pct": ast_res["ast_pass_pct"],
            "functional_pass_at_1_pct": func_res["pass_at_1_pct"],
            "protocol_adherence_pct": vm_res["protocol_adherence_pct"],
            "mean_2gram_diversity": div_res["mean_2gram_diversity"],
            "repetition_collapse_detected": div_res["repetition_collapse_detected"],
            "details": {
                "ast": ast_res,
                "functional": func_res,
                "vm_protocol": vm_res,
                "diversity": div_res,
            },
        }

        # Print structured summary table
        print("\n" + "=" * 75)
        print(f"EVALUATION SUITE RESULTS: {report_name.upper()}")
        print("=" * 75)
        print(f"{'Evaluation Metric':<40} | {'Score / Result':<28}")
        print("-" * 75)
        print(f"{'AST Syntax Pass Rate':<40} | {ast_res['ast_pass_pct']}% ({ast_res['valid_count']}/{ast_res['total_count']})")
        print(f"{'Functional pass@1 Unit Test Rate':<40} | {func_res['pass_at_1_pct']}% ({func_res['passed_tasks']}/{func_res['total_tasks']})")
        print(f"{'Agent VM Protocol Adherence':<40} | {vm_res['protocol_adherence_pct']}% ({vm_res['adherent_count']}/{vm_res['total_prompts']})")
        print(f"{'Mean 2-gram Diversity':<40} | {div_res['mean_2gram_diversity']:.3f}")
        print(f"{'Mean 3-gram Diversity':<40} | {div_res['mean_3gram_diversity']:.3f}")
        print(f"{'Repetition Collapse Detected':<40} | {'YES (Failed)' if div_res['repetition_collapse_detected'] else 'NO (Passed)'}")
        print(f"{'Total Evaluation Latency':<40} | {total_duration:.2f} s")
        print("=" * 75 + "\n")

        # Save to file
        if save_to_file:
            os.makedirs("docs", exist_ok=True)
            report_path = os.path.join("docs", f"eval_{report_name}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Report successfully saved to: {report_path}\n")

        return summary
