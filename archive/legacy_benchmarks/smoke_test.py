"""
Smoke Test: Pure Modular Verification for Feistback Engine
==========================================================
Verifies standalone decoupled integration:
  1. Imports core, mini_vm, and runner without errors.
  2. Checks for checkpoint in checkpoints/feistel_dynamic_hybrid.pt.
  3. Initializes HybridPipeline.
  4. Solves unseen calculation: 'Küsimus: liida arvud 1500 ja 2500.'
  5. Asserts deterministic VM result equals 4000.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("=" * 80)
print("  FEISTBACK ENGINE INTEGRATION SMOKE TEST")
print("=" * 80)

# 1. Clean Imports
print("[1/4] Verifying clean imports...")
try:
    import torch
except ImportError:
    print("❌ PyTorch is not installed in the current Python environment.")
    print("Please install PyTorch using: pip install torch")
    sys.exit(1)

from core import FeistelHolographicLM
from mini_vm import MiniStackVM, OpCode
from runner import HybridTokenizer, extract_numeric_args, HybridPipeline

print("  ✓ core, mini_vm, runner imported successfully.")

# 2. Checkpoint Loading
ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "feistel_dynamic_hybrid.pt")
print(f"[2/4] Checking checkpoint at: {ckpt_path}...")
if not os.path.exists(ckpt_path):
    print(f"⚠️  Checkpoint not found at: {ckpt_path}")
    print("    Train a checkpoint first using: python -m trainer.train_hybrid")
    print("    Smoke test import verification passed successfully.")
    print("=" * 80)
    sys.exit(0)

def select_device() -> str:
    """Selects best available device, testing CUDA kernel execution to avoid runtime crashes."""
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            del t
            return "cuda"
        except Exception:
            print("  ℹ️ CUDA available but GPU kernels not supported by installed PyTorch wheel. Falling back to CPU.")
            return "cpu"
    return "cpu"

device = select_device()
checkpoint = torch.load(ckpt_path, map_location=device)

tokenizer = HybridTokenizer(num_registers=4)
vm = MiniStackVM()
max_seq_len = checkpoint.get("max_seq_len", 112)

model = FeistelHolographicLM(
    vocab_size=tokenizer.vocab_size,
    d_model=checkpoint.get("d_model", 128),
    num_layers=checkpoint.get("num_layers", 16),
    max_seq_len=max_seq_len,
    activation="silu",
    layer_scale=1.0
).to(device)
model.load_state_dict(checkpoint["model_state"])
print(f"  ✓ Model loaded on {device.upper()} (16 layers, d_model=128).")

# 3. Hybrid Pipeline Initialization
print("[3/4] Initializing HybridPipeline...")
pipeline = HybridPipeline(
    model=model,
    tokenizer=tokenizer,
    vm=vm,
    device=device,
    max_seq_len=max_seq_len
)
print("  ✓ HybridPipeline ready.")

# 4. End-to-End Task Verification
test_prompt = "Küsimus: liida arvud 1500 ja 2500."
print(f"[4/4] Executing test prompt: '{test_prompt}'...")
result = pipeline.run_task(test_prompt)

print("\n--- Pipeline Execution Details ---")
print(f"Extracted Registers: {result['args']}")
print(f"Emitted Bytecode:    {result['emitted_str']}")
print(f"Raw OpCodes:         {result['bytecode']}")
print(f"VM Computed Result:  {result['vm_result']}")
print(f"Total Latency:       {result['total_time_ms']:.2f} ms (VM execution: {result['vm_time_ms']*1000.0:.2f} µs)")
print(f"Full Generated Text: {result['full_output']}")
print("----------------------------------\n")

# Assert mathematical accuracy
expected = 4000
assert result["vm_result"] == expected, (
    f"Smoke test failed! Expected {expected}, got {result['vm_result']}"
)

print(f"✅ SMOKE TEST PASSED: 1500 + 2500 = {result['vm_result']} (100% exact math)")
print("=" * 80)
