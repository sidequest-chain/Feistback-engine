# FeistBack Engine

> **A resource-light, accessible neuro-symbolic framework designed to democratize language model development for independent developers, home labs, and small businesses through O(1) memory reversibility, holographic FFT-mixing, and deterministic stack VM execution.**

---

## Vision: Democratizing AI for Local Hardware

State-of-the-art AI development often seems out of reach for small teams, researchers, and home-lab builders due to extreme compute and memory requirements. Standard Transformer architectures present two major hurdles for local deployment:

1. **Quadratic Complexity & Memory Growth:** Standard attention mechanisms and large KV-caches demand substantial VRAM, making training and long-context serving difficult on consumer-grade hardware.
2. **Probabilistic Nature of Exact Tasks:** Relying entirely on probabilistic neural weights to perform strict arithmetic and logical reasoning requires massive parameter counts and remains prone to calculation errors.

**FeistBack Engine** explores a practical, highly efficient alternative:
- **Neural Frontend (`feistel_core`):** A compact syntactic parser and bytecode compiler designed to run efficiently on standard consumer GPUs or CPUs.
- **Symbolic Backend (`stack_vm`):** A lightweight, isolated stack virtual machine that handles logical and mathematical tasks with 100% deterministic accuracy in microseconds.
- **Reversible Dynamics:** Reconstructing forward activations dynamically during backward passes to reduce training memory by **~90%**, keeping resource usage predictable and light.

Our goal is to put capable, reliable, and trainable models directly into the hands of backyard experimenters, boutique development shops, and self-hosted environments.

---

## Architectural Pillars

### 1. O(1) Reversible Feistel Blocks
Instead of caching every forward activation across all layers, FeistBack couples layer transformations using discrete Feistel equations:

```text
x1_(l+1) = x1_(l) + F(x2_(l))
x2_(l+1) = x2_(l) + G(x1_(l+1))
```

During backpropagation, earlier layer states are reconstructed on the fly from later states. This keeps training memory steady and modest regardless of layer depth, making model training accessible on standard desktop hardware.

### 2. Holographic FFT-Mixing
Rather than computing quadratic pairwise attention matrices (Q · K^T), FeistBack applies causal frequency-domain convolutions via Fast Fourier Transforms (FFT):

```text
Mix(X) = IFFT( FFT(X) ⊙ FFT(W) )
```

This brings sequence mixing to O(N log N) complexity, eliminating the need for specialized high-bandwidth memory setups to handle long token interactions.

### 3. Isolated Stack Virtual Machine (`MiniStackVM`)
Logical instructions are emitted as a clean bytecode stream (`<EXEC_VM> ... </EXEC_VM>`). An independent, zero-dependency stack-based virtual machine intercepts computation tokens, reads values from argument registers (`<ARG:0>`..`<ARG:3>`), and returns exact results in under 10 µs.

---

## Empirical Benchmarks & Telemetry

Evaluated against a standard Multi-Head Attention Transformer baseline on 4-digit arithmetic reasoning tasks (1000 ... 9999):

| Metric | Standard Transformer | FeistBack Engine | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | 197,380 | **115,836** | **-41.3%** |
| **Active Vocabulary** | 441 tokens | **124 tokens** | **-71.8%** |
| **Training VRAM** | 2,045 MB | **212 MB** | **-89.6%** |
| **Inference VRAM** | ~180 MB | **~71 MB** | **-60.5%** |
| **Execution Latency** | ~100 ms (probabilistic) | **6.72 µs (VM execution)** | **Deterministic speedup** |
| **4-Digit Generalization**| 12.0% | **100.0% (50/50 Pass)** | **Absolute Precision** |

---

## Repository Structure

```text
feistback-engine/
├── feistel_core/             # Pure PyTorch neural core (Zero VM logic)
│   ├── model.py              # FeistelHolographicLM (d_model=128, 16 layers)
│   ├── reversible_feistel.py # Autograd backward dynamic reconstruction
│   └── holographic_mixer.py  # FFT-based causal frequency mixer
│
├── stack_vm/                 # Pure Python deterministic VM (Zero PyTorch dependencies)
│   ├── ops.py                # Opcode constants (LOAD_ARG, ADD, SUB, MUL, RET)
│   └── vm.py                 # MiniStackVM engine and isolated stack execution
│
├── runtime/                  # Orchestration and pipeline bridge
│   ├── tokenizer.py          # 124-token vocabulary with <ARG:x> support
│   ├── args_parser.py        # Numerical argument parser to memory registers
│   └── pipeline.py           # HybridPipeline (model stream, interrupt, VM injection)
│
├── training/                 # Dataset generation and reproducible training
│   ├── dynamic_dataset.py    # Runtime sample synthesis with jitter augmentation
│   └── train_hybrid.py       # Modular training loop with checkpointing
│
├── evaluation/               # Verification suites
│   └── verify_generalization.py # 50-task out-of-distribution 4-digit test suite
│
├── checkpoints/              # Checkpoints (git-ignored for clean development)
│   └── feistel_dynamic_hybrid.pt
│
└── smoke_test.py             # System verification and end-to-end sanity check
```

---

## Quickstart

### 1. Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended, CPU fully supported)

### 2. Clone and Setup Environment
```bash
git clone https://github.com/<your-username>/feistback-engine.git
cd feistback-engine
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install torch
```

### 3. Run the Smoke Test
Verify that the neural frontend, tokenizer, and stack VM interact seamlessly:
```bash
python smoke_test.py
```

### 4. Run Generalization Verification
Evaluate model generalization across 50 unseen 4-digit numeric tasks:
```bash
python evaluation/verify_generalization.py
```

---

## Usage Example

```python
import torch
from feistel_core import FeistelHolographicLM
from runtime import HybridTokenizer, HybridPipeline
from stack_vm import MiniStackVM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize decoupled components
tokenizer = HybridTokenizer()
model = FeistelHolographicLM(
    vocab_size=len(tokenizer),
    d_model=128,
    n_layers=16
).to(device)

# 2. Load weights
checkpoint = torch.load("checkpoints/feistel_dynamic_hybrid.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Spin up hybrid pipeline
vm = MiniStackVM()
pipeline = HybridPipeline(model, tokenizer, vm, device=device)

# 4. Deterministic inference
prompt = "Küsimus: liida arvud 1842 ja 2359."
response = pipeline.generate(prompt)
print(response)
# Output: Küsimus: liida arvud 1842 ja 2359. <EXEC_VM> ... </EXEC_VM> <RES> Tulemuseks saame 4201.
```

---

## Decentralized Distributed Training (Roadmap)

To keep development accessible without requiring expensive enterprise networking (such as NVLink or InfiniBand), future milestones focus on consumer-grade distributed scaling:
- **DiLoCo (Distributed Low-Communication):** Slices training across independent nodes over ordinary broadband connections, synchronizing outer weights every 500 steps.
- **Local Algorithmic Generation:** Nodes synthesize task variations locally, avoiding high-bandwidth dataset transfers.
- **Quantized Gradient Sharing:** Exchanging compressed INT4 updates to keep network synchronization light and practical.

---

## License

FeistBack Engine is licensed under the **European Union Public Licence (EUPL-1.2)**.

- **Commercial Use & Integration:** You may freely integrate, run, and link this engine into commercial and proprietary stacks without forcing your outer business logic to be open-sourced.
- **Copyleft for Core Modifications:** Any direct optimizations, extensions, or bug fixes made to the engine itself (including SaaS/cloud network hosting) must be shared under the same EUPL-1.2 terms.

See `LICENSE` or the [official EUPL page](https://joinup.ec.europa.eu/software/page/eupl) for details.
