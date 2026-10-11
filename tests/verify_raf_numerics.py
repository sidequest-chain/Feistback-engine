"""
Verification Script for RAF (Reversible Associative Feistel) Numerics
====================================================================
Tests numerical equivalence between:
  1. Parallel causal linear attention (training mode, L=32)
  2. Step-by-step recurrent state tracking (inference mode, O(1) state)
"""

from pathlib import Path
import sys
import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime.models.associative_memory import AssociativeMemoryBlock


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify_raf_numerics] Running verification on device: {device}")

    d_model = 448
    num_heads = 4
    key_dim = 16
    value_dim = 16
    B, L = 2, 32

    block = AssociativeMemoryBlock(
        d_model=d_model,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        decay_bias_init=2.5,
    ).to(device, dtype=torch.float32)
    block.eval()

    # Parameter count reporting
    param_count = sum(p.numel() for p in block.parameters())
    print(f"[verify_raf_numerics] AssociativeMemoryBlock parameters per layer: {param_count:,}")

    # Generate random test tensor [2, 32, 448]
    x = torch.randn(B, L, d_model, device=device, dtype=torch.float32)

    # 1. Parallel mode (without initial state)
    with torch.no_grad():
        out_parallel = block.forward_parallel(x)  # [B, L, d_model]

    # 2. Recurrent mode (step-by-step over 32 steps)
    state = None
    recurrent_steps = []
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t : t + 1, :]
            out_t, state = block.forward_recurrent(x_t, state=state)
            recurrent_steps.append(out_t)
        out_recurrent = torch.cat(recurrent_steps, dim=1)  # [B, L, d_model]

    # 3. Numerical checks
    max_abs_diff = (out_parallel - out_recurrent).abs().max().item()
    mean_abs_diff = (out_parallel - out_recurrent).abs().mean().item()

    print(f"[verify_raf_numerics] Shape parallel:  {list(out_parallel.shape)}")
    print(f"[verify_raf_numerics] Shape recurrent: {list(out_recurrent.shape)}")
    print(f"[verify_raf_numerics] Max absolute diff:  {max_abs_diff:.6e}")
    print(f"[verify_raf_numerics] Mean absolute diff: {mean_abs_diff:.6e}")

    # Check state equivalence at step L
    with torch.no_grad():
        _, state_from_parallel = block.forward_parallel(x, return_state=True)
    state_diff = (state - state_from_parallel).abs().max().item()
    print(f"[verify_raf_numerics] Max state matrix diff: {state_diff:.6e}")

    tolerance = 1e-4
    if max_abs_diff < tolerance and state_diff < tolerance:
        print(f"[verify_raf_numerics] SUCCESS: All numerical differences are below tolerance ({tolerance}).")
    else:
        print(f"[verify_raf_numerics] FAILURE: Differences exceed tolerance {tolerance}!")
        sys.exit(1)


if __name__ == "__main__":
    main()
