"""
Verification Script for RAF (Reversible Associative Feistel) Numerics
====================================================================
Tests numerical equivalence between:
  1. AssociativeMemoryBlock: Parallel causal scan (L=32) vs step-by-step recurrent mode (L=1)
  2. Full PureFeistelBackbone (12 layers, RAF + SpectralMixer): Parallel (L=32) vs Recurrent (L=1)
  3. Analytic Feistel Inverse Reconstruction: R_i = L_{i+1}, L_i = R_{i+1} - gamma * F(L_{i+1})
  4. Reversible Backpropagation Gradients: O(1) ReversibleFeistelBackboneFunction vs standard autograd
"""

from pathlib import Path
import sys
import torch

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime.models.associative_memory import AssociativeMemoryBlock
from runtime.models.pure_feistel_lm import PureFeistelBackbone


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify_raf_numerics] Running verification on device: {device}")

    d_half = 448
    d_model = 896
    num_heads = 4
    key_dim = 16
    value_dim = 16
    B, L = 2, 32
    tolerance = 1e-4

    # =========================================================================
    # 1. AssociativeMemoryBlock Parallel vs Recurrent Parity
    # =========================================================================
    block = AssociativeMemoryBlock(
        d_model=d_half,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        decay_bias_init=2.5,
    ).to(device, dtype=torch.float32)
    block.eval()

    param_count = sum(p.numel() for p in block.parameters())
    print(f"[verify_raf_numerics] 1. AssociativeMemoryBlock parameters per layer: {param_count:,}")

    x = torch.randn(B, L, d_half, device=device, dtype=torch.float32)

    with torch.no_grad():
        out_parallel, state_from_parallel = block.forward_parallel(x, return_state=True)

    state = None
    recurrent_steps = []
    with torch.no_grad():
        for t in range(L):
            x_t = x[:, t : t + 1, :]
            out_t, state = block.forward_recurrent(x_t, state=state)
            recurrent_steps.append(out_t)
        out_recurrent = torch.cat(recurrent_steps, dim=1)

    max_abs_diff = (out_parallel - out_recurrent).abs().max().item()
    mean_abs_diff = (out_parallel - out_recurrent).abs().mean().item()
    state_diff = (state - state_from_parallel).abs().max().item()

    print(f"  Max output diff (parallel vs L=1): {max_abs_diff:.6e}")
    print(f"  Mean output diff                 : {mean_abs_diff:.6e}")
    print(f"  Max state matrix diff            : {state_diff:.6e}")

    assert max_abs_diff < tolerance and state_diff < tolerance, (
        f"AssociativeMemoryBlock parity failed: {max_abs_diff} / {state_diff}"
    )

    # =========================================================================
    # 2. Full 12-Layer PureFeistelBackbone Parallel vs Step-by-Step Parity
    # =========================================================================
    backbone = PureFeistelBackbone(
        num_layers=12,
        d_model=d_model,
        max_seq_len=128,
        memory_type="raf",
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        use_reversible_autograd=True,
    ).to(device, dtype=torch.float32)
    backbone.eval()

    x_full = torch.randn(B, L, d_model, device=device, dtype=torch.float32)

    with torch.no_grad():
        y_parallel, _ = backbone(x_full, states=None)

        layer_states = backbone.init_states(B, device, torch.float32)
        step_outputs = []
        for t in range(L):
            x_step = x_full[:, t : t + 1, :]
            y_step, layer_states = backbone(x_step, states=layer_states)
            step_outputs.append(y_step)
        y_recurrent = torch.cat(step_outputs, dim=1)

    backbone_diff = (y_parallel - y_recurrent).abs().max().item()
    print(f"[verify_raf_numerics] 2. Full 12-Layer PureFeistelBackbone (RAF + SpectralMixer):")
    print(f"  Max output diff (parallel vs L=1): {backbone_diff:.6e}")
    assert backbone_diff < tolerance, f"PureFeistelBackbone L=1 parity failed: {backbone_diff}"

    # =========================================================================
    # 3. Analytic Feistel Inverse Reconstruction Check
    # =========================================================================
    max_rev_err, _ = backbone.verify_reversibility(x_full)
    print(f"[verify_raf_numerics] 3. Analytic Feistel Inverse Reconstruction:")
    print(f"  Max 12-layer reconstruction error: {max_rev_err:.6e}")
    assert max_rev_err < tolerance, f"Analytic inverse reconstruction error too high: {max_rev_err}"

    # =========================================================================
    # 4. Reversible Autograd Backward Gradient Parity Check
    # =========================================================================
    backbone.train()
    x_rev = x_full.detach().clone().requires_grad_(True)
    y_rev, _ = backbone(x_rev)
    loss_rev = y_rev.pow(2).mean()
    loss_rev.backward()
    grad_x_rev = x_rev.grad.detach().clone()
    grad_w0_rev = backbone.layers[0].spectral_mixer.weight.grad.detach().clone()

    backbone.zero_grad()
    backbone.use_reversible_autograd = False
    x_std = x_full.detach().clone().requires_grad_(True)
    y_std, _ = backbone(x_std)
    loss_std = y_std.pow(2).mean()
    loss_std.backward()
    grad_x_std = x_std.grad.detach().clone()
    grad_w0_std = backbone.layers[0].spectral_mixer.weight.grad.detach().clone()
    backbone.use_reversible_autograd = True

    grad_x_diff = (grad_x_rev - grad_x_std).abs().max().item()
    grad_w_diff = (grad_w0_rev - grad_w0_std).abs().max().item()
    print(f"[verify_raf_numerics] 4. O(1) Reversible Backpropagation Gradient Parity:")
    print(f"  Max input grad diff (dL/dX)      : {grad_x_diff:.6e}")
    print(f"  Max layer-0 param grad diff      : {grad_w_diff:.6e}")
    assert grad_x_diff < tolerance and grad_w_diff < tolerance, (
        f"Reversible backward gradient mismatch: {grad_x_diff} / {grad_w_diff}"
    )

    print(f"[verify_raf_numerics] SUCCESS: All 4 numerical & reversibility checks passed (< {tolerance}).")


if __name__ == "__main__":
    main()
