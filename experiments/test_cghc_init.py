"""
Experiment: Verify CGHC at initialization behaves like identity residual.

The claim: at init, with n streams each of block_size d:
  - Phi = I  (stream mixing is identity)
  - read_in  = (1/n) * ones [m, n]  (each backbone stream reads the stream mean)
  - write_out = 1 * ones [n, m]     (each output stream sums all backbone outputs)

So for n identical streams x[i] = v:
  - x_read[j] = (1/n) * sum_i v = v  (mean = v since all identical)
  - Y[i] = sum_j f_j  where [f_1 || ... || f_m] = module([v || ... || v])
  - output[i] = v + Y[i]   (same for every stream, since Phi=I and Y is uniform)

This is equivalent to a single-stream residual:  output = v + g(v)
where g is the module operating on embed_dim = m * block_size.

We verify:
  1. Generator A = 0 at init  → Phi = I
  2. read_in  = (1/n) 1_{m×n}
  3. write_out = 1_{n×m}
  4. With a zero module, output == input exactly (pure identity)
  5. With a linear module and identical streams, all output streams are identical
     and equal to v + g(v) — same as single-stream residual
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from hyperconnections.cghc import ContinuousGenHyperConnections


# ── helpers ──────────────────────────────────────────────────────────────────

def make_cghc(n, m, embed_dim, module, generator_type="conservative_psd_diss"):
    input_dim = int((n / m) * embed_dim)
    return ContinuousGenHyperConnections(
        n=n, m=m, input_dim=input_dim, embed_dim=embed_dim,
        module=module, generator_type=generator_type,
    )


class ZeroModule(nn.Module):
    """Always returns zeros — isolates HC structure from module content."""
    def forward(self, x, **kwargs):
        return torch.zeros_like(x)


class LinearModule(nn.Module):
    """Single linear layer (no bias) acting on embed_dim."""
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, **kwargs):
        return self.fc(x)


def check(cond, a, b, label):
    ok = torch.allclose(a, b, atol=1e-5)
    maxdiff = (a - b).abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  (max |diff| = {maxdiff:.2e})")
    return ok


# ── tests ─────────────────────────────────────────────────────────────────────

def test_static_weights(n, m, embed_dim):
    print(f"\n=== Static weight check  n={n}, m={m}, embed_dim={embed_dim} ===")
    cghc = make_cghc(n, m, embed_dim, ZeroModule())

    actual_read_in = torch.sigmoid(cghc.read_in)        # [n, m]
    check(None, actual_read_in, torch.full_like(actual_read_in, 1.0 / n),
          f"read_in → sigmoid = 1/n = {1/n:.4f}")

    actual_write_out = 2 * torch.sigmoid(cghc.write_out)  # [n, m]
    check(None, actual_write_out, torch.ones_like(actual_write_out),
          "write_out → 2·sigmoid(0) = 1")


def test_generator_is_zero(n, m, embed_dim):
    print(f"\n=== Generator A = 0 at init  n={n}, m={m}, embed_dim={embed_dim} ===")
    cghc = make_cghc(n, m, embed_dim, ZeroModule())

    B = 4
    x = torch.randn(B, n, cghc.block_size)
    A = cghc.compute_generator(x)
    check(None, A, torch.zeros_like(A), "A = 0")

    Phi = cghc.compute_transition(x)
    check(None, Phi, torch.eye(n).expand(B, -1, -1), "Phi = I")


def test_zero_module_is_identity(n, m, embed_dim):
    """With a zero module output == input exactly."""
    print(f"\n=== Zero module → output == input  n={n}, m={m}, embed_dim={embed_dim} ===")
    input_dim = int((n / m) * embed_dim)
    cghc = make_cghc(n, m, embed_dim, ZeroModule())

    B, T = 4, 7
    x = torch.randn(B, T, input_dim)
    with torch.no_grad():
        out = cghc(x)
    check(None, out, x, "output == input")


def test_identical_streams_match_single_residual(n, m, embed_dim):
    """
    With n identical streams and a linear module:
      - every output stream should be identical
      - output stream == v + sum_j g(v)_j  (single-stream residual equiv.)
    """
    print(f"\n=== Identical streams ↔ single residual  n={n}, m={m}, embed_dim={embed_dim} ===")
    block_size = embed_dim // m
    input_dim = n * block_size

    linear_mod = LinearModule(embed_dim)
    cghc = make_cghc(n, m, embed_dim, linear_mod)

    B = 4
    # All n streams hold the same block_size vector v
    v = torch.randn(B, block_size)
    x = v.unsqueeze(1).expand(B, n, block_size).reshape(B, input_dim)

    with torch.no_grad():
        out = cghc(x)  # [B, input_dim]

    out_streams = out.reshape(B, n, block_size)

    # All output streams should be identical
    for i in range(1, n):
        check(None, out_streams[:, i], out_streams[:, 0],
              f"stream {i} == stream 0")

    # Expected: v + sum over m output blocks of g(v_embed)
    # x_read = mean of n copies of v = v, tiled to embed_dim
    v_embed = v.unsqueeze(1).expand(B, m, block_size).reshape(B, embed_dim)
    with torch.no_grad():
        g_v = linear_mod(v_embed)                   # [B, embed_dim]
    Y = g_v.reshape(B, m, block_size).sum(dim=1)    # [B, block_size]
    expected = v + Y

    check(None, out_streams[:, 0], expected,
          "stream output == v + sum_j g(v)_j")


def run_all():
    configs = [
        (2, 1, 64),
        (4, 2, 64),
        (4, 1, 32),
        (8, 4, 128),
    ]
    for n, m, embed_dim in configs:
        test_static_weights(n, m, embed_dim)
        test_generator_is_zero(n, m, embed_dim)
        test_zero_module_is_identity(n, m, embed_dim)
        test_identical_streams_match_single_residual(n, m, embed_dim)


if __name__ == "__main__":
    torch.manual_seed(42)
    run_all()
    print("\nDone.")
