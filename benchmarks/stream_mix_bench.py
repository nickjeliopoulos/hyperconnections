"""
Numerical correctness and performance benchmark for stream_mix_add.

Usage
-----
# Run everything (default):
    python benchmarks/stream_mix_bench.py

# Only correctness:
    python benchmarks/stream_mix_bench.py --mode correctness

# Only performance:
    python benchmarks/stream_mix_bench.py --mode perf

# Restrict to specific N values:
    python benchmarks/stream_mix_bench.py --mode perf --n 4 8

# Only float16:
    python benchmarks/stream_mix_bench.py --mode perf --dtype fp16

Requirements: CUDA GPU, triton, torch.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from typing import Sequence

import torch
import triton
import triton.testing

from hyperconnections.ops.stream_mix import stream_mix_add

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda"

_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"


def _col(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if sys.stdout.isatty() else text


def ok(s="PASS"):
    return _col(s, _GREEN)


def fail(s):
    return _col(s, _RED)


def warn(s):
    return _col(s, _YELLOW)


def bold(s):
    return _col(s, _BOLD)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _make(B, N, D, dtype, seed=0):
    torch.manual_seed(seed)
    Phi = torch.randn(B, N, N, device=DEVICE, dtype=dtype)
    x   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    Y   = torch.randn(B, N, D, device=DEVICE, dtype=dtype)
    return Phi, x, Y


def _make_v(B, N, dtype, seed=7):
    torch.manual_seed(seed)
    v = torch.randn(B, N, device=DEVICE, dtype=dtype)
    return torch.nn.functional.normalize(v, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Reference implementations
# ──────────────────────────────────────────────────────────────────────────────

def ref_no_proj(Phi, x, Y):
    return torch.bmm(Phi, x) + Y


def ref_proj(Phi, x, Y, v):
    alpha = torch.einsum("bn, bnd -> bd", v, x)
    Phi_v = torch.bmm(Phi, v.unsqueeze(-1)).squeeze(-1)
    return torch.bmm(Phi, x) + (v - Phi_v).unsqueeze(2) * alpha.unsqueeze(1) + Y


def ref_proj_backward(Phi, x, Y, v):
    """Return (grad_Phi, grad_x, grad_Y) using PyTorch autograd on ref_proj."""
    Phi_r = Phi.detach().float().requires_grad_(True)
    x_r   = x.detach().float().requires_grad_(True)
    Y_r   = Y.detach().float().requires_grad_(True)
    v_r   = v.detach().float()
    ref_proj(Phi_r, x_r, Y_r, v_r).sum().backward()
    return Phi_r.grad, x_r.grad, Y_r.grad


def ref_no_proj_backward(Phi, x, Y):
    Phi_r = Phi.detach().float().requires_grad_(True)
    x_r   = x.detach().float().requires_grad_(True)
    Y_r   = Y.detach().float().requires_grad_(True)
    ref_no_proj(Phi_r, x_r, Y_r).sum().backward()
    return Phi_r.grad, x_r.grad, Y_r.grad


# ──────────────────────────────────────────────────────────────────────────────
# Correctness checks
# ──────────────────────────────────────────────────────────────────────────────

def _check(label: str, got: torch.Tensor, ref: torch.Tensor, atol: float) -> tuple[bool, float]:
    diff = (got.float() - ref.float()).abs()
    max_err = diff.max().item()
    passed = max_err <= atol
    return passed, max_err


# Column widths for the correctness table
_CORR_HDR = f"{'Config':>30}  {'Variant':>12}  {'Check':>10}  {'MaxErr':>10}  {'atol':>8}  Result"
_CORR_SEP = "-" * 90


def _corr_row(config, variant, check, max_err, atol, passed):
    result = ok("PASS") if passed else fail("FAIL")
    return f"{config:>30}  {variant:>12}  {check:>10}  {max_err:>10.2e}  {atol:>8.0e}  {result}"


def run_correctness(ns: Sequence[int], dtypes: Sequence[str]):
    """Run forward + backward correctness checks and print a summary table."""
    print()
    print(bold("=" * 90))
    print(bold("  CORRECTNESS"))
    print(bold("=" * 90))
    print(_CORR_HDR)
    print(_CORR_SEP)

    configs = list(product(
        [32, 256, 1024],   # B
        ns,                # N
        [64, 128, 256],    # D
    ))
    # Add one non-power-of-2 D to test masking
    configs += [(4, ns[0], 100)]

    all_passed = True

    for dtype_name in dtypes:
        dtype  = _dtype(dtype_name)
        # fp16 tolerances are looser (kernel accumulates in fp32 but stores in fp16)
        atol_f = 1e-3 if dtype == torch.float32 else 2e-2
        atol_b = 2e-3 if dtype == torch.float32 else 4e-2

        for B, N, D in configs:
            Phi, x, Y = _make(B, N, D, dtype)
            v = _make_v(B, N, dtype)
            cfg_str = f"B={B} N={N} D={D} {dtype_name}"

            # ---- forward no-proj ----
            got = stream_mix_add(Phi, x, Y)
            ref = ref_no_proj(Phi.float(), x.float(), Y.float()).to(dtype)
            passed, err = _check("", got, ref, atol_f)
            all_passed &= passed
            print(_corr_row(cfg_str, "no-proj", "fwd", err, atol_f, passed))

            # ---- forward proj ----
            got = stream_mix_add(Phi, x, Y, v=v)
            ref = ref_proj(Phi.float(), x.float(), Y.float(), v.float()).to(dtype)
            passed, err = _check("", got, ref, atol_f)
            all_passed &= passed
            print(_corr_row(cfg_str, "proj", "fwd", err, atol_f, passed))

            # backward only for fp32 (avoids casting complexity in reference)
            if dtype == torch.float32:
                # ---- backward no-proj ----
                Phi_t = Phi.detach().requires_grad_(True)
                x_t   = x.detach().requires_grad_(True)
                Y_t   = Y.detach().requires_grad_(True)
                stream_mix_add(Phi_t, x_t, Y_t).sum().backward()
                gP_r, gx_r, gY_r = ref_no_proj_backward(Phi, x, Y)

                for name, got_g, ref_g in [
                    ("grad_Phi", Phi_t.grad, gP_r),
                    ("grad_x",   x_t.grad,   gx_r),
                    ("grad_Y",   Y_t.grad,   gY_r),
                ]:
                    passed, err = _check("", got_g, ref_g, atol_b)
                    all_passed &= passed
                    print(_corr_row(cfg_str, "no-proj", name, err, atol_b, passed))

                # ---- backward proj ----
                Phi_t = Phi.detach().requires_grad_(True)
                x_t   = x.detach().requires_grad_(True)
                Y_t   = Y.detach().requires_grad_(True)
                stream_mix_add(Phi_t, x_t, Y_t, v=v).sum().backward()
                gP_r, gx_r, gY_r = ref_proj_backward(Phi, x, Y, v)

                for name, got_g, ref_g in [
                    ("grad_Phi", Phi_t.grad, gP_r),
                    ("grad_x",   x_t.grad,   gx_r),
                    ("grad_Y",   Y_t.grad,   gY_r),
                ]:
                    passed, err = _check("", got_g, ref_g, atol_b)
                    all_passed &= passed
                    print(_corr_row(cfg_str, "proj", name, err, atol_b, passed))

        print(_CORR_SEP)

    print()
    if all_passed:
        print(ok("All correctness checks passed."))
    else:
        print(fail("One or more correctness checks FAILED."))
    print()
    return all_passed


# ──────────────────────────────────────────────────────────────────────────────
# Performance benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _bytes_no_proj(B, N, D, elem_bytes):
    """Bytes touched in the ideal (no data reuse) model: read Phi, x, Y; write out."""
    return elem_bytes * (B * N * N + 3 * B * N * D)


def _bytes_proj(B, N, D, elem_bytes):
    """Same as no-proj plus reading v once per row (N reads × N rows per batch)."""
    return elem_bytes * (B * N * N + 3 * B * N * D + B * N)


_PERF_HDR = (
    f"{'Config':>30}  {'Variant':>12}  {'dtype':>6}  "
    f"{'Triton ms':>10}  {'PyTorch ms':>11}  {'Speedup':>8}  {'BW GB/s':>9}"
)
_PERF_SEP = "-" * 100


def _perf_row(config, variant, dtype_name, t_tri, t_ref, bw_gbs):
    speedup = t_ref / t_tri
    sp_str = f"{speedup:.2f}x"
    sp_col = ok(sp_str) if speedup >= 1.05 else (warn(sp_str) if speedup >= 0.95 else fail(sp_str))
    return (
        f"{config:>30}  {variant:>12}  {dtype_name:>6}  "
        f"{t_tri:>10.3f}  {t_ref:>11.3f}  {sp_col:>8}  {bw_gbs:>9.1f}"
    )


def run_perf(ns: Sequence[int], dtypes: Sequence[str], warmup: int = 25, rep: int = 200):
    """Benchmark Triton kernel vs PyTorch bmm+add reference."""
    print()
    print(bold("=" * 100))
    print(bold("  PERFORMANCE"))
    print(bold("=" * 100))
    print(_PERF_HDR)
    print(_PERF_SEP)

    # Realistic CGHC configurations
    B_vals = [64, 512, 2048, 8192]
    D_vals = [64, 128, 256, 512]

    for dtype_name in dtypes:
        dtype = _dtype(dtype_name)
        elem  = torch.finfo(dtype).bits // 8

        for N, B, D in product(ns, B_vals, D_vals):
            Phi, x, Y = _make(B, N, D, dtype)
            v = _make_v(B, N, dtype)
            cfg_str = f"B={B} N={N} D={D}"

            # ---- no-proj ----
            t_tri = triton.testing.do_bench(
                lambda: stream_mix_add(Phi, x, Y),
                warmup=warmup, rep=rep,
            )
            t_ref = triton.testing.do_bench(
                lambda: ref_no_proj(Phi, x, Y),
                warmup=warmup, rep=rep,
            )
            bw = _bytes_no_proj(B, N, D, elem) / (t_tri * 1e-3) / 1e9
            print(_perf_row(cfg_str, "no-proj", dtype_name, t_tri, t_ref, bw))

            # ---- proj ----
            t_tri_p = triton.testing.do_bench(
                lambda: stream_mix_add(Phi, x, Y, v=v),
                warmup=warmup, rep=rep,
            )
            # Reference: the full PyTorch projected computation
            t_ref_p = triton.testing.do_bench(
                lambda: ref_proj(Phi, x, Y, v),
                warmup=warmup, rep=rep,
            )
            bw_p = _bytes_proj(B, N, D, elem) / (t_tri_p * 1e-3) / 1e9
            print(_perf_row(cfg_str, "proj", dtype_name, t_tri_p, t_ref_p, bw_p))

        print(_PERF_SEP)

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="stream_mix_add benchmark")
    parser.add_argument(
        "--mode", choices=["correctness", "perf", "all"], default="all",
        help="Which sections to run (default: all)",
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=[4, 8, 16],
        metavar="N", help="N_STREAMS values to benchmark (default: 4 8 16)",
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], nargs="+",
        default=["fp32", "fp16"], metavar="DTYPE",
        help="dtypes to test (default: fp32 fp16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=25,
        help="Triton do_bench warmup iterations (default: 25)",
    )
    parser.add_argument(
        "--rep", type=int, default=200,
        help="Triton do_bench repetitions (default: 200)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(fail("No CUDA device found. Exiting."))
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    print(f"\nDevice : {dev}")
    print(f"N vals : {args.n}")
    print(f"dtypes : {args.dtype}")

    passed = True
    if args.mode in ("correctness", "all"):
        passed = run_correctness(args.n, args.dtype)

    if args.mode in ("perf", "all"):
        run_perf(args.n, args.dtype, warmup=args.warmup, rep=args.rep)

    if args.mode in ("correctness", "all") and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
