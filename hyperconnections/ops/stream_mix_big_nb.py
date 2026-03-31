"""
Fused stream-mixing Triton kernel — large-N×B variant.

Uses tl.dot (tensor-core backed matmul) and processes *all N output rows*
in a single CTA, which amortises the cost of loading Phi[b] and x[b, :, d_tile]
from global memory instead of relying on L2 reuse across N separate CTAs.

This variant should only be dispatched when:
  N >= 16  and  N % 16 == 0  and  B*N*D*elem_bytes > 40 MB

Below that footprint the L2 cache absorbs the cross-CTA x reuse for free,
and the generic kernel (stream_mix.py) is faster due to static-range unrolling.

Kernel layout
─────────────
  _stream_mix_fwd_big_nb   grid: (B, cdiv(D, BLOCK_D))
      One program per (b, d_tile).  Loads Phi[b] [N,N] + x[b,:,d_tile] [N,BLOCK_D],
      computes acc = tl.dot(Phi, x_tile), adds optional proj correction and Y.

  _stream_mix_bwd_dx_big_nb   grid: (B, cdiv(D, BLOCK_D))
      grad_x = tl.dot(Phi^T, G_tile) + optional v*beta term.

  _stream_mix_bwd_dPhi_big_nb   grid: (B,)
      Accumulates grad_Phi[b] = Σ_{d-tiles} tl.dot(G_tile, x_eff_tile^T) over D.
      v (loop-invariant) is loaded once before the loop.
      alpha ([B,D]) is loaded per tile.

Backward shared intermediates (proj case, precomputed in Python):
  alpha  [B, D]  = v^T x          (removes O(N²) Phi loads from bwd_dx)
  phi_v  [B, N]  = Phi @ v
  c      [B, N]  = v - phi_v
  beta   [B, D]  = einsum("bnd,bn->bd", G, c)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Autotune configs
# BLOCK_D must be a power-of-2 ≥ 16 for tl.dot on Ampere (sm_80).
# ---------------------------------------------------------------------------

_BIG_NB_FWD_CONFIGS = [
    triton.Config({"BLOCK_D": 32},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
]

_BIG_NB_DPHI_CONFIGS = [
    triton.Config({"BLOCK_D": 32},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
]


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_BIG_NB_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_fwd_big_nb(
    Phi_ptr, x_ptr, Y_ptr, out_ptr, v_ptr,
    D,
    stride_phi_b, stride_phi_n1, stride_phi_n2,
    stride_x_b,   stride_x_n,   stride_x_d,
    stride_y_b,   stride_y_n,   stride_y_d,
    stride_o_b,   stride_o_n,   stride_o_d,
    stride_v_b,   stride_v_n,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    DTYPE:     tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D
    n_idx  = tl.arange(0, N_STREAMS)

    # ---- Load Phi[b]: [N, N] ------------------------------------------------
    phi_ptrs = (
        Phi_ptr
        + pid_b * stride_phi_b
        + n_idx[:, None] * stride_phi_n1
        + n_idx[None, :] * stride_phi_n2
    )
    Phi_tile = tl.load(phi_ptrs).to(tl.float32)  # [N, N]

    # ---- Load x[b, :, d_tile]: [N, BLOCK_D] --------------------------------
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_idx[:, None] * stride_x_n
        + d_idx[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)  # [N, BLOCK_D]

    # ---- acc = Phi @ x_tile [N, BLOCK_D] via tensor cores ------------------
    acc = tl.dot(Phi_tile, x_tile, allow_tf32=False)  # [N, BLOCK_D], fp32

    # ---- Optional projection correction ------------------------------------
    if USE_PROJ:
        v_ptrs = v_ptr + pid_b * stride_v_b + n_idx * stride_v_n
        v_tile = tl.load(v_ptrs).to(tl.float32)                       # [N]

        # alpha[d] = v^T x_tile[:, d]  (partial sum over N; shape [BLOCK_D])
        alpha = tl.sum(v_tile[:, None] * x_tile, axis=0)              # [BLOCK_D]

        # phi_v[n1] = Phi[n1, :] . v  (element-wise row dot; shape [N])
        phi_v = tl.sum(Phi_tile * v_tile[None, :], axis=1)            # [N]

        # correction = (v - phi_v) * alpha  [N, BLOCK_D]
        acc = acc + (v_tile - phi_v)[:, None] * alpha[None, :]

    # ---- Add Y[b, :, d_tile] -----------------------------------------------
    Y_ptrs = (
        Y_ptr
        + pid_b * stride_y_b
        + n_idx[:, None] * stride_y_n
        + d_idx[None, :] * stride_y_d
    )
    Y_tile = tl.load(Y_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

    # ---- Store output -------------------------------------------------------
    out_ptrs = (
        out_ptr
        + pid_b * stride_o_b
        + n_idx[:, None] * stride_o_n
        + d_idx[None, :] * stride_o_d
    )
    tl.store(out_ptrs, (acc + Y_tile).to(DTYPE), mask=d_mask[None, :])


# ---------------------------------------------------------------------------
# Backward kernel: grad_x
#
# no-proj:  grad_x[b, :, d] = Phi^T @ G[b, :, d]
# proj:     grad_x[b, :, d] = Phi^T @ G[b, :, d]  +  v * beta[b, d]
#
# beta precomputed in Python — single load per d_tile, no nested N loop.
# ---------------------------------------------------------------------------

@triton.autotune(configs=_BIG_NB_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dx_big_nb(
    G_ptr, Phi_ptr, v_ptr, beta_ptr, grad_x_ptr,
    D,
    stride_g_b,    stride_g_n,    stride_g_d,
    stride_phi_b,  stride_phi_n1, stride_phi_n2,
    stride_v_b,    stride_v_n,
    stride_beta_b, stride_beta_d,
    stride_gx_b,   stride_gx_n,   stride_gx_d,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    DTYPE:     tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D
    n_idx  = tl.arange(0, N_STREAMS)

    # ---- Load Phi[b]^T: [N, N] (load then transpose) -----------------------
    phi_ptrs = (
        Phi_ptr
        + pid_b * stride_phi_b
        + n_idx[:, None] * stride_phi_n1
        + n_idx[None, :] * stride_phi_n2
    )
    Phi_tile = tl.load(phi_ptrs).to(tl.float32)   # [N, N]
    Phi_T    = tl.trans(Phi_tile)                  # [N, N] transposed

    # ---- Load G[b, :, d_tile]: [N, BLOCK_D] --------------------------------
    G_ptrs = (
        G_ptr
        + pid_b * stride_g_b
        + n_idx[:, None] * stride_g_n
        + d_idx[None, :] * stride_g_d
    )
    G_tile = tl.load(G_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)

    # ---- grad_x = Phi^T @ G [N, BLOCK_D] -----------------------------------
    grad_x_tile = tl.dot(Phi_T, G_tile, allow_tf32=False)

    # ---- Optional proj correction: v * beta ---------------------------------
    if USE_PROJ:
        v_ptrs = v_ptr + pid_b * stride_v_b + n_idx * stride_v_n
        v_tile = tl.load(v_ptrs).to(tl.float32)                       # [N]

        beta_ptrs = beta_ptr + pid_b * stride_beta_b + d_idx * stride_beta_d
        beta_vec  = tl.load(beta_ptrs, mask=d_mask, other=0.0).to(tl.float32)  # [BLOCK_D]

        grad_x_tile = grad_x_tile + v_tile[:, None] * beta_vec[None, :]

    # ---- Store grad_x -------------------------------------------------------
    gx_ptrs = (
        grad_x_ptr
        + pid_b * stride_gx_b
        + n_idx[:, None] * stride_gx_n
        + d_idx[None, :] * stride_gx_d
    )
    tl.store(gx_ptrs, grad_x_tile.to(DTYPE), mask=d_mask[None, :])


# ---------------------------------------------------------------------------
# Backward kernel: grad_Phi
#
# grad_Phi[b] = Σ_{d-tiles}  G_tile  @  x_eff_tile^T    [N, BLOCK_D] @ [BLOCK_D, N]
#
# no-proj:  x_eff = x
# proj:     x_eff[b, n, d] = x[b, n, d] - v[b, n] * alpha[b, d]
#
# v is [N], loop-invariant → loaded once before the D loop.
# alpha is [B, D]          → loaded per d_tile inside the loop.
# ---------------------------------------------------------------------------

@triton.autotune(configs=_BIG_NB_DPHI_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dPhi_big_nb(
    G_ptr, x_ptr, v_ptr, alpha_ptr, grad_Phi_ptr,
    D,
    stride_g_b,     stride_g_n,    stride_g_d,
    stride_x_b,     stride_x_n,    stride_x_d,
    stride_v_b,     stride_v_n,
    stride_alpha_b, stride_alpha_d,
    stride_gP_b,    stride_gP_n1,  stride_gP_n2,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_b = tl.program_id(0)
    n_idx = tl.arange(0, N_STREAMS)

    # Load v once — it is [N] and loop-invariant over d_tiles
    if USE_PROJ:
        v_ptrs = v_ptr + pid_b * stride_v_b + n_idx * stride_v_n
        v_tile = tl.load(v_ptrs).to(tl.float32)   # [N]

    # Accumulate grad_Phi[b] = Σ_tiles  G_tile @ x_eff_tile^T
    acc = tl.zeros([N_STREAMS, N_STREAMS], dtype=tl.float32)

    n_blocks = tl.cdiv(D, BLOCK_D)
    for i in range(n_blocks):
        d_off  = i * BLOCK_D
        d_idx  = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D

        G_ptrs = (
            G_ptr
            + pid_b * stride_g_b
            + n_idx[:, None] * stride_g_n
            + d_idx[None, :] * stride_g_d
        )
        G_tile = tl.load(G_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)  # [N, BLOCK_D]

        x_ptrs = (
            x_ptr
            + pid_b * stride_x_b
            + n_idx[:, None] * stride_x_n
            + d_idx[None, :] * stride_x_d
        )
        x_tile = tl.load(x_ptrs, mask=d_mask[None, :], other=0.0).to(tl.float32)  # [N, BLOCK_D]

        if USE_PROJ:
            alpha_ptrs = alpha_ptr + pid_b * stride_alpha_b + d_idx * stride_alpha_d
            alpha_vec  = tl.load(alpha_ptrs, mask=d_mask, other=0.0).to(tl.float32)  # [BLOCK_D]
            x_eff = x_tile - v_tile[:, None] * alpha_vec[None, :]                    # [N, BLOCK_D]
        else:
            x_eff = x_tile

        # [N, BLOCK_D] @ [BLOCK_D, N] → [N, N], fused into accumulator
        acc = tl.dot(G_tile, tl.trans(x_eff), acc=acc, allow_tf32=False)

    # ---- Store grad_Phi[b] --------------------------------------------------
    gP_ptrs = (
        grad_Phi_ptr
        + pid_b * stride_gP_b
        + n_idx[:, None] * stride_gP_n1
        + n_idx[None, :] * stride_gP_n2
    )
    tl.store(gP_ptrs, acc)


# ---------------------------------------------------------------------------
# Python helpers (mirrored from stream_mix.py to keep this file self-contained)
# ---------------------------------------------------------------------------

def _make_v_arg(v: torch.Tensor | None, B: int, N: int, device, dtype):
    if v is not None:
        return v.contiguous()
    return torch.zeros(B, N, dtype=dtype, device=device)


def _make_bd_arg(t: torch.Tensor | None, B: int, D: int, device):
    if t is not None:
        return t.contiguous()
    return torch.zeros(B, D, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Launch wrappers
# ---------------------------------------------------------------------------

_TORCH_TO_TL_DTYPE = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


def _launch_fwd(Phi, x, Y, v, out):
    B, N, D  = x.shape
    use_proj = v is not None
    v_arg    = _make_v_arg(v, B, N, x.device, x.dtype)
    tl_dtype = _TORCH_TO_TL_DTYPE[x.dtype]
    grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_fwd_big_nb[grid](
        Phi, x, Y, out, v_arg,
        D,
        *Phi.stride(), *x.stride(), *Y.stride(), *out.stride(), *v_arg.stride(),
        N_STREAMS=N, USE_PROJ=use_proj, DTYPE=tl_dtype,
    )


def _launch_bwd_dx(G, Phi, v, beta, grad_x, N, out_dtype):
    B, _, D  = G.shape
    use_proj = v is not None
    v_arg    = _make_v_arg(v, B, N, G.device, G.dtype)
    beta_arg = _make_bd_arg(beta, B, D, G.device)
    tl_dtype = _TORCH_TO_TL_DTYPE[out_dtype]
    grid = lambda meta: (B, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_bwd_dx_big_nb[grid](
        G, Phi, v_arg, beta_arg, grad_x,
        D,
        *G.stride(), *Phi.stride(), *v_arg.stride(), *beta_arg.stride(), *grad_x.stride(),
        N_STREAMS=N, USE_PROJ=use_proj, DTYPE=tl_dtype,
    )


def _launch_bwd_dPhi(G, x, v, alpha, grad_Phi, N):
    B, _, D = G.shape
    use_proj  = v is not None
    v_arg     = _make_v_arg(v, B, N, G.device, G.dtype)
    alpha_arg = _make_bd_arg(alpha, B, D, G.device)
    grid = (B,)
    _stream_mix_bwd_dPhi_big_nb[grid](
        G, x, v_arg, alpha_arg, grad_Phi,
        D,
        *G.stride(), *x.stride(), *v_arg.stride(), *alpha_arg.stride(), *grad_Phi.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _StreamMixBigNBFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Phi, x, Y, v):
        Phi_c = Phi.contiguous()
        x_c   = x.contiguous()
        Y_c   = Y.contiguous()
        v_c   = v.contiguous() if v is not None else None

        out = torch.empty_like(x_c)
        _launch_fwd(Phi_c, x_c, Y_c, v_c, out)

        ctx.save_for_backward(Phi_c, x_c, v_c)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Phi, x, v = ctx.saved_tensors
        B, N, D   = x.shape
        use_proj  = v is not None

        G = grad_out.float().contiguous()

        # ---- Shared intermediates (proj only) --------------------------------
        alpha = beta = phi_v = c = None
        if use_proj:
            alpha = torch.einsum("bn,bnd->bd", v.float(), x.float())             # [B, D]
            phi_v = torch.bmm(Phi.float(), v.float().unsqueeze(-1)).squeeze(-1)  # [B, N]
            c     = v.float() - phi_v                                             # [B, N]
            beta  = torch.einsum("bnd,bn->bd", G, c)                             # [B, D]

        # ---- grad_x (Triton) ------------------------------------------------
        grad_x = torch.empty_like(x)
        _launch_bwd_dx(G, Phi, v, beta, grad_x, N, out_dtype=x.dtype)

        # ---- grad_Phi (Triton) ----------------------------------------------
        grad_Phi = torch.empty(B, N, N, dtype=torch.float32, device=x.device)
        _launch_bwd_dPhi(G, x, v, alpha, grad_Phi, N)

        # ---- grad_Y = grad_out (identity) -----------------------------------
        grad_Y = grad_out

        # ---- grad_v (PyTorch) -----------------------------------------------
        grad_v = None
        if use_proj and ctx.needs_input_grad[3]:
            rho       = (G * alpha.unsqueeze(1)).sum(dim=2)
            rho_part  = rho - torch.bmm(Phi.float().mT, rho.unsqueeze(-1)).squeeze(-1)
            beta_part = torch.einsum("bd,bnd->bn", beta, x.float())
            grad_v    = (rho_part + beta_part).to(v.dtype)

        return (
            grad_Phi.to(Phi.dtype) if ctx.needs_input_grad[0] else None,
            grad_x if ctx.needs_input_grad[1] else None,
            grad_Y if ctx.needs_input_grad[2] else None,
            grad_v,
        )


# ---------------------------------------------------------------------------
# Public API (used by the dispatcher in ops/__init__.py)
# ---------------------------------------------------------------------------

def stream_mix_add_big_nb(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Big-N×B variant of stream_mix_add.

    Uses tl.dot (tensor cores) and a (B, cdiv(D, BLOCK_D)) grid so that
    Phi[b] and x[b, :, d_tile] are each loaded from global memory exactly
    once per CTA, rather than N times across N separate CTAs.

    Preconditions (checked by the dispatcher):
      - x.is_cuda
      - N >= 16 and N % 16 == 0
      - B * N * D * elem_bytes > 40 MB
    """
    B, N, D = x.shape
    return _StreamMixBigNBFn.apply(Phi, x, Y, v)
