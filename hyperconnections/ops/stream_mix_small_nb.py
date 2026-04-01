"""
Fused stream-mixing Triton kernel — small-N×B variant.

Computes  out = Phi @ x + Y                               [no-proj]
or        out = Phi @ x + (v - Phi@v)(v^T x) + Y         [proj]

where:
  Phi  [B, N, N]   transition matrix
  x    [B, N, D]   stream state
  Y    [B, N, D]   source term
  v    [B, N]      unit-norm projection direction (None → no-proj)
  out  [B, N, D]

Accumulation is always in float32 regardless of input dtype.

This variant should be dispatched when either N < 16 or the x footprint
(B*N*D*elem_bytes) fits within ~75% of L2 cache, so cross-CTA reuse of
x[b, n2, d_tile] across the N programs per batch element is served from
L2 without explicit shared memory staging.

Kernel layout (forward and bwd_dx):
  Grid: (B * N, cdiv(D, BLOCK_D))
  One program per output row (b, n_out), vectorised over a D-tile.
  N_STREAMS is tl.constexpr so the inner loop is fully unrolled.

Kernel layout (bwd_dPhi):
  Grid: (B * N * N,)
  One program per scalar grad_Phi[b, n1, n2], loops over D.

Backward shared intermediates (proj case only, computed once in Python):
  alpha  [B, D]  = v^T x
  phi_v  [B, N]  = Phi @ v
  c      [B, N]  = v - phi_v
  beta   [B, D]  = einsum("bnd,bn->bd", G, c)

Precomputing alpha and beta in Python eliminates O(N²) redundant Phi
loads from bwd_dx and O(N³) redundant x loads from bwd_dPhi.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

###
### Autotune configs
### num_stages enables software pipelining; narrower BLOCK_D options give the
### autotuner room to avoid register spill at large N.
###

_FWD_CONFIGS = [
    triton.Config({"BLOCK_D": 32},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_D": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
]

_DPHI_CONFIGS = [
    triton.Config({"BLOCK_D": 64},  num_warps=2, num_stages=3),
    triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=2),
]


###
### Forward kernel
###
@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_fwd(
    Phi_ptr, x_ptr, Y_ptr, out_ptr, v_ptr,
    D,
    stride_phi_b, stride_phi_n1, stride_phi_n2,
    stride_x_b,   stride_x_n,   stride_x_d,
    stride_y_b,   stride_y_n,   stride_y_d,
    stride_o_b,   stride_o_n,   stride_o_d,
    stride_v_b,   stride_v_n,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d  = tl.program_id(1)
    b  = pid_bn // N_STREAMS
    n1 = pid_bn %  N_STREAMS

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    if USE_PROJ:
        alpha = tl.zeros([BLOCK_D], dtype=tl.float32)  # v^T x
        phi_v = 0.0   # Phi[n1,:] @ v  (scalar)
        v_n1  = tl.load(v_ptr + b * stride_v_b + n1 * stride_v_n).to(tl.float32)

    # Inner loop — fully unrolled at compile time (N_STREAMS is constexpr)
    for n2 in tl.static_range(N_STREAMS):
        phi_val = tl.load(
            Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2 * stride_phi_n2
        ).to(tl.float32)
        x_vec = tl.load(
            x_ptr + b * stride_x_b + n2 * stride_x_n + d_idx * stride_x_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        acc = acc + phi_val * x_vec

        if USE_PROJ:
            v_n2  = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)
            alpha = alpha + v_n2 * x_vec
            phi_v = phi_v + phi_val * v_n2

    if USE_PROJ:
        acc = acc + (v_n1 - phi_v) * alpha

    y_vec = tl.load(
        Y_ptr + b * stride_y_b + n1 * stride_y_n + d_idx * stride_y_d,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    tl.store(
        out_ptr + b * stride_o_b + n1 * stride_o_n + d_idx * stride_o_d,
        acc + y_vec,
        mask=d_mask,
    )


###
### Backward kernel: grad_x
###
### no-proj:  grad_x[b, n2, d] = (Phi^T @ G)[b, n2, d]
### proj:     grad_x[b, n2, d] = (Phi^T @ G)[b, n2, d] + v[b, n2] * beta[b, d]
###
### beta[b, d] = Σ_n1( G[b,n1,d] * c[b,n1] )  where c = v - Phi@v
### beta is precomputed in Python and passed as beta_ptr.
### This removes the O(N²) nested loop from the original implementation.
###

@triton.autotune(configs=_FWD_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dx(
    G_ptr, Phi_ptr, v_ptr, beta_ptr, grad_x_ptr,
    D,
    stride_g_b,    stride_g_n,    stride_g_d,
    stride_phi_b,  stride_phi_n1, stride_phi_n2,
    stride_v_b,    stride_v_n,
    stride_beta_b, stride_beta_d,
    stride_gx_b,   stride_gx_n,   stride_gx_d,
    N_STREAMS: tl.constexpr,
    USE_PROJ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d  = tl.program_id(1)
    b  = pid_bn // N_STREAMS
    n2 = pid_bn %  N_STREAMS

    d_off  = pid_d * BLOCK_D
    d_idx  = d_off + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    if USE_PROJ:
        v_n2 = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)

    # Phi^T @ G: load column n2 of Phi (= row n2 of Phi^T)
    for n1 in tl.static_range(N_STREAMS):
        phi_n1_n2 = tl.load(
            Phi_ptr + b * stride_phi_b + n1 * stride_phi_n1 + n2 * stride_phi_n2
        ).to(tl.float32)
        g_vec = tl.load(
            G_ptr + b * stride_g_b + n1 * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        acc = acc + phi_n1_n2 * g_vec

    if USE_PROJ:
        # beta precomputed in Python — single load, no nested N loop
        beta_vec = tl.load(
            beta_ptr + b * stride_beta_b + d_idx * stride_beta_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        acc = acc + v_n2 * beta_vec

    tl.store(
        grad_x_ptr + b * stride_gx_b + n2 * stride_gx_n + d_idx * stride_gx_d,
        acc,
        mask=d_mask,
    )


###
### Backward kernel: grad_Phi
###
### grad_Phi[b, n1, n2] = Σ_d( G[b,n1,d] * x_eff[b,n2,d] )
###
### no-proj:  x_eff = x
### proj:     x_eff = x[n2] - v[n2] * alpha        (alpha precomputed in Python)
###
### alpha[b, d] = Σ_n( v[b,n] * x[b,n,d] )
### Passing alpha as alpha_ptr removes the O(N³) inner static_range(N) loop.
###
@triton.autotune(configs=_DPHI_CONFIGS, key=["D", "N_STREAMS"])
@triton.jit
def _stream_mix_bwd_dPhi(
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
    pid = tl.program_id(0)
    b  = pid // (N_STREAMS * N_STREAMS)
    n1 = (pid // N_STREAMS) % N_STREAMS
    n2 = pid % N_STREAMS

    if USE_PROJ:
        v_n2 = tl.load(v_ptr + b * stride_v_b + n2 * stride_v_n).to(tl.float32)

    dp_acc = 0.0

    n_blocks = tl.cdiv(D, BLOCK_D)
    for i in range(n_blocks):
        d_off  = i * BLOCK_D
        d_idx  = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D

        g_vec = tl.load(
            G_ptr + b * stride_g_b + n1 * stride_g_n + d_idx * stride_g_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)
        x_vec = tl.load(
            x_ptr + b * stride_x_b + n2 * stride_x_n + d_idx * stride_x_d,
            mask=d_mask, other=0.0,
        ).to(tl.float32)

        if USE_PROJ:
            # alpha precomputed in Python: single load, no inner N loop
            alpha_vec = tl.load(
                alpha_ptr + b * stride_alpha_b + d_idx * stride_alpha_d,
                mask=d_mask, other=0.0,
            ).to(tl.float32)
            x_vec = x_vec - v_n2 * alpha_vec   # x_eff

        dp_acc = dp_acc + tl.sum(g_vec * x_vec, axis=0)

    tl.store(
        grad_Phi_ptr + b * stride_gP_b + n1 * stride_gP_n1 + n2 * stride_gP_n2,
        dp_acc,
    )


###
### Python helpers
###
def _make_v_arg(v: torch.Tensor | None, B: int, N: int, device, dtype):
    if v is not None:
        return v.contiguous()
    return torch.zeros(B, N, dtype=dtype, device=device)


def _make_bd_arg(t: torch.Tensor | None, B: int, D: int, device):
    """[B, D] dummy when t is None; strides are never dereferenced."""
    if t is not None:
        return t.contiguous()
    return torch.zeros(B, D, dtype=torch.float32, device=device)


def _launch_fwd(Phi, x, Y, v, out):
    B, N, D = x.shape
    use_proj = v is not None
    v_arg = _make_v_arg(v, B, N, x.device, x.dtype)
    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_fwd[grid](
        Phi, x, Y, out, v_arg,
        D,
        *Phi.stride(), *x.stride(), *Y.stride(), *out.stride(), *v_arg.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


def _launch_bwd_dx(G, Phi, v, beta, grad_x, N):
    B, _, D = G.shape
    use_proj = v is not None
    v_arg    = _make_v_arg(v, B, N, G.device, G.dtype)
    beta_arg = _make_bd_arg(beta, B, D, G.device)
    grid = lambda meta: (B * N, triton.cdiv(D, meta["BLOCK_D"]))
    _stream_mix_bwd_dx[grid](
        G, Phi, v_arg, beta_arg, grad_x,
        D,
        *G.stride(), *Phi.stride(), *v_arg.stride(), *beta_arg.stride(), *grad_x.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


def _launch_bwd_dPhi(G, x, v, alpha, grad_Phi, N):
    B, _, D = G.shape
    use_proj  = v is not None
    v_arg     = _make_v_arg(v, B, N, G.device, G.dtype)
    alpha_arg = _make_bd_arg(alpha, B, D, G.device)
    grid = lambda meta: (B * N * N,)
    _stream_mix_bwd_dPhi[grid](
        G, x, v_arg, alpha_arg, grad_Phi,
        D,
        *G.stride(), *x.stride(), *v_arg.stride(), *alpha_arg.stride(), *grad_Phi.stride(),
        N_STREAMS=N, USE_PROJ=use_proj,
    )


###
### Autograd Function
###
class _StreamMixFn(torch.autograd.Function):
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

        ### Shared intermediates (proj only)
        # Computed once in Python; eliminates O(N²) Phi loads from bwd_dx and
        # O(N³) x loads from bwd_dPhi that the previous per-program loops used.
        # Explicit .float() ensures fp32 precision for Python-side intermediates
        # regardless of input dtype (fp16 inputs stay fp16 in the Triton kernels,
        # which do their own .to(tl.float32) on loads).
        alpha = beta = phi_v = c = None
        if use_proj:
            alpha = torch.einsum("bn,bnd->bd", v.float(), x.float())            # [B, D]
            phi_v = torch.bmm(Phi.float(), v.float().unsqueeze(-1)).squeeze(-1) # [B, N]
            c     = v.float() - phi_v                                            # [B, N]
            beta  = torch.einsum("bnd,bn->bd", G, c)                            # [B, D]

        ### grad_x (Triton)
        grad_x = torch.empty_like(x)
        _launch_bwd_dx(G, Phi, v, beta, grad_x, N)

        ### grad_Phi (Triton)
        grad_Phi = torch.empty(B, N, N, dtype=torch.float32, device=x.device)
        _launch_bwd_dPhi(G, x, v, alpha, grad_Phi, N)

        ### grad_Y = grad_out (identity)
        grad_Y = grad_out

        ### grad_v (PyTorch)
        # Differentiating through both alpha = v^T x and c = v - Phi@v:
        #
        #   rho[b,n]    = Σ_d G[b,n,d] * alpha[b,d]
        #   rho_part    = (I - Phi^T) @ rho          [d/dv of the c-term]
        #   beta_part   = einsum("bd,bnd->bn", beta, x)  [d/dv of the alpha-term]
        #   grad_v      = rho_part + beta_part
        grad_v = None
        if use_proj and ctx.needs_input_grad[3]:
            rho       = (G * alpha.unsqueeze(1)).sum(dim=2)                             # [B, N]
            rho_part  = rho - torch.bmm(Phi.float().mT, rho.unsqueeze(-1)).squeeze(-1) # [B, N]
            beta_part = torch.einsum("bd,bnd->bn", beta, x.float())                    # [B, N]
            grad_v    = (rho_part + beta_part).to(v.dtype)

        return (
            grad_Phi.to(Phi.dtype) if ctx.needs_input_grad[0] else None,
            grad_x if ctx.needs_input_grad[1] else None,
            grad_Y if ctx.needs_input_grad[2] else None,
            grad_v,
        )


###
### Public API
###
def stream_mix_add_small_nb(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused  out = Phi @ x + Y  (or projected variant) — small-NB kernel.

    Args:
        Phi: [B, N, N] transition matrix.
        x:   [B, N, D] stream state.
        Y:   [B, N, D] source term.
        v:   [B, N]    unit-norm projection direction, or None.

    Returns:
        out: [B, N, D]
    """
    return _StreamMixFn.apply(Phi, x, Y, v)
