"""
ops/__init__.py — public entry point for stream mixing.

Dispatches to one of two Triton implementations based on a memory-footprint
heuristic:

  big_NB kernel  (stream_mix_big_nb.py)
      Uses tl.dot (tensor cores) and a (B, cdiv(D, BLOCK_D)) grid.
      Each CTA loads Phi[b] and x[b, :, d_tile] once for all N rows.
      Wins when the x footprint exceeds L2 capacity so cross-CTA reuse
      can no longer be served from cache.
      Conditions: N >= 16, N % 16 == 0, B*N*D*elem_bytes > L2_BYTES.

  generic kernel  (stream_mix.py)
      One CTA per output row (b, n1).  N_STREAMS is a constexpr → the
      inner loop is fully unrolled, Phi scalars hit L1, x slices stay in
      the 40 MB A100 L2.  Faster for small N or small B.
"""
from __future__ import annotations
import torch

from hyperconnections.ops.stream_mix import stream_mix_add_small_n
from hyperconnections.ops.stream_mix_big_nb import stream_mix_add_big_nb

### A100 L2 capacity in bytes.  Adjust if targeting a different GPU.
_SM80_L2_BYTES = 40 * 1024 * 1024


def _use_big_nb(x: torch.Tensor) -> bool:
    """Return True when the big-NB kernel is expected to be faster."""
    B, N, D = x.shape
    if N < 16 or N % 16 != 0:
        return False
    elem_bytes = x.element_size()
    ### "Safety" buffer, check if more than 75% of L2 cache would be occupied
    return B * N * D * elem_bytes > _SM80_L2_BYTES * 0.75


def stream_mix_add(
    Phi: torch.Tensor,
    x: torch.Tensor,
    Y: torch.Tensor,
    v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused  out = Phi @ x + Y  (or projected variant).

    Args:
        Phi: [B, N, N] transition matrix.
        x:   [B, N, D] stream state.
        Y:   [B, N, D] source term.
        v:   [B, N]    unit-norm projection direction, or None.

    Returns:
        out: [B, N, D]

    Dispatches to the big-NB kernel (tl.dot, tensor cores) when
      N >= 16  and  B*N*D*elem_bytes > 40 MB,
    otherwise falls back to the generic kernel (static-range unrolled,
    scalar Phi loads, L2-cached x).
    """
    if not x.is_cuda:
        raise RuntimeError("stream_mix_add requires CUDA tensors")
    B, N, D = x.shape
    assert Phi.shape == (B, N, N)
    assert Y.shape   == (B, N, D)
    if v is not None:
        assert v.shape == (B, N)

    if _use_big_nb(x):
        return stream_mix_add_big_nb(Phi, x, Y, v)
    return stream_mix_add_small_n(Phi, x, Y, v)