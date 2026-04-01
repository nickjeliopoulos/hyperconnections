"""
Public interface for fused stream mixing.

Computes  out = Phi @ x + Y                               [no-proj]
or        out = Phi @ x + (v - Phi@v)(v^T x) + Y         [proj]

Dispatches to one of two Triton implementations based on a memory-footprint
heuristic:

  small-NB kernel  (stream_mix_small_nb.py)
      Grid: (B*N, cdiv(D, BLOCK_D)) — one CTA per output row.
      N_STREAMS is constexpr, inner loop fully unrolled; Phi scalars hit L1,
      x slices stay resident in the A100's 40 MB L2.
      Best when N is small or the x footprint fits comfortably in L2.

  big-NB kernel  (stream_mix_big_nb.py)
      Grid: (B, cdiv(D, BLOCK_D)) — one CTA per (b, d_tile), all N rows.
      Uses tl.dot (tensor cores); loads Phi[b] and x[b,:,d_tile] once per CTA.
      Best when N >= 16 and the x footprint exceeds ~75% of L2 capacity.

Threshold:  N >= 16  and  N % 16 == 0  and  B*N*D*elem_bytes > 0.75 * L2_BYTES
  _SM80_L2_BYTES = 40 MB  (A100; adjust for other GPUs)
"""

from __future__ import annotations

import torch

from .stream_mix_small_nb import stream_mix_add_small_nb as _small_nb
from .stream_mix_big_nb   import stream_mix_add_big_nb   as _big_nb

# A100 L2 capacity in bytes.  Adjust if targeting a different GPU.
_SM80_L2_BYTES = 40 * 1024 * 1024


def _use_big_nb(x: torch.Tensor) -> bool:
    B, N, D = x.shape
    if N < 16 or N % 16 != 0:
        return False
    # "Safety" buffer: switch kernels when x occupies > 75% of L2
    return B * N * D * x.element_size() > _SM80_L2_BYTES * 0.75


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
    """
    if not x.is_cuda:
        raise RuntimeError("stream_mix_add requires CUDA tensors")
    B, N, D = x.shape
    assert Phi.shape == (B, N, N)
    assert Y.shape   == (B, N, D)
    if v is not None:
        assert v.shape == (B, N)

    if _use_big_nb(x):
        return _big_nb(Phi, x, Y, v)
    return _small_nb(Phi, x, Y, v)
