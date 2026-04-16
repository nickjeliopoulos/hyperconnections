import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from hyperconnections.ops import stream_mix_add


class ContinuousGenHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        dt: float = 1.0,
        generator_type: Literal[
            "conservative",
            "psd_diss",
            "diagonal_diss",
            "laplacian",
            "conservative_diag_diss",
            "conservative_psd_diss",
            "conservative_laplacian",
        ] = "conservative_psd_diss",
        projection: Literal["mean", "v", "none"] = "none",
        learn_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        bias: bool = False,
        elementwise_affine: bool = False,
        use_triton: bool = True,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.generator_type = generator_type
        self.projection = projection

        assert (
            embed_dim % m == 0
        ), f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        assert input_dim == int(
            (n / m) * embed_dim
        ), f"input_dim must be (n/m)*embed_dim, got {input_dim} vs {int((n/m)*embed_dim)}"

        self.block_size = embed_dim // m

        # Read/write parameters following mHC convention
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(input_dim, n * m, bias=bias)

        # dt parameters
        assert dt > 0, "Initial dt must be positive"
        self.dt_min = dt_min
        self.dt_max = dt_max
        log_dt_init = math.log(dt)
        self.log_dt = nn.Parameter(torch.tensor(log_dt_init), requires_grad=learn_dt)
        self.dt_proj = nn.Linear(input_dim, 1, bias=True)

        # Generator parameters — boolean flags drive which components are created
        conserv = generator_type in {
            "conservative",
            "conservative_diag_diss",
            "conservative_psd_diss",
            "conservative_laplacian",
        }
        psd_diss = generator_type in {"psd_diss", "conservative_psd_diss"}
        diag_diss = generator_type in {"diagonal_diss", "conservative_diag_diss"}
        laplacian = generator_type in {"laplacian", "conservative_laplacian"}

        if conserv:
            self.conserv_A = nn.Parameter(torch.eye(n, n))
            self.conv_pred = nn.Linear(input_dim, n * n, bias=True)
        if psd_diss:
            self.diss_A = nn.Parameter(torch.zeros(n, n))
            self.diss_pred = nn.Linear(input_dim, n * n, bias=True)
        if diag_diss:
            # Diagonal dissipation: store only the diagonal entries for efficiency
            # Initialise so softplus(diss_diag) ≈ 0.007 → Phi ≈ I at start
            self.diss_diag = nn.Parameter(torch.full((n,), -5.0))
            self.diss_pred = nn.Linear(input_dim, n, bias=True)
        if laplacian:
            self.laplacian_A = nn.Parameter(torch.zeros(n, n))
            self.laplacian_q = nn.Linear(self.block_size, self.block_size, bias=True)
            self.laplacian_k = nn.Linear(self.block_size, self.block_size, bias=True)
            self.laplacian_scale = self.block_size**-0.5
            self.norm_lap = nn.RMSNorm(self.block_size, elementwise_affine=True)

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.projection_dir = nn.Linear(input_dim, n, bias=True)
        elif projection == "none":
            self.projection_dir = None

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
        self._stream_mix = (
            self._stream_mix_triton if use_triton else self._stream_mix_eager
        )
        self.init_weights()


    def init_weights(self):
        # read_in: σ(bias) = 1/n  →  bias = log(1/(n-1))
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)
        # write_out: 2·σ(0) = 1
        nn.init.zeros_(self.write_out)
        # Alpha gating: 0.01 so dynamic component starts negligible
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        # Generator Dynamic Parameters
        if hasattr(self, "conserv_A"):
            nn.init.zeros_(self.conv_pred.weight)
            nn.init.zeros_(self.conv_pred.bias)

        if hasattr(self, "diss_A"):
            nn.init.zeros_(self.diss_pred.weight)
            nn.init.zeros_(self.diss_pred.bias)

        if hasattr(self, "diss_diag"):
            nn.init.zeros_(self.diss_pred.weight)
            nn.init.zeros_(self.diss_pred.bias)
        
        if hasattr(self, "laplacian_A"):
            nn.init.zeros_(self.laplacian_A)
            nn.init.zeros_(self.laplacian_q.weight)
            nn.init.zeros_(self.laplacian_q.bias)
            nn.init.zeros_(self.laplacian_k.weight)
            nn.init.zeros_(self.laplacian_k.bias)

        # dt_proj: zero so initial dt comes entirely from log_dt
        nn.init.zeros_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)

        # Projections: zero so initial behaviour matches static biases
        for proj in (self.proj_read_in, self.proj_write_out):
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        # proj_v: zero weight + ones bias → normalise(ones) = 1/√n · 1 (mean direction)
        if self.projection == "v":
            nn.init.zeros_(self.projection_dir.weight)
            nn.init.ones_(self.projection_dir.bias)

    def compute_generator(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (A, dt) where A has shape [B, n, n] and dt has shape [B].

        Each component adds independently to A:
          - conservative:  skew-sym S = (M - M^T),  M = conserv_A + conv_pred(x)
          - psd_diss:      negative PSD K = -R R^T,  R = diss_A + diss_pred(x)
          - diag_diss:     negative diagonal -diag(d),  d = softplus(diss_diag + diss_pred(x))
        Dynamic deltas are zero-init so A starts from the static base alone.
        dt = exp(log_dt) + softplus(dt_proj(x)), clamped to [dt_min, dt_max].
        """
        B = x.shape[0]
        if hasattr(self, "laplacian_A"):
            x_lap_norm = self.norm_lap(x)
        x_norm = self.norm(x.view(B, -1))  # [B, input_dim]
        A = torch.zeros(
            B, self.n, self.n, device=x.device, dtype=x.dtype
        )  # match input dtype

        if hasattr(self, "conserv_A"):
            M = self.conserv_A + self.conv_pred(x_norm).reshape(B, self.n, self.n)
            A = A + (M - M.transpose(-1, -2))  # skew-symmetric

        if hasattr(self, "diss_A"):
            R = self.diss_A + self.diss_pred(x_norm).reshape(B, self.n, self.n)
            A = A - R @ R.transpose(-1, -2)  # subtract PSD K

        if hasattr(self, "diss_diag"):
            d = F.softplus(self.diss_diag + self.diss_pred(x_norm))  # [B, n], positive
            A = A - torch.diag_embed(d)

        if hasattr(self, "laplacian_A"):
            score_bias = self.laplacian_A
            lap_q = self.laplacian_q(x_lap_norm) # [B, n, block_size]
            lap_k = self.laplacian_k(x_lap_norm) # [B, n, block_size]
            scores = lap_q @ lap_k.transpose(-1, -2) * self.laplacian_scale
            scores = score_bias + scores
            scores = 0.5 * (scores + scores.transpose(-1, -2)) # symmetrize
            adjacency = F.softplus(scores) - math.log(2)  # shift so zero scores → zero adjacency
            adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency, dim1=-2, dim2=-1))
            degree = torch.diag_embed(adjacency.sum(dim=-1))
            laplacian = degree - adjacency
            A = A - laplacian

        dt = self.log_dt.exp() + F.softplus(self.dt_proj(x_norm).squeeze(-1))  # [B]
        dt = torch.clamp(dt, self.dt_min, self.dt_max)  # [B]
        return A, dt


    def compute_transition(self, x: torch.Tensor) -> torch.Tensor:
        """Return Phi = exp(dt * A), shape [B, n, n]."""
        A, dt = self.compute_generator(x)
        return torch.linalg.matrix_exp((dt[:, None, None] * A).float()).to(x.dtype)


    def compute_read_write_weights(self, x: torch.Tensor):
        """Compute dynamic read/write weights from the current stream state."""
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, input_dim]
        x_norm = self.norm(x_flat)  # [B, input_dim]

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm).reshape(B, self.n, self.m)

        read_in = torch.sigmoid(self.alpha_read_in * h_read_in + self.read_in).transpose(
            1, 2
        )  # [B, m, n]
        write_out = 2 * torch.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

        return write_out, read_in

    def compute_projection(self, x: torch.Tensor):
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n]
        elif self.projection == "v":
            B = x.shape[0]
            x_flat = x.view(B, -1)
            v = self.projection_dir(self.norm(x_flat))  # [B, n]
            return F.normalize(v, dim=-1)  # [B, n], unit norm
        else:
            return None


    def _stream_mix_triton(
        self,
        x: torch.Tensor,
        transition_matrix: torch.Tensor,
        Y: torch.Tensor,
        projection_dir: torch.Tensor | None,
    ) -> torch.Tensor:
        if projection_dir is not None:
            projection_dir = projection_dir.expand(x.shape[0], -1) ### [1, N] ("mean" mode) --> [B, N]
        return stream_mix_add(transition_matrix, x, Y, projection_dir)


    def _stream_mix_eager(
        self,
        x: torch.Tensor,
        transition_matrix: torch.Tensor,
        Y: torch.Tensor,
        projection_dir: torch.Tensor | None,
    ) -> torch.Tensor:
        if projection_dir is None:
            x_mixed = einsum(
                transition_matrix, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [B*, n, block_size]
        else:
            proj_matrix = einsum(
                projection_dir, projection_dir, "b n1, b n2 -> b n1 n2"
            )  # [b, n, n]
            orthogonal_proj = (
                torch.eye(self.n, device=x.device, dtype=x.dtype) - proj_matrix
            )  # [b, n, n]
            x_proj = einsum(
                proj_matrix, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [b, n, block_size]
            x_orth = einsum(
                orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d"
            )  # [b, n, block_size]
            x_mixed = x_proj + einsum(
                transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
            )  # [B*, n, block_size]
        return (x_mixed + Y)


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        ### x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size) ### [B*, n, block_size]
        B = x.shape[0]

        write_out, read_in = self.compute_read_write_weights(x)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        ### Read in from over-width space to backbone width
        x_read = einsum(read_in, x, "b m n, b n d -> b m d") ### [B*, m, block_size]

        ### Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        ### Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)
        Y = einsum(write_out, out, "b n m, b m d -> b n d") ### [B*, n, block_size]

        ### Steam Mixing
        ### Mixing: X_new_mix = Phi @ X  (or protected variant)
        transition_matrix = self.compute_transition(x) ### [B, n, n]

        ### compute projection direction for projected mixing
        projection_dir = self.compute_projection(x) ### [B, n] or None

        # if projection_dir is None:
        #     x_mixed = einsum(
        #         transition_matrix, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [B*, n, block_size]
        # else:
        #     proj_matrix = einsum(
        #         projection_dir, projection_dir, "b n1, b n2 -> b n1 n2"
        #     )  # [b, n, n]
        #     orthogonal_proj = (
        #         torch.eye(self.n, device=x.device, dtype=x.dtype) - proj_matrix
        #     )  # [b, n, n]
        #     x_proj = einsum(
        #         proj_matrix, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [b, n, block_size]
        #     x_orth = einsum(
        #         orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [b, n, block_size]
        #     x_mixed = x_proj + einsum(
        #         transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d"
        #     )  # [B*, n, block_size]
        # return (x_mixed + Y).unflatten(0, leading).flatten(-2)

        return self._stream_mix(
            x=x,
            transition_matrix=transition_matrix,
            Y=Y,
            projection_dir=projection_dir,
        ).unflatten(0, leading).flatten(-2)
