import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


@dataclass
class ContinuousHCConfig:
    n_streams: int
    d_model: int
    dt: float = 1.0
    generator_type: Literal[
        "conservative",
        "diffusion_psd",
        "diagonal_dissipative",
        "conservative_dissipative",
    ] = "conservative_dissipative"
    protection_mode: Literal["none", "mean", "v"] = "none"
    eps: float = 1e-5
    learn_dt: bool = False


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
            "conservative_diag_diss",
            "conservative_psd_diss",
        ] = "conservative_psd_diss",
        projection: Literal["mean", "v", "none"] = "none",
        learn_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 1.0,
        bias: bool = False,
        elementwise_affine: bool = False,
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

        # Generator parameters — boolean flags drive which components are created
        conserv = generator_type in {
            "conservative",
            "conservative_diag_diss",
            "conservative_psd_diss",
        }
        psd_diss = generator_type in {"psd_diss", "conservative_psd_diss"}
        diag_diss = generator_type in {"diagonal_diss", "conservative_diag_diss"}

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

        # Projection Direction
        if projection == "mean":
            self.register_buffer("projection_dir", torch.ones(n) / math.sqrt(n))
        elif projection == "v":
            self.projection_dir = nn.Linear(input_dim, n, bias=True)
        elif projection == "none":
            self.projection_dir = None

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)
        self.module = module
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

        # Projections: zero so initial behaviour matches static biases
        for proj in (self.proj_read_in, self.proj_write_out):
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        # proj_v: zero weight + ones bias → normalise(ones) = 1/√n · 1 (mean direction)
        if self.projection == "v":
            nn.init.zeros_(self.projection_dir.weight)
            nn.init.ones_(self.projection_dir.bias)

    def compute_generator(self, x: torch.Tensor) -> torch.Tensor:
        """Return A of shape [B, n, n].

        Each component adds independently to A:
          - conservative:  skew-sym S = (M - M^T),  M = conserv_A + conv_pred(x)
          - psd_diss:      negative PSD K = -R R^T,  R = diss_A + diss_pred(x)
          - diag_diss:     negative diagonal -diag(d),  d = softplus(diss_diag + diss_pred(x))
        Dynamic deltas are zero-init so A starts from the static base alone.
        """
        B = x.shape[0]
        x_norm = self.norm(x.view(B, -1)).float()  # [B, input_dim], float32 for linear stability
        A = torch.zeros(B, self.n, self.n, device=x.device, dtype=x.dtype)

        if hasattr(self, "conserv_A"):
            M = self.conserv_A + self.conv_pred(x_norm).reshape(B, self.n, self.n)
            A = A + (M - M.transpose(-1, -2))  # skew-symmetric

        if hasattr(self, "diss_A"):
            R = self.diss_A + self.diss_pred(x_norm).reshape(B, self.n, self.n)
            A = A - R @ R.transpose(-1, -2)  # subtract PSD K

        if hasattr(self, "diss_diag"):
            d = F.softplus(self.diss_diag + self.diss_pred(x_norm))  # [B, n], positive
            A = A - torch.diag_embed(d)

        return A

    def compute_transition(self, x: torch.Tensor) -> torch.Tensor:
        """Return Phi = exp(dt * A), shape [B, n, n]."""
        dt = torch.clamp(self.log_dt.exp(), self.dt_min, self.dt_max)
        return torch.linalg.matrix_exp(dt * self.compute_generator(x))

    def compute_read_write_weights(self, x: torch.Tensor):
        """Compute dynamic read/write weights from the current stream state."""
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, input_dim]
        x_norm = self.norm(x_flat).float()  # float32 for linear layers under torch.compile

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)
        h_write_out = self.proj_write_out(x_norm).reshape(B, self.n, self.m)

        read_in = F.sigmoid(self.alpha_read_in * h_read_in + self.read_in).transpose(
            1, 2
        )  # [B, m, n]
        write_out = 2 * F.sigmoid(
            self.alpha_write_out * h_write_out + self.write_out
        )  # [B, n, m]

        return write_out, read_in
    
    def compute_projection(self, x: torch.Tensor):
        if self.projection == "mean":
            return self.projection_dir.unsqueeze(0)  # [1, n]
        elif self.projection == "v":
            B = x.shape[0]
            x_flat = x.view(B, -1)
            v = self.projection_dir(self.norm(x_flat).float())  # [B, n]
            return F.normalize(v, dim=-1)  # [B, n], unit norm
        else:
            return None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  # [B*, n, block_size]
        B = x.shape[0]

        write_out, read_in = self.compute_read_write_weights(x)
        write_out = write_out.to(x.dtype)
        read_in = read_in.to(x.dtype)

        ### Source term Y = H^post F(H^pre X)  (read → compute → write)
        # Read in from over-width space to backbone width
        x_read = einsum(read_in, x, "b m n, b n d -> b m d")  # [B*, m, block_size]

        # Process through the backbone module
        out = self.module(x_read.reshape(*leading, self.embed_dim), **kwargs)

        # Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)
        Y = einsum(write_out, out, "b n m, b m d -> b n d")  # [B*, n, block_size]

        ### Steam Mixing
        # Mixing: X_new_mix = Phi @ X  (or protected variant)
        transition_matrix = self.compute_transition(x)  # [B, n, n]
        transition_matrix = transition_matrix.to(x.dtype)

        # compute projection direction for projected mixing
        projection_dir = self.compute_projection(x) # [B, n] or None
        if projection_dir is not None:
            projection_dir = projection_dir.to(x.dtype)

        if projection_dir is None:
            x_mixed = einsum(transition_matrix, x, "b n1 n2, b n2 d -> b n1 d")  # [B*, n, block_size]
        else:
            proj_matrix = einsum(projection_dir, projection_dir, "b n1, b n2 -> b n1 n2")  # [b, n, n]
            orthogonal_proj = torch.eye(self.n, device=x.device) - proj_matrix  # [b, n, n]
            x_proj = einsum(proj_matrix, x, "b n1 n2, b n2 d -> b n1 d")  # [b, n, block_size]
            x_orth = einsum(orthogonal_proj, x, "b n1 n2, b n2 d -> b n1 d")  # [b, n, block_size]
            x_mixed = x_proj + einsum(transition_matrix, x_orth, "b n1 n2, b n2 d -> b n1 d")  # [B*, n, block_size]

        return (x_mixed + Y).unflatten(0, leading).flatten(-2)
