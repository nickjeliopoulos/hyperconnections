import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim * num_heads == dim
        ), "Embedding dimension must be divisible by number of heads"
        self.scale = self.head_dim**-0.5

        self.qk = nn.Linear(dim, dim * 2, bias=bias)

        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x):
        B, N, C = x.shape

        qk = rearrange(
            self.qk(x),
            "b n (two h d) -> two b h n d",
            two=2,
            h=self.num_heads,
            d=self.head_dim,
        )

        q, k = qk
        q = self.q_norm(q)
        k = self.k_norm(k)

        x = rearrange(x, "b n (h d) -> b h n d", h=self.num_heads, d=self.head_dim)
        out = F.scaled_dot_product_attention(q, k, x)
        return rearrange(out, "b h n d -> b n (h d)")


# Input Dimension: d_in = (n / m) * embed_dim
# n / m is the expansion ratio of the embedding dimension.
# when m = 1, we get the regular hyperconnections, where the input dimension is n * embed_dim.
# when m = n, we get the regular connections, where the input dimension is embed_dim.
# when n > m > 1, we get a generalized version of hyperconnections, where the input dimension is (n / m) * embed_dim.
class AttentionHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        num_heads: int = 8,
        bias: bool = False,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        assert (
            embed_dim % m == 0
        ), f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        assert input_dim == int(
            (n / m) * embed_dim
        ), f"Input dimension must be (n / m) * embed_dim, but got {input_dim} and {(n / m) * embed_dim}"

        self.block_size = embed_dim // m  # block size = embed_dim / m = input_dim / n

        assert (
            self.block_size % num_heads == 0
        ), f"block_size ({self.block_size}) must be divisible by num_heads ({num_heads})"

        # read_in (H^pre): [n, m]
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.alpha_read_in = nn.Parameter(torch.empty(1))

        # write_out (H^post): [n, m]
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.alpha_write_out = nn.Parameter(torch.empty(1))

        # Projection Matrices
        self.proj_read_in = nn.Linear(input_dim, n * m, bias=bias)
        self.proj_write_out = nn.Linear(input_dim, n * m, bias=bias)

        # Stream mixing via attention (operates on block_size channels, N=n streams)
        self.attn = Attention(self.block_size, num_heads=num_heads, bias=bias)

        self.norm = nn.RMSNorm(input_dim, elementwise_affine=elementwise_affine)

        self.module = module
        self.init_weights()

    def init_weights(self):
        # read_in: initialise so σ(read_in) = 1/n (uniform read across all streams).
        # σ(b) = 1/n  →  b = logit(1/n) = log(1 / (n-1))  for n > 1.
        logit_1_over_n = math.log(1.0 / (self.n - 1)) if self.n > 1 else 10.0
        nn.init.constant_(self.read_in, logit_1_over_n)

        # write_out: initialise so 2·σ(write_out) = 1.
        # 2·σ(0) = 1  →  write_out = 0.
        nn.init.zeros_(self.write_out)

        # Alpha gating factors: 0.01 per the mHC paper (Table 5).
        nn.init.constant_(self.alpha_read_in, 0.01)
        nn.init.constant_(self.alpha_write_out, 0.01)

        # Projection weights: zero so the dynamic component starts negligible and
        # the initial behaviour is entirely determined by the static biases above.
        for proj in (self.proj_read_in, self.proj_write_out):
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def compute_mixing_weights(self, x: torch.Tensor):
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, input_dim] — flatten all streams
        x_norm = self.norm(x_flat)

        h_read_in = self.proj_read_in(x_norm).reshape(B, self.n, self.m)  # [B, n, m]
        h_write_out = self.proj_write_out(x_norm).reshape(
            B, self.n, self.m
        )  # [B, n, m]

        # Scale by learnable alpha and add static bias
        h_read_in = self.alpha_read_in * h_read_in + self.read_in
        h_write_out = self.alpha_write_out * h_write_out + self.write_out

        # Apply manifold constraints
        read_in = F.sigmoid(h_read_in).transpose(1, 2)
        write_out = 2 * F.sigmoid(h_write_out)

        return (
            write_out,
            read_in,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  # [B*, n, block_size]
        B = x.shape[0]

        write_out, read_in = self.compute_mixing_weights(x)
        # [B*, n, m], [B*, m, n]

        # Read in from the over-width space to backbone width
        x_read_in = einsum(read_in, x, "b m n, b n d -> b m d")  # [B*, m, block_size]

        # Process through the backbone module
        out = self.module(x_read_in.reshape(*leading, self.embed_dim), **kwargs)

        # Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  # [B*, m, block_size]
        out = einsum(write_out, out, "b n m, b m d -> b n d")  # [B*, n, block_size]

        # x = einsum(stream_mixing, x, "b n1 n2, b n2 d -> b n1 d")  # [B*, n, block_size]
        # Mix within the over-width space and add residual)
        x = self.attn(x) # [B*, n, block_size]

        out = out + x  # [B*, n, block_size]
        return out.unflatten(0, leading).flatten(-2)
