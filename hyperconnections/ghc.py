import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


# Input Dimension: d_in = (n / m) * embed_dim
# n / m is the expansion ratio of the embedding dimension.
# when m = 1, we get the regular hyperconnections, where the input dimension is n * embed_dim.
# when m = n, we get the regular connections, where the input dimension is embed_dim.
# when n > m > 1, we get a generalized version of hyperconnections, where the input dimension is (n / m) * embed_dim.
class GeneralizedHyperConnections(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        input_dim: int,
        embed_dim: int,
        module: nn.Module,
        bias: bool = False,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.n = n
        self.m = m
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        assert embed_dim % m == 0, f"embed_dim ({embed_dim}) must be divisible by m ({m})"
        assert input_dim == int(
            (n / m) * embed_dim
        ), f"Input dimension must be (n / m) * embed_dim, but got {input_dim} and {(n / m) * embed_dim}"

        self.block_size = embed_dim // m  # the block size = d_in / n = embed_dim / m
        self.scaling_factor = math.sqrt(self.block_size)

        # write_out (B): [n, m]
        self.write_out = nn.Parameter(torch.empty(n, m))
        self.dynamic_scaling_write_out = nn.Parameter(torch.ones(n, m))
        self.dynamic_scaling_weight_write_out = nn.Linear(self.block_size, m, bias=bias)

        # read_in (Å): [n, m]
        self.read_in = nn.Parameter(torch.empty(n, m))
        self.dynamic_scaling_read_in = nn.Parameter(torch.ones(n, m))
        self.dynamic_scaling_weight_read_in = nn.Linear(self.block_size, m, bias=bias)

        # stream_mixing (Â): [n, n]
        self.stream_mixing = nn.Parameter(torch.empty(n, n))
        self.dynamic_scaling_stream_mixing = nn.Parameter(torch.ones(n, n))
        self.dynamic_scaling_weight_stream_mixing = nn.Linear(self.block_size, n, bias=bias)

        self.norm = nn.RMSNorm(self.block_size, elementwise_affine=elementwise_affine)
        self.module = module

        self.init_weights()

    def init_weights(self):
        # Static matrices
        write_out = torch.zeros(self.m, self.n)
        for j in range(self.n):
            write_out[j % self.m, j] = 1.0
        self.write_out.data.copy_(write_out.T)  # [n, m]

        read_in = torch.zeros(self.n, self.m)
        read_in[: self.m, : self.m] = torch.eye(self.m)
        self.read_in.data.copy_(read_in)  # [n, m]

        self.stream_mixing.data.copy_(torch.eye(self.n))  # [n, n]

        # Dynamic scaling weights start at zero so initial behavior matches static connections
        nn.init.zeros_(self.dynamic_scaling_weight_write_out.weight)
        if self.dynamic_scaling_weight_write_out.bias is not None:
            nn.init.zeros_(self.dynamic_scaling_weight_write_out.bias)
        nn.init.zeros_(self.dynamic_scaling_weight_read_in.weight)
        if self.dynamic_scaling_weight_read_in.bias is not None:
            nn.init.zeros_(self.dynamic_scaling_weight_read_in.bias)
        nn.init.zeros_(self.dynamic_scaling_weight_stream_mixing.weight)
        if self.dynamic_scaling_weight_stream_mixing.bias is not None:
            nn.init.zeros_(self.dynamic_scaling_weight_stream_mixing.bias)

    def compute_mixing_weights(self, x: torch.Tensor):
        # x: [B, n, block_size]
        x = self.norm(x)  # [B, n, block_size]

        write_out = (
            self.dynamic_scaling_write_out
            * F.tanh(self.dynamic_scaling_weight_write_out(x) / self.scaling_factor)
            + self.write_out
        )  # [B, n, m]

        read_in = (
            self.dynamic_scaling_read_in
            * F.tanh(self.dynamic_scaling_weight_read_in(x) / self.scaling_factor)
            + self.read_in
        ).transpose(1, 2)  # [B, n, m] -> [B, m, n]

        stream_mixing = (
            self.dynamic_scaling_stream_mixing
            * F.tanh(self.dynamic_scaling_weight_stream_mixing(x) / self.scaling_factor)
            + self.stream_mixing
        )  # [B, n, n]

        return write_out, read_in, stream_mixing

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, *, input_dim]
        leading = x.shape[:-1]
        x = x.reshape(-1, self.n, self.block_size)  # [B*, n, block_size]
        B = x.shape[0]

        write_out, read_in, stream_mixing = self.compute_mixing_weights(x)
        # [B*, n, m], [B*, m, n], [B*, n, n]

        # Read in from the over-width space to backbone width
        x_read_in = einsum(read_in, x, "b m n, b n d -> b m d")  # [B*, m, block_size]

        # Process through the backbone module
        out = self.module(x_read_in.reshape(*leading, self.embed_dim), **kwargs)

        # Write out from backbone width back to the over-width space
        out = out.reshape(B, self.m, self.block_size)  # [B*, m, block_size]
        out = einsum(write_out, out, "b n m, b m d -> b n d")  # [B*, n, block_size]

        # Mix within the over-width space and add residual
        x = einsum(stream_mixing, x, "b n1 n2, b n2 d -> b n1 d")  # [B*, n, block_size]
        out = out + x  # [B*, n, block_size]
        return out.unflatten(0, leading).flatten(-2)
