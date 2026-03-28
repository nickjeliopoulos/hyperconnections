"""
Tests for ContinuousGenHyperConnections (cghc.py).

Configurations used throughout: (n, m, embed_dim)
  - (2, 1, 8)  : hyperconnections (m=1)
  - (4, 2, 8)  : generalized, r=2
  - (4, 4, 8)  : regular connections (m=n)
  - (6, 3, 12) : generalized, r=3
"""

import math

import pytest
import torch
import torch.nn as nn

from hyperconnections.cghc import ContinuousGenHyperConnections
from tests.conftest import IdentityModule, ZeroModule


CONFIGS = [
    (2, 1, 8),
    (4, 2, 8),
    (4, 4, 8),
    (6, 3, 12),
]

ALL_GENERATOR_TYPES = [
    "conservative",
    "psd_diss",
    "diagonal_diss",
    "laplacian",
    "conservative_diag_diss",
    "conservative_psd_diss",
    "conservative_laplacian",
]


def make_cghc(
    n: int,
    m: int,
    embed_dim: int,
    module: nn.Module | None = None,
    **kwargs,
) -> ContinuousGenHyperConnections:
    input_dim = (n * embed_dim) // m
    if module is None:
        module = IdentityModule()
    return ContinuousGenHyperConnections(
        n=n, m=m, input_dim=input_dim, embed_dim=embed_dim, module=module, **kwargs
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_parameter_shapes(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.read_in.shape == (n, m)
        assert cghc.write_out.shape == (n, m)
        assert cghc.alpha_read_in.shape == (1,)
        assert cghc.alpha_write_out.shape == (1,)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_block_size(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.block_size == embed_dim // m

    def test_invalid_embed_dim_not_divisible_by_m_raises(self):
        with pytest.raises(AssertionError):
            make_cghc(n=6, m=4, embed_dim=10)

    def test_invalid_input_dim_raises(self):
        with pytest.raises(AssertionError):
            ContinuousGenHyperConnections(
                n=4, m=2, input_dim=99, embed_dim=8, module=IdentityModule()
            )

    # Regression test for the conservative_laplacian bug fix
    def test_conservative_laplacian_creates_conserv_params(self):
        cghc = make_cghc(4, 2, 8, generator_type="conservative_laplacian")
        assert hasattr(cghc, "conserv_A"), "conserv_A must exist for conservative_laplacian"
        assert hasattr(cghc, "conv_pred"), "conv_pred must exist for conservative_laplacian"
        assert hasattr(cghc, "laplacian_A"), "laplacian_A must exist for conservative_laplacian"

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_generator_type_constructs_without_error(self, generator_type):
        make_cghc(4, 2, 8, generator_type=generator_type)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_init_value(self, n, m, embed_dim):
        """sigmoid(read_in bias) should equal 1/n at init."""
        cghc = make_cghc(n, m, embed_dim)
        expected = 1.0 / n
        actual = torch.sigmoid(cghc.read_in)
        assert torch.allclose(actual, torch.full_like(actual, expected), atol=1e-6)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_init_value(self, n, m, embed_dim):
        """2 * sigmoid(write_out bias) should equal 1 at init (bias=0)."""
        cghc = make_cghc(n, m, embed_dim)
        actual = 2 * torch.sigmoid(cghc.write_out)
        assert torch.allclose(actual, torch.ones_like(actual), atol=1e-6)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_alpha_negligible_at_init(self, n, m, embed_dim):
        """Alpha gating should be 0.01 so dynamic component starts negligible."""
        cghc = make_cghc(n, m, embed_dim)
        assert cghc.alpha_read_in.item() == pytest.approx(0.01)
        assert cghc.alpha_write_out.item() == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Generator structural properties
# ---------------------------------------------------------------------------


class TestGeneratorStructure:
    """After perturbing parameters, verify each generator type maintains its structural guarantee."""

    def _perturb(self, cghc: ContinuousGenHyperConnections):
        """Add random noise to generator params so we don't just test at zero."""
        with torch.no_grad():
            if hasattr(cghc, "conserv_A"):
                cghc.conserv_A.add_(torch.randn_like(cghc.conserv_A) * 0.1)
            if hasattr(cghc, "diss_A"):
                cghc.diss_A.add_(torch.randn_like(cghc.diss_A) * 0.1)
            if hasattr(cghc, "diss_diag"):
                cghc.diss_diag.add_(torch.randn_like(cghc.diss_diag) * 0.5)
            if hasattr(cghc, "laplacian_A"):
                cghc.laplacian_A.add_(torch.randn_like(cghc.laplacian_A) * 0.1)

    def test_conservative_is_skew_symmetric(self):
        cghc = make_cghc(4, 2, 8, generator_type="conservative")
        self._perturb(cghc)
        x = torch.randn(3, cghc.n, cghc.block_size)
        A = cghc.compute_generator(x)
        assert torch.allclose(A + A.transpose(-1, -2), torch.zeros_like(A), atol=1e-5)

    def test_psd_diss_is_negative_semidefinite(self):
        cghc = make_cghc(4, 2, 8, generator_type="psd_diss")
        self._perturb(cghc)
        x = torch.randn(3, cghc.n, cghc.block_size)
        A = cghc.compute_generator(x)
        eigvals = torch.linalg.eigvalsh(A)
        assert (eigvals <= 1e-4).all(), "psd_diss generator must be NSD"

    def test_diagonal_diss_has_nonpositive_diagonal_and_zero_offdiag(self):
        cghc = make_cghc(4, 2, 8, generator_type="diagonal_diss")
        self._perturb(cghc)
        x = torch.randn(3, cghc.n, cghc.block_size)
        A = cghc.compute_generator(x)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)
        off_diag_mask = ~torch.eye(cghc.n, dtype=torch.bool)
        assert (diag <= 0).all(), "diagonal entries must be non-positive"
        assert torch.allclose(
            A[:, off_diag_mask], torch.zeros(3, cghc.n * cghc.n - cghc.n), atol=1e-6
        ), "off-diagonal entries must be zero"

    def test_laplacian_A_is_zero_at_init(self):
        """At init (all params zero), adjacency = softplus(0) - log(2) = 0, so Laplacian = 0 and A = 0."""
        cghc = make_cghc(4, 2, 8, generator_type="laplacian")
        x = torch.randn(3, cghc.n, cghc.block_size)
        A = cghc.compute_generator(x)
        assert torch.allclose(A, torch.zeros_like(A), atol=1e-6), "laplacian generator must be zero at init"

    def test_conservative_laplacian_is_neither_symmetric_nor_skew(self):
        """Combined generator should have both conservative and dissipative parts."""
        cghc = make_cghc(4, 2, 8, generator_type="conservative_laplacian")
        self._perturb(cghc)
        x = torch.randn(3, cghc.n, cghc.block_size)
        A = cghc.compute_generator(x)
        # Not purely skew-symmetric (laplacian adds symmetric negative part)
        assert not torch.allclose(A + A.transpose(-1, -2), torch.zeros_like(A), atol=1e-3)


# ---------------------------------------------------------------------------
# Transition matrix at initialization
# ---------------------------------------------------------------------------


class TestInitialTransition:
    """At initialization all dynamic deltas are zero, so A starts from its static base.
    For all types except laplacian the static A = 0, giving Phi = I.
    After the softplus shift fix, the laplacian also starts with A = 0, giving Phi = I.
    """

    # diagonal_diss is intentionally initialized with diss_diag=-5 so softplus(-5)≈0.007 → Phi≈I,
    # not exactly I. All other types produce A=0 exactly at init → Phi=I exactly.
    EXACT_IDENTITY_TYPES = [
        "conservative",
        "psd_diss",
        "laplacian",
        "conservative_psd_diss",
        "conservative_laplacian",
    ]
    NEAR_IDENTITY_TYPES = ["diagonal_diss", "conservative_diag_diss"]

    @pytest.mark.parametrize("generator_type", EXACT_IDENTITY_TYPES)
    def test_transition_is_identity_at_init(self, generator_type):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, n, embed_dim // m)
        Phi = cghc.compute_transition(x)
        I = torch.eye(n).unsqueeze(0).expand_as(Phi)
        assert torch.allclose(Phi, I, atol=1e-5), (
            f"Transition should be I at init for generator_type='{generator_type}'"
        )

    @pytest.mark.parametrize("generator_type", NEAR_IDENTITY_TYPES)
    def test_transition_is_near_identity_at_init(self, generator_type):
        """diagonal_diss starts with softplus(-5)≈0.007 dissipation → Phi≈I but not exact."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, n, embed_dim // m)
        Phi = cghc.compute_transition(x)
        I = torch.eye(n).unsqueeze(0).expand_as(Phi)
        assert torch.allclose(Phi, I, atol=0.01), (
            f"Transition should be near-I at init for generator_type='{generator_type}'"
        )

    @pytest.mark.parametrize("generator_type", ["conservative", "conservative_psd_diss", "conservative_laplacian"])
    def test_transition_is_orthogonal_after_perturbation(self, generator_type):
        """exp(skew-sym) is orthogonal; mixed types with conservative component should satisfy Phi Phi^T ≈ I only when purely conservative."""
        if generator_type != "conservative":
            pytest.skip("orthogonality only holds for purely conservative generator")
        cghc = make_cghc(4, 2, 8, generator_type=generator_type)
        with torch.no_grad():
            cghc.conserv_A.add_(torch.randn_like(cghc.conserv_A) * 0.5)
        x = torch.randn(3, 4, 4)
        Phi = cghc.compute_transition(x)
        I = torch.eye(4).unsqueeze(0).expand_as(Phi)
        assert torch.allclose(Phi @ Phi.transpose(-1, -2), I, atol=1e-4)


# ---------------------------------------------------------------------------
# Read/write weights
# ---------------------------------------------------------------------------


class TestReadWriteWeights:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shapes(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        write_out, read_in = cghc.compute_read_write_weights(x)
        assert write_out.shape == (B, n, m)
        assert read_in.shape == (B, m, n)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_read_in_static_at_init(self, n, m, embed_dim):
        """With alpha ≈ 0.01, dynamic component is negligible; read_in ≈ sigmoid(bias) = 1/n."""
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        _, read_in = cghc.compute_read_write_weights(x)
        expected = 1.0 / n
        # tolerance loosened slightly because alpha=0.01 gives a small but nonzero dynamic offset
        assert torch.allclose(read_in, torch.full_like(read_in, expected), atol=0.02)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_write_out_static_at_init(self, n, m, embed_dim):
        """With alpha ≈ 0.01, write_out ≈ 2 * sigmoid(0) = 1."""
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        x = torch.randn(B, n, embed_dim // m)
        write_out, _ = cghc.compute_read_write_weights(x)
        assert torch.allclose(write_out, torch.ones_like(write_out), atol=0.02)


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_output_shape_matches_input(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        B = 3
        input_dim = (n * embed_dim) // m
        x = torch.randn(B, input_dim)
        assert cghc(x).shape == (B, input_dim)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_batch_size_one(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        input_dim = (n * embed_dim) // m
        x = torch.randn(1, input_dim)
        assert cghc(x).shape == (1, input_dim)

    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_extra_leading_dims(self, n, m, embed_dim):
        """Input [B, S, input_dim] should produce output of the same shape."""
        cghc = make_cghc(n, m, embed_dim)
        B, S, input_dim = 2, 7, (n * embed_dim) // m
        x = torch.randn(B, S, input_dim)
        assert cghc(x).shape == (B, S, input_dim)

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_all_generator_types_produce_correct_shape(self, generator_type):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, (n * embed_dim) // m)
        assert cghc(x).shape == x.shape

    @pytest.mark.parametrize("projection", ["mean", "v", "none"])
    def test_all_projection_modes_produce_correct_shape(self, projection):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, projection=projection)
        x = torch.randn(3, (n * embed_dim) // m)
        assert cghc(x).shape == x.shape


# ---------------------------------------------------------------------------
# Forward behavior
# ---------------------------------------------------------------------------


class TestForwardBehavior:
    EXACT_IDENTITY_TYPES = [
        "conservative",
        "psd_diss",
        "laplacian",
        "conservative_psd_diss",
        "conservative_laplacian",
    ]

    @pytest.mark.parametrize("generator_type", EXACT_IDENTITY_TYPES)
    def test_zero_module_output_equals_input_at_init(self, generator_type):
        """
        At init, Phi = I for these types (A=0 exactly).
        ZeroModule gives Y = 0, so output = Phi @ x + 0 = x.
        """
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type, module=ZeroModule())
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=1e-5)

    @pytest.mark.parametrize("generator_type", ["diagonal_diss", "conservative_diag_diss"])
    def test_zero_module_output_near_input_at_init(self, generator_type):
        """diagonal_diss has Phi≈I (not exact) so output≈x with ZeroModule."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type, module=ZeroModule())
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=0.02)

    def test_module_receives_correct_embed_dim(self):
        """Inner module receives [B, embed_dim] when input is [B, input_dim]."""
        received = []

        class ShapeCapture(nn.Module):
            def forward(self, x, **kwargs):
                received.append(x.shape)
                return torch.zeros_like(x)

        n, m, embed_dim = 4, 2, 8
        B = 5
        cghc = make_cghc(n, m, embed_dim, module=ShapeCapture())
        cghc(torch.randn(B, (n * embed_dim) // m))
        assert len(received) == 1
        assert received[0] == (B, embed_dim)

    def test_module_receives_correct_shape_with_leading_dims(self):
        """Inner module receives [B, S, embed_dim] when input is [B, S, input_dim]."""
        received = []

        class ShapeCapture(nn.Module):
            def forward(self, x, **kwargs):
                received.append(x.shape)
                return torch.zeros_like(x)

        n, m, embed_dim = 4, 2, 8
        B, S = 2, 7
        cghc = make_cghc(n, m, embed_dim, module=ShapeCapture())
        cghc(torch.randn(B, S, (n * embed_dim) // m))
        assert len(received) == 1
        assert received[0] == (B, S, embed_dim)

    @pytest.mark.parametrize("projection", ["mean", "v", "none"])
    def test_zero_module_with_projections_at_init(self, projection):
        """Projection doesn't change the identity-at-init result with ZeroModule."""
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(
            n, m, embed_dim, projection=projection, module=ZeroModule(), generator_type="conservative_psd_diss"
        )
        x = torch.randn(3, (n * embed_dim) // m)
        assert torch.allclose(cghc(x), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize("n,m,embed_dim", CONFIGS)
    def test_backward_produces_gradients(self, n, m, embed_dim):
        cghc = make_cghc(n, m, embed_dim)
        x = torch.randn(3, (n * embed_dim) // m, requires_grad=True)
        cghc(x).sum().backward()
        assert x.grad is not None
        assert cghc.read_in.grad is not None
        assert cghc.write_out.grad is not None

    @pytest.mark.parametrize("generator_type", ALL_GENERATOR_TYPES)
    def test_backward_through_all_generator_types(self, generator_type):
        n, m, embed_dim = 4, 2, 8
        cghc = make_cghc(n, m, embed_dim, generator_type=generator_type)
        x = torch.randn(3, (n * embed_dim) // m, requires_grad=True)
        cghc(x).sum().backward()
        assert x.grad is not None
