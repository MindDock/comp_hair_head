"""Unit tests for GaussianModel."""

import torch
import pytest
from comp_hair_head.gaussian.model import GaussianModel


class TestGaussianModel:
    def test_create_empty(self):
        model = GaussianModel(num_gaussians=0)
        assert model.num_gaussians == 0

    def test_create_with_count(self):
        model = GaussianModel(num_gaussians=100)
        assert model.num_gaussians == 100
        assert model.positions.shape == (100, 3)
        assert model.rotations.shape == (100, 4)
        assert model.scales.shape == (100, 3)
        assert model.opacities.shape == (100, 1)

    def test_properties_activated(self):
        model = GaussianModel(num_gaussians=10)
        # Scales should be positive (exp activation)
        assert (model.scales > 0).all()
        # Opacities should be in [0, 1] (sigmoid activation)
        assert (model.opacities >= 0).all() and (model.opacities <= 1).all()
        # Rotations should be normalized
        norms = model.rotations.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_clone(self):
        model = GaussianModel(num_gaussians=50)
        clone = model.clone()
        assert clone.num_gaussians == 50
        assert not (clone._positions is model._positions)
        assert torch.allclose(clone.positions, model.positions)

    def test_filter_by_mask(self):
        model = GaussianModel(num_gaussians=100)
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:30] = True
        filtered = model.filter_by_mask(mask)
        assert filtered.num_gaussians == 30

    def test_merge(self):
        m1 = GaussianModel(num_gaussians=20)
        m2 = GaussianModel(num_gaussians=30)
        merged = m1.merge(m2)
        assert merged.num_gaussians == 50

    def test_from_point_cloud(self):
        positions = torch.randn(500, 3)
        colors = torch.rand(500, 3)
        model = GaussianModel.from_point_cloud(positions, colors)
        assert model.num_gaussians == 500

    def test_ellipsoid_endpoints(self):
        model = GaussianModel(num_gaussians=10)
        endpoints = model.get_ellipsoid_endpoints()
        assert endpoints.shape == (10, 7, 3)
        # First point should be center (position)
        assert torch.allclose(endpoints[:, 0], model.positions, atol=1e-5)

    def test_covariance_3d(self):
        model = GaussianModel(num_gaussians=5)
        cov = model.covariance_3d
        assert cov.shape == (5, 3, 3)
        # Should be symmetric
        assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-5)

    def test_binding(self):
        model = GaussianModel(num_gaussians=10)
        assert (model.binding == -1).all()
        new_binding = torch.arange(10)
        model.binding = new_binding
        assert torch.equal(model.binding, new_binding)


class TestGaussianModelDevice:
    @pytest.mark.skipif(
        not (torch.cuda.is_available() or
             (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())),
        reason="No GPU available"
    )
    def test_gpu_creation(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        model = GaussianModel(num_gaussians=10, device=device)
        assert model.positions.device.type in ("cuda", "mps")
