"""3D Gaussian model: data structure for Gaussian primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.geometry import quaternion_to_matrix, quaternion_multiply


class GaussianModel(nn.Module):
    """3D Gaussian Splatting model.

    Represents a scene/object as a collection of 3D Gaussian ellipsoids.
    Each Gaussian has: position (μ), rotation (r as quaternion),
    scale (s), opacity (α), color (c via SH coefficients),
    and optional learnable semantic features (f).

    Attributes:
        _positions: (N, 3) Gaussian centers.
        _rotations: (N, 4) Quaternions (w, x, y, z).
        _scales: (N, 3) Log-space scales.
        _opacities: (N, 1) Logit-space opacities.
        _sh_coeffs: (N, C, 3) Spherical harmonics coefficients.
        _features: (N, D) Optional learnable semantic features.
        _binding: (N,) Optional triangle binding indices for rigging.
    """

    def __init__(
        self,
        num_gaussians: int = 0,
        sh_degree: int = 3,
        feature_dim: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.feature_dim = feature_dim
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        if num_gaussians > 0:
            self._init_params(num_gaussians, device)
        else:
            self._positions = nn.Parameter(torch.empty(0, 3, device=device))
            self._rotations = nn.Parameter(torch.empty(0, 4, device=device))
            self._scales = nn.Parameter(torch.empty(0, 3, device=device))
            self._opacities = nn.Parameter(torch.empty(0, 1, device=device))
            self._sh_coeffs = nn.Parameter(
                torch.empty(0, self.num_sh_coeffs, 3, device=device)
            )
            self._features = nn.Parameter(
                torch.empty(0, feature_dim, device=device)
            )

        # Non-learnable binding indices
        self.register_buffer(
            "_binding", torch.full((max(num_gaussians, 0),), -1, dtype=torch.long)
        )

    def _init_params(self, N: int, device: str) -> None:
        """Initialize N Gaussian primitives with default values."""
        self._positions = nn.Parameter(torch.randn(N, 3, device=device) * 0.1)
        self._rotations = nn.Parameter(
            F.normalize(torch.randn(N, 4, device=device), dim=-1)
        )
        self._scales = nn.Parameter(torch.full((N, 3), -3.0, device=device))
        self._opacities = nn.Parameter(torch.full((N, 1), 2.0, device=device))
        self._sh_coeffs = nn.Parameter(
            torch.zeros(N, self.num_sh_coeffs, 3, device=device)
        )
        self._features = nn.Parameter(
            torch.zeros(N, self.feature_dim, device=device)
        )

    # ── Property accessors ───────────────────────────────────────────────

    @property
    def num_gaussians(self) -> int:
        return self._positions.shape[0]

    @property
    def positions(self) -> torch.Tensor:
        """World-space positions (N, 3)."""
        return self._positions

    @property
    def rotations(self) -> torch.Tensor:
        """Normalized quaternions (N, 4)."""
        return F.normalize(self._rotations, dim=-1)

    @property
    def scales(self) -> torch.Tensor:
        """Activated scales (N, 3), positive via exp."""
        return torch.exp(self._scales)

    @property
    def opacities(self) -> torch.Tensor:
        """Activated opacities (N, 1) in [0, 1] via sigmoid."""
        return torch.sigmoid(self._opacities)

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """SH coefficients (N, C, 3)."""
        return self._sh_coeffs

    @property
    def features(self) -> torch.Tensor:
        """Semantic features (N, D)."""
        return self._features

    @property
    def binding(self) -> torch.Tensor:
        """Triangle binding indices (N,). -1 means unbound."""
        return self._binding

    @binding.setter
    def binding(self, indices: torch.Tensor) -> None:
        self._binding.copy_(indices)

    @property
    def rotation_matrices(self) -> torch.Tensor:
        """Rotation matrices (N, 3, 3)."""
        return quaternion_to_matrix(self.rotations)

    @property
    def covariance_3d(self) -> torch.Tensor:
        """3D covariance matrices (N, 3, 3) = R @ S @ S^T @ R^T."""
        R = self.rotation_matrices
        S = torch.diag_embed(self.scales)
        M = R @ S
        return M @ M.transpose(-1, -2)

    # ── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def from_point_cloud(
        cls,
        positions: torch.Tensor,
        colors: torch.Tensor | None = None,
        sh_degree: int = 3,
        feature_dim: int = 2,
        device: str = "cpu",
    ) -> GaussianModel:
        """Create Gaussian model from a point cloud.

        Args:
            positions: (N, 3) point positions.
            colors: Optional (N, 3) RGB in [0, 1].
            sh_degree: SH degree.
            feature_dim: Feature dimension.
            device: Target device.

        Returns:
            Initialized GaussianModel.
        """
        N = positions.shape[0]
        model = cls(num_gaussians=0, sh_degree=sh_degree, feature_dim=feature_dim)

        model._positions = nn.Parameter(positions.to(device).float())
        model._rotations = nn.Parameter(
            F.normalize(torch.randn(N, 4, device=device), dim=-1)
        )

        # Estimate initial scale from nearest neighbor distance
        with torch.no_grad():
            from scipy.spatial import KDTree

            tree = KDTree(positions.cpu().numpy())
            dists, _ = tree.query(positions.cpu().numpy(), k=4)
            avg_dist = np.mean(dists[:, 1:], axis=1)
            init_scale = np.log(np.clip(avg_dist, 1e-7, None))

        model._scales = nn.Parameter(
            torch.from_numpy(init_scale).float().unsqueeze(-1).expand(-1, 3).to(device).clone()
        )
        model._opacities = nn.Parameter(torch.full((N, 1), 2.0, device=device))

        num_sh = (sh_degree + 1) ** 2
        sh = torch.zeros(N, num_sh, 3, device=device)
        if colors is not None:
            # Set DC component (0th SH)
            C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
            sh[:, 0, :] = (colors.to(device).float() - 0.5) / C0
        model._sh_coeffs = nn.Parameter(sh)
        model._features = nn.Parameter(torch.zeros(N, feature_dim, device=device))
        model.register_buffer("_binding", torch.full((N,), -1, dtype=torch.long, device=device))

        return model

    # ── Operations ───────────────────────────────────────────────────────

    def clone(self) -> GaussianModel:
        """Deep clone this model."""
        new = GaussianModel.__new__(GaussianModel)
        nn.Module.__init__(new)
        new.sh_degree = self.sh_degree
        new.feature_dim = self.feature_dim
        new.num_sh_coeffs = self.num_sh_coeffs

        new._positions = nn.Parameter(self._positions.detach().clone())
        new._rotations = nn.Parameter(self._rotations.detach().clone())
        new._scales = nn.Parameter(self._scales.detach().clone())
        new._opacities = nn.Parameter(self._opacities.detach().clone())
        new._sh_coeffs = nn.Parameter(self._sh_coeffs.detach().clone())
        new._features = nn.Parameter(self._features.detach().clone())
        new.register_buffer("_binding", self._binding.detach().clone())

        return new

    def filter_by_mask(self, mask: torch.Tensor) -> GaussianModel:
        """Return a new model containing only the Gaussians where mask is True.

        Args:
            mask: Boolean tensor of shape (N,).

        Returns:
            Filtered GaussianModel.
        """
        new = GaussianModel.__new__(GaussianModel)
        nn.Module.__init__(new)
        new.sh_degree = self.sh_degree
        new.feature_dim = self.feature_dim
        new.num_sh_coeffs = self.num_sh_coeffs

        new._positions = nn.Parameter(self._positions[mask].detach().clone())
        new._rotations = nn.Parameter(self._rotations[mask].detach().clone())
        new._scales = nn.Parameter(self._scales[mask].detach().clone())
        new._opacities = nn.Parameter(self._opacities[mask].detach().clone())
        new._sh_coeffs = nn.Parameter(self._sh_coeffs[mask].detach().clone())
        new._features = nn.Parameter(self._features[mask].detach().clone())
        new.register_buffer("_binding", self._binding[mask].detach().clone())

        return new

    def merge(self, other: GaussianModel) -> GaussianModel:
        """Merge another GaussianModel into this one.

        Returns:
            New merged GaussianModel (non-destructive).
        """
        new = GaussianModel.__new__(GaussianModel)
        nn.Module.__init__(new)
        new.sh_degree = self.sh_degree
        new.feature_dim = self.feature_dim
        new.num_sh_coeffs = self.num_sh_coeffs

        new._positions = nn.Parameter(
            torch.cat([self._positions, other._positions]).detach().clone()
        )
        new._rotations = nn.Parameter(
            torch.cat([self._rotations, other._rotations]).detach().clone()
        )
        new._scales = nn.Parameter(
            torch.cat([self._scales, other._scales]).detach().clone()
        )
        new._opacities = nn.Parameter(
            torch.cat([self._opacities, other._opacities]).detach().clone()
        )
        new._sh_coeffs = nn.Parameter(
            torch.cat([self._sh_coeffs, other._sh_coeffs]).detach().clone()
        )
        new._features = nn.Parameter(
            torch.cat([self._features, other._features]).detach().clone()
        )
        new.register_buffer(
            "_binding",
            torch.cat([self._binding, other._binding]).detach().clone(),
        )

        return new

    def get_ellipsoid_endpoints(self) -> torch.Tensor:
        """Get 7 representative points per Gaussian (center + 6 axis endpoints).

        This is used for cage-based deformation (MVC).

        Returns:
            (N, 7, 3) tensor: [center, x+, x-, y+, y-, z+, z-].
        """
        R = self.rotation_matrices  # (N, 3, 3)
        s = self.scales  # (N, 3)
        mu = self.positions  # (N, 3)

        endpoints = [mu]  # center
        for axis_idx in range(3):
            axis_dir = R[:, :, axis_idx]  # (N, 3)
            offset = axis_dir * s[:, axis_idx:axis_idx + 1]  # (N, 3)
            endpoints.append(mu + offset)  # axis+
            endpoints.append(mu - offset)  # axis-

        return torch.stack(endpoints, dim=1)  # (N, 7, 3)

    def __repr__(self) -> str:
        return (
            f"GaussianModel(N={self.num_gaussians}, "
            f"sh_degree={self.sh_degree}, "
            f"feature_dim={self.feature_dim})"
        )
