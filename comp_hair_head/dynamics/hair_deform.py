"""Hair Gaussian deformation via cage-based MVC interpolation.

Propagates cage deformation to hair Gaussian primitives using
pre-computed Mean Value Coordinates, with lightweight principal-axis
approximation for rotation and scale updates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.geometry import quaternion_to_matrix, matrix_to_quaternion, quaternion_multiply
from ..utils.logging import get_logger

logger = get_logger("dynamics.hair_deform")


class HairDeformer:
    """Deform hair Gaussians based on cage deformation.

    Given a deformed cage, propagates the deformation to all hair Gaussian
    primitives using pre-computed MVC weights. Uses lightweight principal-axis
    approximation (no SVD) for rotation and scale updates.

    The deformation process:
    1. Use MVC to compute deformed 7-point representation of each Gaussian
    2. Extract new position (midpoint of principal axis endpoints)
    3. Compute rotation delta from principal axis direction change
    4. Adjust scale from axis length ratio
    """

    def __init__(self):
        self.source_endpoints: torch.Tensor | None = None  # (N, 7, 3)
        self.mvc_weights: torch.Tensor | None = None  # (N, 7, M)
        self.principal_axis_indices: torch.Tensor | None = None  # (N,) index in {0,1,2}

    def initialize(
        self,
        source_endpoints: torch.Tensor,
        mvc_weights: torch.Tensor,
        source_scales: torch.Tensor,
    ) -> None:
        """Initialize deformer with source Gaussian data.

        Args:
            source_endpoints: (N, 7, 3) from GaussianModel.get_ellipsoid_endpoints().
            mvc_weights: (N, 7, M) MVC weights from compute_gaussian_mvc_weights().
            source_scales: (N, 3) activated scales of source Gaussians.
        """
        self.source_endpoints = source_endpoints
        self.mvc_weights = mvc_weights

        # Identify principal axis: i* = argmax_i ||x_s^{i+} - x_s^{i-}||
        # Endpoints layout: [center, x+, x-, y+, y-, z+, z-]
        axis_lengths = torch.stack([
            torch.norm(source_endpoints[:, 1] - source_endpoints[:, 2], dim=-1),  # x
            torch.norm(source_endpoints[:, 3] - source_endpoints[:, 4], dim=-1),  # y
            torch.norm(source_endpoints[:, 5] - source_endpoints[:, 6], dim=-1),  # z
        ], dim=-1)  # (N, 3)

        self.principal_axis_indices = axis_lengths.argmax(dim=-1)  # (N,)
        self.source_principal_lengths = axis_lengths.gather(
            1, self.principal_axis_indices.unsqueeze(1)
        ).squeeze(1)  # (N,)

        # Source principal axis directions
        self.source_principal_dirs = self._get_principal_directions(source_endpoints)

        logger.info(f"HairDeformer initialized: {source_endpoints.shape[0]} Gaussians")

    def deform(
        self,
        deformed_cage_vertices: torch.Tensor,
        source_rotations: torch.Tensor,
        source_scales_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute deformed Gaussian attributes from deformed cage.

        Args:
            deformed_cage_vertices: (M, 3) deformed cage vertex positions.
            source_rotations: (N, 4) source Gaussian quaternions.
            source_scales_log: (N, 3) source Gaussian log-scales.

        Returns:
            deformed_positions: (N, 3) new Gaussian centers.
            deformed_rotations: (N, 4) new quaternions.
            deformed_scales_log: (N, 3) new log-scales.
        """
        assert self.mvc_weights is not None, "Call initialize() first"
        N = self.mvc_weights.shape[0]

        # Step 1: Deform all 7 control points via MVC
        # deformed_endpoints[n, k] = sum_m (weights[n, k, m] * cage_verts[m])
        deformed_endpoints = torch.matmul(
            self.mvc_weights, deformed_cage_vertices
        )  # (N, 7, 3)

        # Step 2: Extract deformed principal axis endpoints
        # Layout: [center, x+, x-, y+, y-, z+, z-]
        # For axis i, positive endpoint is at index 2*i+1, negative at 2*i+2
        deformed_principal_dirs = self._get_principal_directions(deformed_endpoints)

        # New position = midpoint of principal axis
        pos_idx = self.principal_axis_indices  # (N,)
        pos_plus_indices = 2 * pos_idx + 1
        pos_minus_indices = 2 * pos_idx + 2

        pos_plus = deformed_endpoints[
            torch.arange(N), pos_plus_indices
        ]  # (N, 3)
        pos_minus = deformed_endpoints[
            torch.arange(N), pos_minus_indices
        ]  # (N, 3)

        deformed_positions = (pos_plus + pos_minus) / 2.0  # (N, 3)

        # Step 3: Compute rotation delta from principal axis change
        deformed_rotations = self._compute_rotation_update(
            source_rotations, deformed_principal_dirs
        )

        # Step 4: Compute scale adjustment from axis length ratio
        deformed_principal_lengths = torch.norm(pos_plus - pos_minus, dim=-1)  # (N,)
        scale_ratio = deformed_principal_lengths / (
            self.source_principal_lengths.to(deformed_cage_vertices.device) + 1e-8
        )

        deformed_scales_log = source_scales_log.clone()
        # Adjust only the principal axis scale
        deformed_scales_log[
            torch.arange(N), self.principal_axis_indices
        ] += torch.log(scale_ratio.clamp(min=1e-4))

        return deformed_positions, deformed_rotations, deformed_scales_log

    def _get_principal_directions(self, endpoints: torch.Tensor) -> torch.Tensor:
        """Get normalized principal axis direction for each Gaussian.

        Args:
            endpoints: (N, 7, 3)

        Returns:
            directions: (N, 3) unit vectors along principal axis.
        """
        N = endpoints.shape[0]
        idx = self.principal_axis_indices  # (N,)

        plus_indices = 2 * idx + 1
        minus_indices = 2 * idx + 2

        plus_pts = endpoints[torch.arange(N), plus_indices]
        minus_pts = endpoints[torch.arange(N), minus_indices]

        dirs = plus_pts - minus_pts
        return F.normalize(dirs, dim=-1)

    def _compute_rotation_update(
        self,
        source_rotations: torch.Tensor,
        deformed_dirs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated rotation from principal axis direction change.

        Finds ΔR that rotates source_dir to deformed_dir, then applies
        to source rotation: R_new = ΔR @ R_source.

        Args:
            source_rotations: (N, 4) source quaternions.
            deformed_dirs: (N, 3) deformed principal axis directions.

        Returns:
            (N, 4) updated quaternions.
        """
        source_dirs = self.source_principal_dirs.to(deformed_dirs.device)

        # Rotation from source_dir to deformed_dir via cross product
        cross = torch.cross(source_dirs, deformed_dirs, dim=-1)
        dot = (source_dirs * deformed_dirs).sum(dim=-1)

        # Quaternion from rotation: q = [1 + dot, cross] (un-normalized)
        # This handles the case where directions are nearly parallel
        delta_q = torch.cat([
            (1.0 + dot).unsqueeze(-1),
            cross,
        ], dim=-1)  # (N, 4)

        delta_q = F.normalize(delta_q, dim=-1)

        # Apply: r_new = delta_q * r_source
        return quaternion_multiply(delta_q, source_rotations)
