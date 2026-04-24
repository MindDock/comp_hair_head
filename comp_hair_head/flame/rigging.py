"""Gaussian-to-mesh rigging: local ↔ global coordinate transforms.

Implements T_l2g and T_g2l from the paper, which transform Gaussian
primitives between triangle-local and world-global coordinate systems.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.geometry import (
    compute_triangle_local_frame,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from ..utils.logging import get_logger

logger = get_logger("flame.rigging")


def compute_binding_indices(
    positions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Find the nearest mesh triangle for each Gaussian position.

    Args:
        positions: (N, 3) Gaussian centers.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) face indices.

    Returns:
        binding: (N,) triangle index for each Gaussian.
    """
    # Compute triangle centroids
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0  # (F, 3)

    # Find nearest centroid for each Gaussian
    # (N, F)
    dists = torch.cdist(positions.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    binding = dists.argmin(dim=-1)  # (N,)

    return binding


def transform_local_to_global(
    positions_local: torch.Tensor,
    rotations_local: torch.Tensor,
    scales_local: torch.Tensor,
    binding: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T_l2g: Transform Gaussian primitives from triangle-local to global space.

    Each Gaussian is bound to a mesh triangle. Its local-space attributes
    are transformed to global coordinates using the triangle's frame.

    Args:
        positions_local: (N, 3) positions in local coords.
        rotations_local: (N, 4) quaternions in local frame.
        scales_local: (N, 3) log-scales in local frame.
        binding: (N,) triangle indices.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) face indices.

    Returns:
        positions_global: (N, 3) world positions.
        rotations_global: (N, 4) world quaternions.
        scales_global: (N, 3) world scales (log-space).
    """
    # Get triangle vertices for bound faces
    bound_faces = faces[binding]  # (N, 3)
    v0 = vertices[bound_faces[:, 0]]  # (N, 3)
    v1 = vertices[bound_faces[:, 1]]
    v2 = vertices[bound_faces[:, 2]]

    # Compute local frame
    t, R, eta = compute_triangle_local_frame(v0, v1, v2)

    # Transform position: μ_global = R @ (η * μ_local) + t
    positions_global = torch.bmm(
        R, (eta.unsqueeze(-1) * positions_local).unsqueeze(-1)
    ).squeeze(-1) + t

    # Transform rotation: r_global = R @ r_local (as quaternion multiplication)
    R_quat = matrix_to_quaternion(R)  # (N, 4)
    from ..utils.geometry import quaternion_multiply
    rotations_global = quaternion_multiply(R_quat, rotations_local)

    # Transform scale: s_global = log(η * exp(s_local))
    scales_global = scales_local + torch.log(eta + 1e-8)

    return positions_global, rotations_global, scales_global


def transform_global_to_local(
    positions_global: torch.Tensor,
    rotations_global: torch.Tensor,
    scales_global: torch.Tensor,
    binding: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T_g2l: Transform Gaussian primitives from global to triangle-local space.

    Inverse of T_l2g.

    Args:
        positions_global: (N, 3) world positions.
        rotations_global: (N, 4) world quaternions.
        scales_global: (N, 3) world scales (log-space).
        binding: (N,) triangle indices.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) face indices.

    Returns:
        positions_local: (N, 3) positions in local coords.
        rotations_local: (N, 4) quaternions in local frame.
        scales_local: (N, 3) log-scales in local frame.
    """
    bound_faces = faces[binding]
    v0 = vertices[bound_faces[:, 0]]
    v1 = vertices[bound_faces[:, 1]]
    v2 = vertices[bound_faces[:, 2]]

    t, R, eta = compute_triangle_local_frame(v0, v1, v2)

    # Inverse position: μ_local = R^T @ (μ_global - t) / η
    R_inv = R.transpose(-1, -2)
    positions_local = torch.bmm(
        R_inv, (positions_global - t).unsqueeze(-1)
    ).squeeze(-1) / (eta.unsqueeze(-1) + 1e-8)

    # Inverse rotation: r_local = R^T_quat * r_global
    R_inv_quat = matrix_to_quaternion(R_inv)
    from ..utils.geometry import quaternion_multiply
    rotations_local = quaternion_multiply(R_inv_quat, rotations_global)

    # Inverse scale
    scales_local = scales_global - torch.log(eta + 1e-8)

    return positions_local, rotations_local, scales_local


class GaussianRigging(nn.Module):
    """Manages rigging of Gaussians to a FLAME mesh.

    Stores binding indices and provides methods to transform
    between local and global coordinate systems as the mesh deforms.
    """

    def __init__(self):
        super().__init__()

    def bind(
        self,
        positions: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        binding_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute or set binding indices.

        Args:
            positions: (N, 3) Gaussian positions.
            vertices: (V, 3) mesh vertices.
            faces: (F, 3) face indices.
            binding_indices: Optional pre-computed bindings.

        Returns:
            binding: (N,) triangle indices.
        """
        if binding_indices is not None:
            return binding_indices
        return compute_binding_indices(positions, vertices, faces)

    def to_local(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        binding: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform from global to local coordinates."""
        return transform_global_to_local(
            positions, rotations, scales, binding, vertices, faces
        )

    def to_global(
        self,
        positions_local: torch.Tensor,
        rotations_local: torch.Tensor,
        scales_local: torch.Tensor,
        binding: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform from local to global coordinates."""
        return transform_local_to_global(
            positions_local, rotations_local, scales_local,
            binding, vertices, faces
        )
