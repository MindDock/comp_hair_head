"""Mean Value Coordinates (MVC) computation.

Computes MVC weights that express interior points as convex combinations
of cage vertices, enabling smooth deformation propagation from cage to
enclosed Gaussian primitives.
"""

from __future__ import annotations

import numpy as np
import torch
from ..utils.logging import get_logger

logger = get_logger("dynamics.mvc")


def compute_mvc_weights(
    points: torch.Tensor,
    cage_vertices: torch.Tensor,
    cage_faces: torch.Tensor,
) -> torch.Tensor:
    """Compute Mean Value Coordinates for points inside a cage.

    For each point x and cage vertex c_j, the MVC weight w_j is:
        w_j = sum_{faces containing c_j} [tan(α_{j-1}/2) + tan(α_{j+1}/2)] / ||c_j - x||
    Then normalized: w_j = w_j / sum_k w_k

    Args:
        points: (P, 3) interior points.
        cage_vertices: (M, 3) cage vertex positions.
        cage_faces: (F, 3) cage triangle faces.

    Returns:
        weights: (P, M) MVC weights, each row sums to ~1.
    """
    P = points.shape[0]
    M = cage_vertices.shape[0]
    device = points.device

    weights = torch.zeros(P, M, device=device, dtype=torch.float32)

    # Compute vectors from each point to each cage vertex
    # d[p, m] = cage_vertices[m] - points[p]
    d = cage_vertices.unsqueeze(0) - points.unsqueeze(1)  # (P, M, 3)
    d_norm = torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-8)  # (P, M, 1)
    u = d / d_norm  # (P, M, 3) unit vectors

    # For each triangle face, accumulate MVC weights
    for face_idx in range(cage_faces.shape[0]):
        i0, i1, i2 = cage_faces[face_idx]

        u0 = u[:, i0]  # (P, 3)
        u1 = u[:, i1]
        u2 = u[:, i2]

        # Compute angles of the spherical triangle
        # l0 = angle between u1 and u2, etc.
        l0 = torch.acos(torch.clamp(
            torch.sum(u1 * u2, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7
        ))
        l1 = torch.acos(torch.clamp(
            torch.sum(u0 * u2, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7
        ))
        l2 = torch.acos(torch.clamp(
            torch.sum(u0 * u1, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7
        ))

        # Half perimeter
        h = (l0 + l1 + l2) / 2.0  # (P,)

        # Spherical excess angles using half-angle formula
        # theta_i = 2 * arctan(sqrt(sin(h-l0)*sin(h-l1)) / (sin(h)*sin(h-l2)))
        eps = 1e-8

        denom = torch.sqrt(
            torch.clamp(torch.sin(h) * torch.sin(h - l0), min=eps)
        )

        # theta for vertex i0
        num0 = torch.sqrt(
            torch.clamp(torch.sin(h - l1) * torch.sin(h - l2), min=eps)
        )
        theta0 = 2.0 * torch.atan2(num0, denom)

        # theta for vertex i1
        denom1 = torch.sqrt(
            torch.clamp(torch.sin(h) * torch.sin(h - l1), min=eps)
        )
        num1 = torch.sqrt(
            torch.clamp(torch.sin(h - l0) * torch.sin(h - l2), min=eps)
        )
        theta1 = 2.0 * torch.atan2(num1, denom1)

        # theta for vertex i2
        denom2 = torch.sqrt(
            torch.clamp(torch.sin(h) * torch.sin(h - l2), min=eps)
        )
        num2 = torch.sqrt(
            torch.clamp(torch.sin(h - l0) * torch.sin(h - l1), min=eps)
        )
        theta2 = 2.0 * torch.atan2(num2, denom2)

        # Accumulate: w_j += (theta_{j-1} + theta_{j+1} - pi) / (d_norm * sin(l_opposite))
        # Simplified: w_j += tan(theta_j / 2) / d_norm_j
        w0 = torch.tan(theta0 / 2.0) / d_norm[:, i0, 0].clamp(min=eps)
        w1 = torch.tan(theta1 / 2.0) / d_norm[:, i1, 0].clamp(min=eps)
        w2 = torch.tan(theta2 / 2.0) / d_norm[:, i2, 0].clamp(min=eps)

        weights[:, i0] += w0
        weights[:, i1] += w1
        weights[:, i2] += w2

    # Normalize
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = weights / weight_sum

    logger.info(f"MVC weights computed: {P} points × {M} cage vertices")
    return weights


def compute_gaussian_mvc_weights(
    endpoints: torch.Tensor,
    cage_vertices: torch.Tensor,
    cage_faces: torch.Tensor,
) -> torch.Tensor:
    """Compute MVC weights for Gaussian ellipsoid endpoints.

    Each Gaussian is represented by 7 control points (center + 6 axis endpoints).
    MVC weights are computed for all 7 points.

    Args:
        endpoints: (N, 7, 3) Gaussian ellipsoid endpoints.
        cage_vertices: (M, 3) cage vertex positions.
        cage_faces: (F, 3) cage faces.

    Returns:
        weights: (N, 7, M) MVC weights.
    """
    N = endpoints.shape[0]
    M = cage_vertices.shape[0]
    device = endpoints.device

    # Flatten to (N*7, 3)
    flat_points = endpoints.reshape(-1, 3)

    # Compute MVC for all points
    flat_weights = compute_mvc_weights(flat_points, cage_vertices, cage_faces)

    # Reshape back
    return flat_weights.reshape(N, 7, M)


def deform_points_with_mvc(
    weights: torch.Tensor,
    deformed_cage_vertices: torch.Tensor,
) -> torch.Tensor:
    """Deform points using pre-computed MVC weights and deformed cage vertices.

    x_d = sum_j (w_j * c_d,j)

    Args:
        weights: (P, M) or (N, 7, M) MVC weights.
        deformed_cage_vertices: (M, 3) deformed cage vertex positions.

    Returns:
        Deformed points: same shape as weights prefix dims, with final dim 3.
    """
    # weights: (..., M), cage_verts: (M, 3)
    return torch.matmul(weights, deformed_cage_vertices)
