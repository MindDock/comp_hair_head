"""Boundary-aware reassignment for hair/face Gaussian reclassification.

Addresses inaccurate semantic labels at hair-face boundaries by
reclassifying boundary Gaussians based on color and scale features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.logging import get_logger

logger = get_logger("segmentation.boundary_reassign")


def boundary_aware_reassignment(
    positions: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    hair_mask: torch.Tensor,
    depth_maps: list[torch.Tensor],
    label_maps: list[torch.Tensor],
    view_matrices: list[torch.Tensor],
    proj_matrices: list[torch.Tensor],
    boundary_threshold: int = 5,
    image_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """Reassign boundary Gaussians based on color/scale similarity.

    Steps:
    1. Project Gaussians to each view and identify 2D boundary regions
    2. Classify Gaussians as boundary or interior
    3. Compute feature centroids for hair (interior) and bald classes
    4. Reclassify boundary Gaussians by nearest centroid

    Args:
        positions: (N, 3) Gaussian positions.
        colors: (N, 3) Gaussian colors (DC SH component).
        scales: (N, 3) Gaussian scales.
        hair_mask: (N,) initial boolean hair mask.
        depth_maps: List of (H, W) rendered depth maps per view.
        label_maps: List of (H, W) binary label maps per view.
        view_matrices: List of (4, 4) view matrices.
        proj_matrices: List of (4, 4) projection matrices.
        boundary_threshold: Pixel distance for boundary region.
        image_size: (H, W) image dimensions.

    Returns:
        refined_mask: (N,) refined boolean hair mask.
    """
    N = positions.shape[0]
    device = positions.device
    H, W = image_size

    # Identify boundary Gaussians across all views
    is_boundary = torch.zeros(N, dtype=torch.bool, device=device)

    for view_idx in range(len(label_maps)):
        labels = label_maps[view_idx]  # (H, W)

        # Compute boundary pixels using morphological gradient
        boundary_2d = _compute_boundary(labels, boundary_threshold)  # (H, W) bool

        # Project Gaussians to this view
        projected = _project_gaussians(
            positions, view_matrices[view_idx], proj_matrices[view_idx], (H, W)
        )  # (N, 2) pixel coords

        # Check which Gaussians project to boundary regions
        px = projected[:, 0].long().clamp(0, W - 1)
        py = projected[:, 1].long().clamp(0, H - 1)

        in_boundary = boundary_2d[py, px]
        is_boundary |= in_boundary

    # Separate into boundary and non-boundary subsets
    interior_hair = hair_mask & ~is_boundary  # secure hair
    interior_bald = ~hair_mask & ~is_boundary  # secure bald

    # Compute class centroids from non-boundary Gaussians
    features = torch.cat([colors, scales], dim=-1)  # (N, 6)

    if interior_hair.sum() > 0:
        hair_centroid = features[interior_hair].mean(dim=0)  # (6,)
    else:
        hair_centroid = features[hair_mask].mean(dim=0)

    if interior_bald.sum() > 0:
        bald_centroid = features[interior_bald].mean(dim=0)  # (6,)
    else:
        bald_centroid = features[~hair_mask].mean(dim=0)

    # Reclassify boundary Gaussians
    refined_mask = hair_mask.clone()
    boundary_indices = is_boundary.nonzero(as_tuple=True)[0]

    if boundary_indices.numel() > 0:
        boundary_features = features[boundary_indices]  # (B, 6)

        hair_dist = torch.norm(boundary_features - hair_centroid, dim=-1)
        bald_dist = torch.norm(boundary_features - bald_centroid, dim=-1)

        # Assign to nearest class
        is_hair = hair_dist < bald_dist
        refined_mask[boundary_indices] = is_hair

    num_changed = (refined_mask != hair_mask).sum().item()
    logger.info(
        f"Boundary reassignment: {boundary_indices.numel()} boundary Gaussians, "
        f"{num_changed} reclassified"
    )

    return refined_mask


def _compute_boundary(
    labels: torch.Tensor, threshold: int = 5
) -> torch.Tensor:
    """Compute boundary region from binary label map.

    Args:
        labels: (H, W) binary labels.
        threshold: Pixel width of boundary band.

    Returns:
        (H, W) boolean boundary mask.
    """
    import cv2
    import numpy as np

    label_np = labels.cpu().numpy().astype(np.uint8)

    # Dilate and erode to find boundary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (threshold * 2 + 1, threshold * 2 + 1)
    )
    dilated = cv2.dilate(label_np, kernel)
    eroded = cv2.erode(label_np, kernel)

    boundary = (dilated - eroded) > 0
    return torch.tensor(boundary, dtype=torch.bool, device=labels.device)


def _project_gaussians(
    positions: torch.Tensor,
    view_matrix: torch.Tensor,
    proj_matrix: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Project 3D positions to 2D pixel coordinates.

    Args:
        positions: (N, 3) 3D positions.
        view_matrix: (4, 4) world-to-camera.
        proj_matrix: (4, 4) projection matrix.
        image_size: (H, W).

    Returns:
        (N, 2) pixel coordinates (x, y).
    """
    H, W = image_size
    N = positions.shape[0]
    device = positions.device

    pos_h = torch.cat([positions, torch.ones(N, 1, device=device)], dim=-1)
    proj = (proj_matrix @ view_matrix @ pos_h.T).T  # (N, 4)
    ndc = proj[:, :2] / (proj[:, 3:4] + 1e-8)

    px = ((ndc[:, 0] + 1) / 2 * W).clamp(0, W - 1)
    py = ((1 - ndc[:, 1]) / 2 * H).clamp(0, H - 1)

    return torch.stack([px, py], dim=-1)
