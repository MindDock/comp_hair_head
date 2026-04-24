"""Non-rigid registration and component assembly.

Implements the joint reconstruction + registration loss for binding
bald Gaussians to FLAME mesh, and the component assembly with
collision-aware optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger

logger = get_logger("registration.assembly")


def chamfer_distance(
    points_a: torch.Tensor, points_b: torch.Tensor
) -> torch.Tensor:
    """Compute symmetric Chamfer distance between two point sets.

    Args:
        points_a: (N, 3)
        points_b: (M, 3)

    Returns:
        Scalar Chamfer distance.
    """
    dist = torch.cdist(points_a.unsqueeze(0), points_b.unsqueeze(0)).squeeze(0)
    # A → B: for each point in A, min distance to B
    a_to_b = dist.min(dim=1).values.mean()
    # B → A
    b_to_a = dist.min(dim=0).values.mean()
    return a_to_b + b_to_a


def reconstruction_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.1,
    lpips_fn=None,
) -> torch.Tensor:
    """Multi-term reconstruction loss: L1 + SSIM + LPIPS.

    L_rec = (1 - λ₂) * L1 + λ₂ * L_ssim + λ₃ * L_lpips

    Args:
        rendered: (3, H, W) rendered image.
        target: (3, H, W) target image.
        lambda_ssim: Weight for SSIM loss.
        lambda_lpips: Weight for LPIPS loss.
        lpips_fn: Optional LPIPS loss function.

    Returns:
        Scalar loss.
    """
    # L1
    l1 = F.l1_loss(rendered, target)

    # SSIM (simplified implementation)
    ssim_loss = 1.0 - _ssim(rendered.unsqueeze(0), target.unsqueeze(0))

    loss = (1 - lambda_ssim) * l1 + lambda_ssim * ssim_loss

    # LPIPS
    if lpips_fn is not None:
        lpips_loss = lpips_fn(rendered.unsqueeze(0), target.unsqueeze(0)).mean()
        loss += lambda_lpips * lpips_loss

    return loss


def collision_loss(
    hair_positions: torch.Tensor,
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    margin: float = 0.002,
) -> torch.Tensor:
    """Collision penalty: ensure hair points stay outside head mesh.

    L_collision = mean(max(0, margin - signed_distance))²

    Args:
        hair_positions: (N, 3) hair Gaussian centers.
        mesh_vertices: (V, 3) FLAME mesh vertices.
        mesh_faces: (F, 3) face indices.
        margin: Minimum allowed distance (epsilon in paper).

    Returns:
        Scalar collision loss.
    """
    from ..utils.geometry import signed_distance_to_mesh

    sdf = signed_distance_to_mesh(hair_positions, mesh_vertices, mesh_faces)

    # Penalize points that are inside or too close
    violation = F.relu(margin - sdf)

    return (violation ** 2).mean()


def registration_loss(
    rendered_prior: torch.Tensor,
    rendered_bald: torch.Tensor,
    positions_prior: torch.Tensor,
    positions_bald: torch.Tensor,
    target_image: torch.Tensor,
    lambda_chamfer: float = 0.1,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.1,
    lpips_fn=None,
) -> torch.Tensor:
    """Joint reconstruction + registration loss (Eq. 5 in paper).

    L = L_rec(C_prior, I_bald) + L_rec(C_bald, I_bald) + λ₁ * L_chamfer

    Args:
        rendered_prior: (3, H, W) rendered prior model.
        rendered_bald: (3, H, W) rendered bald Gaussians.
        positions_prior: (N1, 3) prior Gaussian positions.
        positions_bald: (N2, 3) bald Gaussian positions.
        target_image: (3, H, W) target bald image.
        lambda_chamfer: Chamfer distance weight.
        lambda_ssim: SSIM weight in reconstruction loss.
        lambda_lpips: LPIPS weight in reconstruction loss.
        lpips_fn: LPIPS function.

    Returns:
        Scalar loss.
    """
    # Reconstruction losses
    loss_prior = reconstruction_loss(
        rendered_prior, target_image, lambda_ssim, lambda_lpips, lpips_fn
    )
    loss_bald = reconstruction_loss(
        rendered_bald, target_image, lambda_ssim, lambda_lpips, lpips_fn
    )

    # Chamfer distance
    loss_chamfer = chamfer_distance(positions_prior, positions_bald)

    return loss_prior + loss_bald + lambda_chamfer * loss_chamfer


def assembly_loss(
    rendered_head: torch.Tensor,
    target_image: torch.Tensor,
    hair_positions: torch.Tensor,
    mesh_vertices: torch.Tensor,
    mesh_faces: torch.Tensor,
    lambda_geo: float = 0.5,
    collision_margin: float = 0.002,
    lambda_ssim: float = 0.2,
    lambda_lpips: float = 0.1,
    lpips_fn=None,
) -> torch.Tensor:
    """Component assembly loss (Eq. 6 in paper).

    L = L_rec(C_head, I_head) + λ_geo * L_collision

    Args:
        rendered_head: (3, H, W) composited head rendering.
        target_image: (3, H, W) original head image.
        hair_positions: (N, 3) hair Gaussian positions.
        mesh_vertices: (V, 3) FLAME vertices.
        mesh_faces: (F, 3) FLAME faces.
        lambda_geo: Weight for collision loss.
        collision_margin: Margin epsilon.
        lambda_ssim: SSIM weight.
        lambda_lpips: LPIPS weight.
        lpips_fn: LPIPS function.

    Returns:
        Scalar loss.
    """
    loss_rec = reconstruction_loss(
        rendered_head, target_image, lambda_ssim, lambda_lpips, lpips_fn
    )
    loss_col = collision_loss(
        hair_positions, mesh_vertices, mesh_faces, collision_margin
    )

    return loss_rec + lambda_geo * loss_col


# ── Helper: Simplified SSIM ─────────────────────────────────────────────────

def _ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """Simplified SSIM computation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()
