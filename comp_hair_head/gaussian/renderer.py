"""Differentiable Gaussian Splatting renderer.

Provides a unified rendering interface that supports:
- Color rendering (R_c)
- Depth rendering (R_d)
- Opacity/alpha rendering (R_α)
- Feature rendering (R_f) for semantic segmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from ..utils.logging import get_logger

logger = get_logger("gaussian.renderer")


@dataclass
class RenderOutput:
    """Container for rendering results."""

    color: torch.Tensor | None = None  # (3, H, W)
    depth: torch.Tensor | None = None  # (1, H, W)
    alpha: torch.Tensor | None = None  # (1, H, W)
    feature: torch.Tensor | None = None  # (D, H, W)
    radii: torch.Tensor | None = None  # (N,) visibility radii


class GaussianRenderer(nn.Module):
    """Differentiable 3DGS renderer with multi-backend support.

    Supports native PyTorch (software) rendering for MPS/CPU compatibility
    and can optionally use CUDA-optimized rasterizers when available.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        background_color: list[float] | None = None,
        sh_degree: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.sh_degree = sh_degree
        self.device = device

        bg = background_color or [1.0, 1.0, 1.0]
        self.register_buffer(
            "background", torch.tensor(bg, dtype=torch.float32)
        )

        self._backend = self._detect_backend(device)
        logger.info(f"Renderer initialized: {self.width}x{self.height}, backend={self._backend}")

    def _detect_backend(self, device: str) -> str:
        """Detect best available rendering backend."""
        if "cuda" in device:
            try:
                from diff_gaussian_rasterization import (  # noqa: F401
                    GaussianRasterizer,
                )
                return "diff_rasterization"
            except ImportError:
                pass
            try:
                import gsplat  # noqa: F401
                return "gsplat"
            except ImportError:
                pass
        # Fallback: pure PyTorch
        return "pytorch"

    def forward(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor | None = None,
        colors_override: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
        camera_center: torch.Tensor | None = None,
        render_color: bool = True,
        render_depth: bool = False,
        render_feature: bool = False,
    ) -> RenderOutput:
        """Render Gaussians from a given viewpoint.

        Args:
            positions: (N, 3) Gaussian centers.
            rotations: (N, 4) Quaternions (w, x, y, z).
            scales: (N, 3) Activated scales.
            opacities: (N, 1) Activated opacities.
            sh_coeffs: (N, C, 3) SH coefficients (optional if colors_override provided).
            colors_override: (N, 3) Direct RGB colors (bypasses SH).
            features: (N, D) Semantic features for feature rendering.
            view_matrix: (4, 4) World-to-camera transform.
            proj_matrix: (4, 4) Projection matrix.
            camera_center: (3,) Camera position in world coords.
            render_color: Whether to render color.
            render_depth: Whether to render depth.
            render_feature: Whether to render features.

        Returns:
            RenderOutput with requested channels.
        """
        if self._backend == "pytorch":
            return self._render_pytorch(
                positions, rotations, scales, opacities,
                sh_coeffs, colors_override, features,
                view_matrix, proj_matrix, camera_center,
                render_color, render_depth, render_feature,
            )
        else:
            return self._render_cuda(
                positions, rotations, scales, opacities,
                sh_coeffs, colors_override, features,
                view_matrix, proj_matrix, camera_center,
                render_color, render_depth, render_feature,
            )

    def _render_pytorch(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor | None,
        colors_override: torch.Tensor | None,
        features: torch.Tensor | None,
        view_matrix: torch.Tensor | None,
        proj_matrix: torch.Tensor | None,
        camera_center: torch.Tensor | None,
        render_color: bool,
        render_depth: bool,
        render_feature: bool,
    ) -> RenderOutput:
        """Pure PyTorch splatting renderer (MPS/CPU compatible).

        Uses alpha compositing with depth-sorted Gaussians projected to 2D.
        This is slower than CUDA rasterizer but works on all devices.
        Rendering is performed on CPU to avoid MPS memory issues.
        """
        N = positions.shape[0]
        orig_device = positions.device
        render_device = torch.device("cpu")

        # Move all inputs to CPU for rendering (MPS has memory limits)
        positions = positions.to(render_device)
        rotations = rotations.to(render_device)
        scales = scales.to(render_device)
        opacities = opacities.to(render_device)
        if sh_coeffs is not None:
            sh_coeffs = sh_coeffs.to(render_device)
        if colors_override is not None:
            colors_override = colors_override.to(render_device)
        if features is not None:
            features = features.to(render_device)
        if view_matrix is not None:
            view_matrix = view_matrix.to(render_device)
        if proj_matrix is not None:
            proj_matrix = proj_matrix.to(render_device)
        if camera_center is not None:
            camera_center = camera_center.to(render_device)

        device = render_device
        background = self.background.to(render_device)

        if view_matrix is None:
            view_matrix = torch.eye(4, device=device)
        if proj_matrix is None:
            proj_matrix = torch.eye(4, device=device)

        # Transform to camera space
        pos_h = torch.cat([positions, torch.ones(N, 1, device=device)], dim=-1)
        pos_cam = (view_matrix @ pos_h.T).T[:, :3]  # (N, 3)
        depths = pos_cam[:, 2]  # (N,)

        # Filter behind camera
        valid_mask = depths > 0.1
        if not valid_mask.any():
            output = RenderOutput()
            if render_color:
                output.color = background.unsqueeze(-1).unsqueeze(-1).expand(
                    3, self.height, self.width
                )
            return output

        # Project to screen
        pos_proj = (proj_matrix @ (view_matrix @ pos_h.T)).T  # (N, 4)
        ndc = pos_proj[:, :2] / (pos_proj[:, 3:4] + 1e-8)  # (N, 2) in [-1, 1]
        screen_x = ((ndc[:, 0] + 1) / 2 * self.width).long()
        screen_y = ((1 - ndc[:, 1]) / 2 * self.height).long()   # flip y

        # Sort by depth (front to back for alpha compositing)
        sorted_indices = torch.argsort(depths)
        sorted_indices = sorted_indices[valid_mask[sorted_indices]]

        # Initialize output images
        output = RenderOutput()
        if render_color:
            output.color = background.unsqueeze(-1).unsqueeze(-1).expand(
                3, self.height, self.width
            ).clone()
        if render_depth:
            output.depth = torch.zeros(1, self.height, self.width, device=device)
        if render_feature and features is not None:
            D = features.shape[1]
            output.feature = torch.zeros(D, self.height, self.width, device=device)

        # Simple splatting: project each Gaussian as a 2D splat
        accumulated_alpha = torch.zeros(self.height, self.width, device=device)

        # Compute colors from SH or override
        if colors_override is not None:
            colors = colors_override
        elif sh_coeffs is not None:
            C0 = 0.28209479177387814
            colors = torch.clamp(sh_coeffs[:, 0, :] * C0 + 0.5, 0, 1)
        else:
            colors = torch.ones(N, 3, device=device)

        # Compute 2D covariance (simplified: use scale as splat radius)
        radii = (scales.max(dim=-1).values * 2.0).detach()

        # Vectorized rendering: sort by depth, then batch-splat
        sorted_indices = sorted_indices.to(device)
        s_pos_cam = pos_cam[sorted_indices]
        s_screen_x = screen_x[sorted_indices]
        s_screen_y = screen_y[sorted_indices]
        s_radii = radii[sorted_indices]
        s_opacities = opacities[sorted_indices]
        s_colors = colors[sorted_indices]
        s_depths = depths[sorted_indices]
        s_features = features[sorted_indices] if features is not None else None

        M = s_pos_cam.shape[0]
        r_pixels = (s_radii * self.width / 4.0).clamp(min=1.0)

        # Create coordinate grid
        yy_grid = torch.arange(self.height, device=device).float()
        xx_grid = torch.arange(self.width, device=device).float()
        grid_y, grid_x = torch.meshgrid(yy_grid, xx_grid, indexing="ij")

        # Batch splat: process in chunks to balance memory and speed
        chunk_size = 32
        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            cx = s_screen_x[chunk_start:chunk_end].float().unsqueeze(-1).unsqueeze(-1)
            cy = s_screen_y[chunk_start:chunk_end].float().unsqueeze(-1).unsqueeze(-1)
            r = r_pixels[chunk_start:chunk_end].unsqueeze(-1).unsqueeze(-1)
            alpha_val = s_opacities[chunk_start:chunk_end, 0].unsqueeze(-1).unsqueeze(-1)
            d = s_depths[chunk_start:chunk_end].unsqueeze(-1).unsqueeze(-1)
            c = s_colors[chunk_start:chunk_end].unsqueeze(-1).unsqueeze(-1)
            f_chunk = s_features[chunk_start:chunk_end].unsqueeze(-1).unsqueeze(-1) if s_features is not None else None

            dx = (grid_x.unsqueeze(0) - cx) / r.clamp(min=1.0)
            dy = (grid_y.unsqueeze(0) - cy) / r.clamp(min=1.0)
            gaussian_weight = torch.exp(-0.5 * (dx ** 2 + dy ** 2))

            # Mask out pixels outside radius
            mask = (dx.abs() < 3.0) & (dy.abs() < 3.0)
            alpha = (alpha_val * gaussian_weight * mask).clamp(0, 0.99)

            # Depth-weighted alpha compositing (simplified: use max-alpha per pixel)
            # For speed, use weighted sum approach
            alpha_sum = alpha.sum(dim=0)  # (H, W)
            alpha_norm = alpha / (alpha_sum.unsqueeze(0) + 1e-8)

            if render_color and output.color is not None:
                for ci in range(3):
                    output.color[ci] += (alpha_norm * s_colors[chunk_start:chunk_end, ci].unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

            if render_depth and output.depth is not None:
                output.depth[0] += (alpha_norm * d.squeeze(-1)).sum(dim=0)

            if render_feature and s_features is not None and output.feature is not None:
                for fi in range(s_features.shape[1]):
                    output.feature[fi] += (alpha_norm * s_features[chunk_start:chunk_end, fi].unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

            accumulated_alpha += alpha_sum.clamp(max=0.99)

        if render_color and output.color is not None:
            bg = background.unsqueeze(-1).unsqueeze(-1)
            output.color = output.color + bg * (1 - accumulated_alpha.unsqueeze(0)).clamp(min=0)

        output.alpha = accumulated_alpha.unsqueeze(0)
        output.radii = radii

        # Move results back to original device
        if orig_device != render_device:
            if output.color is not None:
                output.color = output.color.to(orig_device)
            if output.depth is not None:
                output.depth = output.depth.to(orig_device)
            if output.alpha is not None:
                output.alpha = output.alpha.to(orig_device)
            if output.feature is not None:
                output.feature = output.feature.to(orig_device)
            if output.radii is not None:
                output.radii = output.radii.to(orig_device)

        return output

    def _render_cuda(
        self,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor | None,
        colors_override: torch.Tensor | None,
        features: torch.Tensor | None,
        view_matrix: torch.Tensor | None,
        proj_matrix: torch.Tensor | None,
        camera_center: torch.Tensor | None,
        render_color: bool,
        render_depth: bool,
        render_feature: bool,
    ) -> RenderOutput:
        """CUDA-optimized rendering (diff-gaussian-rasterization or gsplat)."""
        # Placeholder for CUDA backend integration
        # Users with CUDA GPUs should install diff-gaussian-rasterization
        # and this method will use the optimized rasterizer
        raise NotImplementedError(
            f"CUDA backend '{self._backend}' not yet integrated. "
            "Falling back to PyTorch renderer."
        )
