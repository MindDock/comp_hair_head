"""Learnable semantic features for 3D Gaussian hair/face classification.

Augments each Gaussian with a 2D learnable feature vector, optimized with
binary cross-entropy loss against multi-view hair segmentation labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger

logger = get_logger("segmentation.learnable_feat")


class LearnableFeatureModule(nn.Module):
    """Learnable per-Gaussian semantic features for hair segmentation.

    Each Gaussian gets a learnable 2D feature f ∈ ℝ². These features are
    rendered via differentiable rasterization and supervised by SAM2-generated
    hair/non-hair labels using binary cross-entropy loss.

    After optimization, hair Gaussians are extracted via softmax on f.
    """

    def __init__(self, num_gaussians: int, feature_dim: int = 2, device: str = "cpu"):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Parameter(
            torch.zeros(num_gaussians, feature_dim, device=device)
        )

    def get_hair_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Get binary hair mask from learned features via softmax.

        Args:
            threshold: Classification threshold.

        Returns:
            (N,) boolean mask, True for hair Gaussians.
        """
        probs = F.softmax(self.features, dim=-1)  # (N, 2)
        # Channel 1 = hair probability
        return probs[:, 1] > threshold

    def compute_loss(
        self,
        rendered_features: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss.

        Args:
            rendered_features: (2, H, W) rendered feature map.
            target_labels: (H, W) binary labels (1 = hair).

        Returns:
            Scalar loss.
        """
        # Softmax over feature channels → (2, H, W) hair probability map
        probs = F.softmax(rendered_features, dim=0)
        hair_prob = probs[1]  # (H, W)

        return F.binary_cross_entropy(
            hair_prob,
            target_labels.float(),
            reduction="mean",
        )


def optimize_hair_features(
    gaussian_model,
    renderer,
    segmentor,
    views: list[dict],
    num_iterations: int = 200,
    lr: float = 0.01,
    device: str = "cpu",
) -> torch.Tensor:
    """Optimize learnable features to classify hair vs non-hair Gaussians.

    Args:
        gaussian_model: GaussianModel instance.
        renderer: GaussianRenderer instance.
        segmentor: HairSegmentor instance.
        views: List of dicts with 'image' and 'view_matrix' keys.
        num_iterations: Optimization iterations.
        lr: Learning rate.
        device: Torch device.

    Returns:
        hair_mask: (N,) boolean mask for hair Gaussians.
    """
    feature_module = LearnableFeatureModule(
        gaussian_model.num_gaussians,
        feature_dim=2,
        device=device,
    )
    optimizer = torch.optim.Adam(feature_module.parameters(), lr=lr)

    # Generate segmentation labels for all views
    logger.info("Generating multi-view hair labels...")
    images = [v["image"] for v in views]
    labels = segmentor.segment_views(images)

    logger.info(f"Optimizing hair features for {num_iterations} iterations...")
    for it in range(num_iterations):
        total_loss = 0.0
        optimizer.zero_grad()

        for view_idx, view in enumerate(views):
            # Render features
            output = renderer(
                positions=gaussian_model.positions,
                rotations=gaussian_model.rotations,
                scales=gaussian_model.scales,
                opacities=gaussian_model.opacities,
                features=feature_module.features,
                view_matrix=view.get("view_matrix"),
                proj_matrix=view.get("proj_matrix"),
                render_color=False,
                render_feature=True,
            )

            if output.feature is not None:
                loss = feature_module.compute_loss(output.feature, labels[view_idx])
                total_loss += loss

        if total_loss > 0:
            total_loss.backward()
            optimizer.step()

        if (it + 1) % 50 == 0:
            logger.info(f"  Iteration {it + 1}/{num_iterations}, loss: {total_loss.item():.4f}")

    hair_mask = feature_module.get_hair_mask()
    num_hair = hair_mask.sum().item()
    logger.info(f"Hair segmentation complete: {num_hair}/{gaussian_model.num_gaussians} hair Gaussians")

    return hair_mask
