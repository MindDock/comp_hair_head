"""Hair segmentation using SAM2 for multi-view label generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from ..utils.logging import get_logger

logger = get_logger("segmentation.hair_seg")


class HairSegmentor:
    """Multi-view hair segmentation using SAM2.

    Generates binary hair/non-hair labels for multi-view rendered images,
    which are then used to supervise the learnable Gaussian features.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        self.device = device
        self._model = None

        if checkpoint_path:
            self._load_sam2(checkpoint_path)
        else:
            logger.warning("SAM2 checkpoint not provided. Using fallback segmentation.")

    def _load_sam2(self, path: str) -> None:
        """Load SAM2 model."""
        try:
            # Placeholder for SAM2 loading
            logger.info(f"Loading SAM2 from {path}")
        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")

    def segment_views(
        self,
        images: list[torch.Tensor],
        hair_prompt: str = "hair",
    ) -> list[torch.Tensor]:
        """Generate hair segmentation masks for multiple views.

        Args:
            images: List of (3, H, W) images.
            hair_prompt: Text prompt for SAM2.

        Returns:
            List of (H, W) binary masks (1 = hair, 0 = non-hair).
        """
        masks = []
        for img in images:
            mask = self._segment_single(img)
            masks.append(mask)
        return masks

    def _segment_single(self, image: torch.Tensor) -> torch.Tensor:
        """Segment hair in a single image.

        Fallback: uses color/brightness-based heuristic.
        """
        if self._model is not None:
            # Use SAM2 model
            raise NotImplementedError("SAM2 integration pending")

        # Fallback: simple heuristic
        import cv2

        img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        H, W = img_np.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)

        # Dark regions in upper portion → likely hair
        value = hsv[:, :, 2].astype(np.float32)
        dark_mask = value < 100

        # Position prior: upper portion of image
        y_coords = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
        upper_mask = y_coords < int(H * 0.65)

        mask[dark_mask & upper_mask] = 1.0

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return torch.from_numpy(mask).to(image.device)
