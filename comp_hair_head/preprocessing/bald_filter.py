"""2D Bald Filter: removes hair from portrait images.

Transforms a frontal portrait into a hairless (bald) version while
preserving face identity and skin appearance.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
from PIL import Image

from ..utils.logging import get_logger

logger = get_logger("preprocessing.bald_filter")


class BaldFilter:
    """Remove hair from portrait images.

    Supports multiple backends:
    - 'inpainting': Mask hair region and inpaint with OpenCV
    - 'barbershop': GAN-based hair removal (requires pretrained model)
    - 'diffusion': Diffusion-based inpainting (requires stable diffusion)

    For production use, a GAN/diffusion-based approach is recommended.
    The inpainting fallback provides basic functionality.
    """

    def __init__(self, method: str = "inpainting", device: str = "cpu"):
        self.method = method
        self.device = device
        self._model = None

        logger.info(f"BaldFilter initialized with method={method}")

    def process(self, image: torch.Tensor, hair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Remove hair from a portrait image.

        Args:
            image: (3, H, W) input portrait in [0, 1].
            hair_mask: Optional (1, H, W) binary hair mask in [0, 1].
                If None, will attempt to detect hair automatically.

        Returns:
            bald_image: (3, H, W) hairless image in [0, 1].
        """
        if self.method == "inpainting":
            return self._inpaint(image, hair_mask)
        elif self.method == "barbershop":
            return self._barbershop(image)
        elif self.method == "diffusion":
            return self._diffusion_inpaint(image, hair_mask)
        else:
            raise ValueError(f"Unknown bald filter method: {self.method}")

    def _inpaint(self, image: torch.Tensor, hair_mask: torch.Tensor | None) -> torch.Tensor:
        """OpenCV-based inpainting approach."""
        import cv2

        img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        if hair_mask is None:
            hair_mask = self._detect_hair_mask(img_np)
        else:
            hair_mask = (hair_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        # Dilate mask slightly for better coverage
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hair_mask = cv2.dilate(hair_mask, kernel, iterations=2)

        # Inpaint
        result = cv2.inpaint(img_np, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).permute(2, 0, 1)
        return result_tensor

    def _detect_hair_mask(self, img_np: np.ndarray) -> np.ndarray:
        """Simple hair detection using color thresholding.

        This is a basic fallback. Production use should employ
        a proper hair segmentation model (e.g., SAM2 or BiSeNet).
        """
        import cv2

        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Hair tends to be dark or specific colors
        # Low value (dark) regions above forehead likely hair
        h, w = img_np.shape[:2]

        # Simple heuristic: upper portion of image + dark pixels
        mask = np.zeros((h, w), dtype=np.uint8)
        value_channel = hsv[:, :, 2]

        # Dark regions in the upper half
        dark_mask = value_channel < 80
        upper_mask = np.zeros_like(dark_mask)
        upper_mask[:int(h * 0.7), :] = True

        mask[dark_mask & upper_mask] = 255

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _barbershop(self, image: torch.Tensor) -> torch.Tensor:
        """GAN-based hair removal (placeholder for BarbershopGAN integration)."""
        logger.warning("Barbershop GAN not loaded. Returning original image.")
        return image.clone()

    def _diffusion_inpaint(
        self, image: torch.Tensor, hair_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Diffusion-based inpainting (placeholder for SD inpainting pipeline)."""
        logger.warning("Diffusion inpainting not configured. Falling back to CV inpainting.")
        return self._inpaint(image, hair_mask)
