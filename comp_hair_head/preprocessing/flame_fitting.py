"""FLAME parameter extraction from images."""

from __future__ import annotations

import torch
from ..utils.logging import get_logger

logger = get_logger("preprocessing.flame_fitting")


class FLAMEFitter:
    """Extract FLAME parameters from portrait images using VHAP or similar.

    Provides interface for FLAME parameter estimation from images,
    supporting multiple backends (VHAP, DECA, Deep3DFaceRecon).
    """

    def __init__(self, method: str = "vhap", device: str = "cpu"):
        self.method = method
        self.device = device
        logger.info(f"FLAMEFitter initialized with method={method}")

    def fit(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Estimate FLAME parameters from an image.

        Args:
            image: (3, H, W) portrait image in [0, 1].

        Returns:
            Dict with FLAME parameters:
                - "shape": (100,) shape coefficients
                - "expression": (50,) expression coefficients
                - "pose": (15,) joint rotations (axis-angle)
                - "global_rotation": (3,) global rotation
                - "global_translation": (3,) global translation
        """
        logger.warning(f"FLAME fitting ({self.method}) not yet integrated. Using defaults.")

        return {
            "shape": torch.zeros(100, device=self.device),
            "expression": torch.zeros(50, device=self.device),
            "pose": torch.zeros(15, device=self.device),
            "global_rotation": torch.zeros(3, device=self.device),
            "global_translation": torch.zeros(3, device=self.device),
        }
