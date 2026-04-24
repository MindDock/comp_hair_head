"""Hairstyle transfer pipeline.

Enables transferring hair from one identity (source) to another (target),
leveraging the decoupled compositional representation.
"""

from __future__ import annotations

import torch

from ..gaussian.model import GaussianModel
from ..flame.flame_model import FLAMEModel
from ..flame.rigging import GaussianRigging
from ..utils.logging import get_logger

logger = get_logger("pipeline.transfer")


class HairstyleTransfer:
    """Transfer hairstyle between identities.

    Since hair and face are stored in triangle-local coordinates,
    hairstyle transfer simply means:
    1. Take G'_hair from source identity
    2. Apply it to target identity's FLAME mesh
    3. Re-run assembly optimization if needed
    """

    def __init__(self):
        self.rigging = GaussianRigging()

    def transfer(
        self,
        source_hair_local: GaussianModel,
        target_bald_local: GaussianModel,
        target_flame: FLAMEModel,
        target_flame_params: dict[str, torch.Tensor],
    ) -> dict:
        """Transfer hairstyle from source to target.

        Args:
            source_hair_local: Source hair Gaussians in local coords.
            target_bald_local: Target bald Gaussians in local coords.
            target_flame: Target FLAME model.
            target_flame_params: Target FLAME parameters.

        Returns:
            Dict with:
                - "G_hair_transferred": GaussianModel (transferred hair)
                - "G_combined": GaussianModel (hair + bald combined)
        """
        logger.info("Transferring hairstyle...")

        # Get target mesh
        output = target_flame(
            shape_params=target_flame_params["shape"],
            expression_params=target_flame_params.get("expression"),
        )
        vertices = output["vertices"].squeeze(0)
        faces = output["faces"]

        # Transform source hair to target's global space
        hair_pos_global, hair_rot_global, hair_scale_global = self.rigging.to_global(
            source_hair_local.positions,
            source_hair_local.rotations,
            source_hair_local._scales,
            source_hair_local.binding,
            vertices, faces,
        )

        # Create transferred hair model
        G_transferred = source_hair_local.clone()
        G_transferred._positions = torch.nn.Parameter(hair_pos_global)
        G_transferred._rotations = torch.nn.Parameter(hair_rot_global)
        G_transferred._scales = torch.nn.Parameter(hair_scale_global)

        # Transform bald to global
        bald_pos, bald_rot, bald_scale = self.rigging.to_global(
            target_bald_local.positions,
            target_bald_local.rotations,
            target_bald_local._scales,
            target_bald_local.binding,
            vertices, faces,
        )

        G_bald_global = target_bald_local.clone()
        G_bald_global._positions = torch.nn.Parameter(bald_pos)
        G_bald_global._rotations = torch.nn.Parameter(bald_rot)
        G_bald_global._scales = torch.nn.Parameter(bald_scale)

        # Combine
        G_combined = G_transferred.merge(G_bald_global)

        logger.info(
            f"✓ Transfer complete: {G_transferred.num_gaussians} hair + "
            f"{G_bald_global.num_gaussians} bald = {G_combined.num_gaussians} total"
        )

        return {
            "G_hair_transferred": G_transferred,
            "G_combined": G_combined,
        }
