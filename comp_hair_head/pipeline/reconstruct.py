"""End-to-end reconstruction pipeline.

Implements the full reconstruction workflow from a single input image
to a compositional {G'_hair, G'_bald, θ_head} representation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from ..config import CompHairHeadConfig, load_config
from ..gaussian.model import GaussianModel
from ..gaussian.renderer import GaussianRenderer
from ..flame.flame_model import FLAMEModel
from ..flame.rigging import GaussianRigging, compute_binding_indices
from ..preprocessing.bald_filter import BaldFilter
from ..preprocessing.face_lift import FaceLiftWrapper
from ..segmentation.hair_seg import HairSegmentor
from ..segmentation.learnable_feat import optimize_hair_features
from ..utils.geometry import create_camera_poses_on_sphere
from ..utils.logging import get_logger
from ..utils import io as uio

logger = get_logger("pipeline.reconstruct")


class ReconstructionPipeline:
    """Full reconstruction pipeline from single image to 3D avatar.

    Pipeline stages:
    1. Preprocess: apply bald filter to get hairless image
    2. Lift: convert both images to 3DGS (via FaceLift)
    3. Segment: extract hair Gaussians from head 3DGS
    4. Register: bind bald Gaussians to FLAME via non-rigid registration
    5. Assemble: combine hair + bald with collision optimization
    """

    def __init__(self, config: CompHairHeadConfig | None = None):
        self.config = config or load_config()
        self.device = self.config.get_device()

        # Initialize submodules
        self.bald_filter = BaldFilter(device=self.device)
        self.face_lift = FaceLiftWrapper(device=self.device)
        self.flame = FLAMEModel(
            model_path=self.config.flame.model_path,
            num_shape_params=self.config.flame.num_shape_params,
            num_expression_params=self.config.flame.num_expression_params,
            device=self.device,
        ).to(self.device)
        self.renderer = GaussianRenderer(
            width=self.config.render.width,
            height=self.config.render.height,
            background_color=self.config.render.background_color,
            sh_degree=self.config.gaussian.sh_degree,
            device=self.device,
        ).to(self.device)
        self.segmentor = HairSegmentor(
            checkpoint_path=self.config.segmentation.sam2_checkpoint,
            device=self.device,
        )
        self.rigging = GaussianRigging()

        logger.info(f"ReconstructionPipeline initialized on device={self.device}")

    def reconstruct(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> dict:
        """Run full reconstruction from a single image.

        Args:
            image_path: Path to frontal portrait image.
            output_dir: Optional output directory.

        Returns:
            Dict with:
                - "G_hair_local": GaussianModel (hair in local coords)
                - "G_bald_local": GaussianModel (bald in local coords)
                - "flame_params": dict of FLAME parameters
                - "flame_model": FLAMEModel instance
        """
        output_dir = Path(output_dir) if output_dir else Path(self.config.output.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ─── Stage 1: Load and preprocess ────────────────────────────────
        logger.info("Stage 1: Preprocessing...")
        image = uio.load_image(image_path, size=self.config.preprocess.image_size)
        image = image.to(self.device)

        bald_image = self.bald_filter.process(image)
        uio.save_image(bald_image, output_dir / "bald_image.png")

        # ─── Stage 2: Image-to-3DGS lifting ─────────────────────────────
        logger.info("Stage 2: Lifting images to 3DGS...")
        head_result = self.face_lift.lift(image, sh_degree=self.config.gaussian.sh_degree)
        bald_result = self.face_lift.lift(bald_image, sh_degree=self.config.gaussian.sh_degree)

        G_head = head_result["gaussian_model"]
        G_bald = bald_result["gaussian_model"]
        flame_params = head_result["flame_params"]

        logger.info(f"  G_head: {G_head.num_gaussians} Gaussians")
        logger.info(f"  G_bald: {G_bald.num_gaussians} Gaussians")

        # ─── Stage 3: Hair segmentation ──────────────────────────────────
        logger.info("Stage 3: Hair segmentation...")

        # Generate multi-view supervision
        K = self.config.preprocess.num_views
        camera_poses = create_camera_poses_on_sphere(K, device=self.device)

        views = []
        for k in range(K):
            view_matrix = camera_poses[k]
            output = self.renderer(
                positions=G_head.positions,
                rotations=G_head.rotations,
                scales=G_head.scales,
                opacities=G_head.opacities,
                sh_coeffs=G_head.sh_coeffs,
                view_matrix=view_matrix,
                render_color=True,
            )
            views.append({
                "image": output.color,
                "view_matrix": view_matrix,
            })

        hair_mask = optimize_hair_features(
            G_head, self.renderer, self.segmentor, views,
            num_iterations=self.config.segmentation.seg_iterations,
            lr=self.config.segmentation.seg_lr,
            device=self.device,
        )

        G_hair = G_head.filter_by_mask(hair_mask)
        logger.info(f"  G_hair: {G_hair.num_gaussians} Gaussians")

        # ─── Stage 4: Bind bald Gaussians to FLAME ──────────────────────
        logger.info("Stage 4: Rigging bald Gaussians to FLAME...")

        flame_output = self.flame(
            shape_params=flame_params["shape"],
            expression_params=flame_params["expression"],
        )
        vertices = flame_output["vertices"].squeeze(0)  # (V, 3)
        faces = flame_output["faces"]  # (F, 3)

        binding = compute_binding_indices(
            G_bald.positions, vertices, faces
        )
        G_bald.binding = binding

        # Transform to local coordinates
        pos_local, rot_local, scale_local = self.rigging.to_local(
            G_bald.positions, G_bald.rotations, G_bald._scales,
            binding, vertices, faces,
        )

        G_bald_local = G_bald.clone()
        G_bald_local._positions = torch.nn.Parameter(pos_local)
        G_bald_local._rotations = torch.nn.Parameter(rot_local)
        G_bald_local._scales = torch.nn.Parameter(scale_local)

        # ─── Stage 5: Component assembly ─────────────────────────────────
        logger.info("Stage 5: Component assembly...")

        # Bind hair to nearest scalp triangle for hairstyle transfer
        hair_binding = compute_binding_indices(
            G_hair.positions, vertices, faces
        )
        G_hair.binding = hair_binding

        hair_pos_local, hair_rot_local, hair_scale_local = self.rigging.to_local(
            G_hair.positions, G_hair.rotations, G_hair._scales,
            hair_binding, vertices, faces,
        )

        G_hair_local = G_hair.clone()
        G_hair_local._positions = torch.nn.Parameter(hair_pos_local)
        G_hair_local._rotations = torch.nn.Parameter(hair_rot_local)
        G_hair_local._scales = torch.nn.Parameter(hair_scale_local)

        # Save results
        logger.info("Saving reconstruction results...")
        torch.save({
            "G_hair_local": G_hair_local.state_dict(),
            "G_bald_local": G_bald_local.state_dict(),
            "flame_params": flame_params,
            "hair_binding": hair_binding,
            "bald_binding": binding,
        }, output_dir / "avatar.pt")

        logger.info("✓ Reconstruction complete!")

        return {
            "G_hair_local": G_hair_local,
            "G_bald_local": G_bald_local,
            "flame_params": flame_params,
            "flame_model": self.flame,
        }
