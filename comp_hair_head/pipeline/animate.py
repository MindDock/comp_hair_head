"""Animation driving pipeline.

Drives the reconstructed avatar with new FLAME parameters,
applying Cage-PBD hair dynamics simulation.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from ..config import CompHairHeadConfig, load_config
from ..gaussian.model import GaussianModel
from ..gaussian.renderer import GaussianRenderer, RenderOutput
from ..flame.flame_model import FLAMEModel
from ..flame.rigging import GaussianRigging
from ..dynamics.cage_builder import CageBuilder, compute_rest_lengths
from ..dynamics.mvc import compute_gaussian_mvc_weights
from ..dynamics.pbd_solver import PBDSolver, PBDSolverPyTorch
from ..dynamics.collision import ProxyCollisionHandler
from ..dynamics.hair_deform import HairDeformer
from ..utils.logging import get_logger

logger = get_logger("pipeline.animate")


class AnimationPipeline:
    """Drive reconstructed avatar with expression/pose sequences.

    Combines:
    - FLAME mesh deformation for face
    - Cage-PBD simulation for hair dynamics
    - Composited rendering of both components
    """

    def __init__(self, config: CompHairHeadConfig | None = None):
        self.config = config or load_config()
        self.device = self.config.get_device()

        self.rigging = GaussianRigging()
        self.renderer = GaussianRenderer(
            width=self.config.render.width,
            height=self.config.render.height,
            background_color=self.config.render.background_color,
            device=self.device,
        ).to(self.device)

        # Will be initialized in setup()
        self.cage_data: dict | None = None
        self.pbd_solver = None
        self.hair_deformer: HairDeformer | None = None
        self.collision_handler: ProxyCollisionHandler | None = None
        self.mvc_weights: torch.Tensor | None = None

    def setup(
        self,
        G_hair_local: GaussianModel,
        G_bald_local: GaussianModel,
        flame_model: FLAMEModel,
        flame_params: dict[str, torch.Tensor],
    ) -> None:
        """Initialize animation components from reconstruction results.

        Args:
            G_hair_local: Hair Gaussians in local coordinates.
            G_bald_local: Bald Gaussians in local coordinates.
            flame_model: FLAME model instance.
            flame_params: FLAME parameter dict.
        """
        self.G_hair_local = G_hair_local
        self.G_bald_local = G_bald_local
        self.flame_model = flame_model
        self.flame_params = flame_params

        # ── Get reference mesh and hair in global coords ─────────────────
        ref_output = flame_model(
            shape_params=flame_params["shape"],
            expression_params=flame_params["expression"],
        )
        ref_vertices = ref_output["vertices"].squeeze(0)
        faces = ref_output["faces"]

        # Transform hair to global for cage construction
        hair_global_pos, _, _ = self.rigging.to_global(
            G_hair_local.positions, G_hair_local.rotations, G_hair_local._scales,
            G_hair_local.binding, ref_vertices, faces,
        )

        # ── Build cage ───────────────────────────────────────────────────
        logger.info("Building hair cage...")
        cage_builder = CageBuilder(
            voxel_resolution=self.config.cage.voxel_resolution,
            target_vertices=self.config.cage.target_vertices,
        )

        # Get scalp vertices for kinematic identification
        scalp_vertices = ref_vertices  # Simplified: use all vertices

        self.cage_data = cage_builder.build(
            hair_positions=hair_global_pos,
            scalp_vertices=scalp_vertices,
        )

        # ── Compute MVC weights ──────────────────────────────────────────
        logger.info("Computing MVC weights...")
        # Get 7-point endpoints for hair Gaussians (in global space)
        # We need a temporary global GaussianModel for this
        G_hair_global = G_hair_local.clone()
        G_hair_global._positions = torch.nn.Parameter(hair_global_pos)

        endpoints = G_hair_global.get_ellipsoid_endpoints()  # (N, 7, 3)
        self.mvc_weights = compute_gaussian_mvc_weights(
            endpoints,
            self.cage_data["vertices"],
            self.cage_data["faces"],
        )

        # ── Initialize PBD solver ────────────────────────────────────────
        logger.info("Initializing PBD solver...")
        rest_lengths = compute_rest_lengths(
            self.cage_data["vertices"], self.cage_data["edges"]
        )

        try:
            self.pbd_solver = PBDSolver(
                num_vertices=self.cage_data["vertices"].shape[0],
                num_edges=self.cage_data["edges"].shape[0],
                num_faces=self.cage_data["faces"].shape[0],
                dt=self.config.pbd.dt,
                gravity=tuple(self.config.pbd.gravity),
                num_iterations=self.config.pbd.num_iterations,
                stretch_compliance=self.config.pbd.stretch_compliance,
                damping=self.config.pbd.damping,
                arch=self.config.get_taichi_arch(),
            )
            self.pbd_solver.initialize(
                self.cage_data["vertices"],
                self.cage_data["edges"],
                self.cage_data["faces"],
                rest_lengths,
                self.cage_data["is_kinematic"],
            )
        except Exception as e:
            logger.warning(f"Taichi PBD failed: {e}. Using PyTorch fallback.")
            self.pbd_solver = PBDSolverPyTorch(
                dt=self.config.pbd.dt,
                gravity=tuple(self.config.pbd.gravity),
                num_iterations=self.config.pbd.num_iterations,
                stretch_compliance=self.config.pbd.stretch_compliance,
                damping=self.config.pbd.damping,
            )
            self.pbd_solver.initialize(
                self.cage_data["vertices"],
                self.cage_data["edges"],
                self.cage_data["faces"],
                rest_lengths,
                self.cage_data["is_kinematic"],
            )

        # ── Initialize hair deformer ─────────────────────────────────────
        self.hair_deformer = HairDeformer()
        self.hair_deformer.initialize(
            source_endpoints=endpoints,
            mvc_weights=self.mvc_weights,
            source_scales=G_hair_local.scales,
        )

        # ── Initialize collision handler ─────────────────────────────────
        self.collision_handler = ProxyCollisionHandler(
            collision_margin=self.config.pbd.collision_margin
        )

        # Center MVC weights for proxy
        center_weights = self.mvc_weights[:, 0, :]  # (N, M) - center point weights
        cage_center_mvc = center_weights  # (N, M)

        # For proxy: we need (M, M) weights mapping cage verts → proxy positions
        # Each cage vertex's proxy is its nearest Gaussian
        self.collision_handler.initialize(
            cage_vertices=self.cage_data["vertices"],
            gaussian_positions=hair_global_pos,
            gaussian_mvc_weights=cage_center_mvc,
        )

        logger.info("✓ Animation pipeline ready")

    def animate_frame(
        self,
        expression: torch.Tensor | None = None,
        pose: torch.Tensor | None = None,
        global_rotation: torch.Tensor | None = None,
        global_translation: torch.Tensor | None = None,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
    ) -> RenderOutput:
        """Render a single animation frame.

        Args:
            expression: (50,) FLAME expression params.
            pose: (15,) FLAME pose params.
            global_rotation: (3,) global rotation.
            global_translation: (3,) global translation.
            view_matrix: (4, 4) camera view matrix.
            proj_matrix: (4, 4) projection matrix.

        Returns:
            RenderOutput with composited image.
        """
        assert self.pbd_solver is not None, "Call setup() first"

        # Get deformed FLAME mesh
        flame_output = self.flame_model(
            shape_params=self.flame_params["shape"],
            expression_params=expression,
            pose_params=pose,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        vertices = flame_output["vertices"].squeeze(0)
        faces = flame_output["faces"]

        # ── Deform bald Gaussians via rigging ────────────────────────────
        bald_pos, bald_rot, bald_scale = self.rigging.to_global(
            self.G_bald_local.positions,
            self.G_bald_local.rotations,
            self.G_bald_local._scales,
            self.G_bald_local.binding,
            vertices, faces,
        )

        # ── Simulate hair via Cage-PBD ───────────────────────────────────
        # Update kinematic cage vertices (roots follow LBS)
        # Simplified: use current cage positions
        deformed_cage = self.pbd_solver.step()

        # Deform hair Gaussians from cage
        hair_pos, hair_rot, hair_scale = self.hair_deformer.deform(
            deformed_cage,
            self.G_hair_local.rotations,
            self.G_hair_local._scales,
        )

        # ── Composited rendering ─────────────────────────────────────────
        all_positions = torch.cat([bald_pos, hair_pos])
        all_rotations = torch.cat([bald_rot, hair_rot])
        all_scales = torch.cat([
            torch.exp(bald_scale), torch.exp(hair_scale)
        ])
        all_opacities = torch.cat([
            self.G_bald_local.opacities, self.G_hair_local.opacities
        ])
        all_sh = torch.cat([
            self.G_bald_local.sh_coeffs, self.G_hair_local.sh_coeffs
        ])

        output = self.renderer(
            positions=all_positions,
            rotations=all_rotations,
            scales=all_scales,
            opacities=all_opacities,
            sh_coeffs=all_sh,
            view_matrix=view_matrix,
            proj_matrix=proj_matrix,
            render_color=True,
        )

        return output

    def animate_sequence(
        self,
        expression_sequence: torch.Tensor | None = None,
        pose_sequence: torch.Tensor | None = None,
        view_matrix: torch.Tensor | None = None,
        proj_matrix: torch.Tensor | None = None,
    ) -> list[RenderOutput]:
        """Render a sequence of animation frames.

        Args:
            expression_sequence: (T, 50) expression params per frame.
            pose_sequence: (T, 15) pose params per frame.
            view_matrix: (4, 4) fixed camera view.
            proj_matrix: (4, 4) fixed projection.

        Returns:
            List of RenderOutputs.
        """
        T = 0
        if expression_sequence is not None:
            T = expression_sequence.shape[0]
        elif pose_sequence is not None:
            T = pose_sequence.shape[0]

        outputs = []
        for t in tqdm(range(T), desc="Animating"):
            expr = expression_sequence[t] if expression_sequence is not None else None
            pose = pose_sequence[t] if pose_sequence is not None else None

            output = self.animate_frame(
                expression=expr,
                pose=pose,
                view_matrix=view_matrix,
                proj_matrix=proj_matrix,
            )
            outputs.append(output)

        return outputs
