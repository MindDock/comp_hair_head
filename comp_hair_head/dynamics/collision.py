"""Proxy-based collision constraints for cage-PBD simulation.

Implements the paper's novel collision handling strategy where proxy
particles are used instead of direct cage vertex collision detection,
preventing excessive hair-skin separation caused by cage-to-Gaussian gaps.
"""

from __future__ import annotations

import torch
import numpy as np

from ..utils.logging import get_logger

logger = get_logger("dynamics.collision")


class ProxyCollisionHandler:
    """Proxy-based collision constraint handler.

    Instead of detecting collisions between cage vertices and the FLAME mesh
    directly (which would cause excessive separation due to cage-Gaussian gaps),
    we compute proxy particles that represent the actual Gaussian positions
    under deformation, and apply collision constraints to those proxies.

    Pipeline:
        1. During init: assign each cage vertex its nearest Gaussian's index
        2. At each step: compute proxy = MVC interpolation of predicted cage positions
        3. Test proxy against FLAME mesh surface
        4. If penetrating, correct the cage vertex position
    """

    def __init__(self, collision_margin: float = 0.001):
        self.collision_margin = collision_margin
        self.proxy_gaussian_indices: torch.Tensor | None = None
        self.proxy_mvc_weights: torch.Tensor | None = None

    def initialize(
        self,
        cage_vertices: torch.Tensor,
        gaussian_positions: torch.Tensor,
        gaussian_mvc_weights: torch.Tensor,
    ) -> None:
        """Initialize proxy assignments.

        For each cage vertex, find its nearest Gaussian and store the
        Gaussian's MVC weights as the proxy weights.

        Args:
            cage_vertices: (M, 3) cage vertex positions.
            gaussian_positions: (N, 3) hair Gaussian centers.
            gaussian_mvc_weights: (N, M) MVC weights of Gaussians w.r.t. cage.
                Note: these are the CENTER weights only (not all 7 endpoints).
        """
        # For each cage vertex, find nearest Gaussian
        dists = torch.cdist(cage_vertices, gaussian_positions)  # (M, N)
        nearest = dists.argmin(dim=1)  # (M,)
        self.proxy_gaussian_indices = nearest

        # Store the MVC weights for each cage vertex's proxy Gaussian
        # proxy_mvc_weights[m] = gaussian_mvc_weights[nearest[m]]
        self.proxy_mvc_weights = gaussian_mvc_weights[nearest]  # (M, M)

        logger.info(f"Proxy collision initialized for {cage_vertices.shape[0]} vertices")

    def compute_proxy_positions(
        self,
        predicted_cage_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute proxy particle positions from predicted cage positions.

        proxy_m = sum_j (proxy_mvc_weights[m, j] * predicted_cage_vertices[j])

        Args:
            predicted_cage_vertices: (M, 3) predicted cage positions.

        Returns:
            proxy_positions: (M, 3) proxy particle positions.
        """
        assert self.proxy_mvc_weights is not None, "Call initialize() first"
        return torch.matmul(self.proxy_mvc_weights, predicted_cage_vertices)

    def resolve_collisions(
        self,
        predicted_cage_vertices: torch.Tensor,
        mesh_vertices: torch.Tensor,
        mesh_faces: torch.Tensor,
        inv_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve collisions using proxy-based detection.

        1. Compute proxy positions from predicted cage vertices
        2. Check if proxies penetrate the FLAME mesh
        3. If penetrating, push the cage vertex outward along the surface normal

        Args:
            predicted_cage_vertices: (M, 3) current predicted positions.
            mesh_vertices: (V, 3) FLAME mesh vertices.
            mesh_faces: (F, 3) FLAME mesh face indices.
            inv_mass: (M,) inverse mass (0 for kinematic).

        Returns:
            corrected_cage_vertices: (M, 3) collision-corrected positions.
        """
        corrected = predicted_cage_vertices.clone()

        # Compute proxy positions
        proxy_pos = self.compute_proxy_positions(predicted_cage_vertices)

        # Find closest point on mesh for each proxy
        closest_points, closest_normals, signed_dists = self._closest_point_on_mesh(
            proxy_pos, mesh_vertices, mesh_faces
        )

        # Detect penetration: proxy inside mesh (negative signed distance)
        penetrating = signed_dists < self.collision_margin

        if penetrating.any():
            # Correction: push cage vertex along surface normal
            # The correction is applied to the cage vertex, not the proxy
            correction_magnitude = self.collision_margin - signed_dists[penetrating]
            correction = closest_normals[penetrating] * correction_magnitude.unsqueeze(-1)

            # Only correct free (non-kinematic) vertices
            pen_indices = penetrating.nonzero(as_tuple=True)[0]
            for idx, pen_idx in enumerate(pen_indices):
                if inv_mass[pen_idx] > 0:
                    corrected[pen_idx] += correction[idx]

        return corrected

    def _closest_point_on_mesh(
        self,
        query_points: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find closest point on mesh surface for each query point.

        Args:
            query_points: (P, 3) query positions.
            vertices: (V, 3) mesh vertices.
            faces: (F, 3) face indices.

        Returns:
            closest_points: (P, 3) nearest surface points.
            normals: (P, 3) surface normals at closest points.
            signed_distances: (P,) signed distances (negative = inside).
        """
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
        )

        pts = query_points.detach().cpu().numpy()
        closest, distances, face_ids = trimesh.proximity.closest_point(mesh, pts)

        # Compute normals
        face_normals = mesh.face_normals[face_ids]

        # Signed distance
        direction = pts - closest
        signs = np.sign(np.sum(direction * face_normals, axis=-1))
        signs[signs == 0] = 1.0
        signed_dists = distances * signs

        device = query_points.device
        return (
            torch.tensor(closest, dtype=torch.float32, device=device),
            torch.tensor(face_normals, dtype=torch.float32, device=device),
            torch.tensor(signed_dists, dtype=torch.float32, device=device),
        )
