"""Cage construction: voxelization → watertight mesh for PBD simulation.

Constructs a coarse control cage (< 500 vertices) around hair Gaussians
for efficient physics-based simulation via PBD.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from ..utils.logging import get_logger

logger = get_logger("dynamics.cage_builder")


class CageBuilder:
    """Build a watertight cage mesh from a hair Gaussian point cloud.

    Pipeline:
        1. Voxelize the point cloud
        2. Extract isosurface via Marching Cubes
        3. Simplify to target vertex count
        4. Identify kinematic (root) vertices near the scalp
    """

    def __init__(
        self,
        voxel_resolution: int = 64,
        target_vertices: int = 400,
        padding: float = 0.05,
    ):
        self.voxel_resolution = voxel_resolution
        self.target_vertices = target_vertices
        self.padding = padding

    def build(
        self,
        hair_positions: torch.Tensor,
        scalp_vertices: torch.Tensor | None = None,
        scalp_threshold: float = 0.02,
    ) -> dict[str, torch.Tensor]:
        """Build cage mesh from hair point cloud.

        Args:
            hair_positions: (N, 3) hair Gaussian centers.
            scalp_vertices: (S, 3) optional scalp mesh vertices for
                identifying kinematic root vertices.
            scalp_threshold: Distance threshold for root classification.

        Returns:
            Dict with:
                - "vertices": (M, 3) cage vertex positions
                - "faces": (F, 3) cage face indices
                - "is_kinematic": (M,) bool, True for root/kinematic vertices
                - "edges": (E, 2) edge connectivity
        """
        pts = hair_positions.detach().cpu().numpy()

        # Step 1: Voxelize
        voxel_grid, origin, voxel_size = self._voxelize(pts)

        # Step 2: Marching cubes
        try:
            from skimage.measure import marching_cubes
            verts, faces, normals, _ = marching_cubes(
                voxel_grid, level=0.5, spacing=(voxel_size,) * 3
            )
            verts = verts + origin
        except Exception as e:
            logger.warning(f"Marching cubes failed: {e}. Using convex hull fallback.")
            hull = trimesh.convex.convex_hull(pts)
            verts = hull.vertices
            faces = hull.faces

        # Step 3: Create trimesh and simplify
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh = self._ensure_watertight(mesh)

        # Simplify to target vertex count
        if mesh.vertices.shape[0] > self.target_vertices:
            mesh = self._simplify_mesh(mesh)

        logger.info(
            f"Cage built: {mesh.vertices.shape[0]} vertices, "
            f"{mesh.faces.shape[0]} faces"
        )

        # Step 4: Identify kinematic vertices
        cage_verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        cage_faces = torch.tensor(mesh.faces, dtype=torch.long)

        is_kinematic = torch.zeros(cage_verts.shape[0], dtype=torch.bool)
        if scalp_vertices is not None:
            scalp_pts = scalp_vertices.detach().cpu()
            # Find cage vertices close to scalp
            dists = torch.cdist(cage_verts, scalp_pts)
            min_dists = dists.min(dim=1).values
            is_kinematic = min_dists < scalp_threshold

        # Extract edges
        edges = self._extract_edges(mesh)

        return {
            "vertices": cage_verts,
            "faces": cage_faces,
            "is_kinematic": is_kinematic,
            "edges": torch.tensor(edges, dtype=torch.long),
        }

    def _voxelize(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Voxelize a point cloud into a binary occupancy grid.

        Args:
            points: (N, 3) point positions.

        Returns:
            grid: (R, R, R) float occupancy grid.
            origin: (3,) grid origin.
            voxel_size: Size of each voxel.
        """
        # Bounding box with padding
        min_pt = points.min(axis=0) - self.padding
        max_pt = points.max(axis=0) + self.padding
        extent = max_pt - min_pt
        voxel_size = extent.max() / self.voxel_resolution

        R = self.voxel_resolution
        grid = np.zeros((R, R, R), dtype=np.float32)

        # Map points to voxel indices
        indices = ((points - min_pt) / voxel_size).astype(int)
        indices = np.clip(indices, 0, R - 1)

        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0

        # Dilate to fill gaps
        from scipy.ndimage import binary_dilation
        struct = np.ones((3, 3, 3))
        grid = binary_dilation(grid, structure=struct, iterations=2).astype(np.float32)

        # Smooth for marching cubes
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=1.0)

        return grid, min_pt, voxel_size

    def _ensure_watertight(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Ensure the mesh is watertight."""
        if not mesh.is_watertight:
            # Fill holes
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_winding(mesh)

        return mesh

    def _simplify_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Simplify mesh to target vertex count.

        Uses quadric decimation via open3d if available.
        """
        try:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices),
                triangles=o3d.utility.Vector3iVector(mesh.faces),
            )
            target_faces = max(self.target_vertices * 2, 100)
            simplified = o3d_mesh.simplify_quadric_decimation(target_faces)

            result = trimesh.Trimesh(
                vertices=np.asarray(simplified.vertices),
                faces=np.asarray(simplified.triangles),
            )
            return result
        except ImportError:
            logger.warning("open3d not available for mesh simplification")
            return mesh

    def _extract_edges(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Extract unique edges from mesh faces.

        Returns:
            (E, 2) array of vertex index pairs.
        """
        edges = mesh.edges_unique
        return edges


def compute_rest_lengths(
    vertices: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Compute rest edge lengths for stretch constraints.

    Args:
        vertices: (M, 3) cage vertices.
        edges: (E, 2) edge indices.

    Returns:
        rest_lengths: (E,) rest lengths.
    """
    v0 = vertices[edges[:, 0]]
    v1 = vertices[edges[:, 1]]
    return torch.norm(v1 - v0, dim=-1)
