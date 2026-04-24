"""Geometry utility functions: rotations, transforms, and mesh helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


# ── Quaternion utilities ─────────────────────────────────────────────────────

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        q: Quaternions of shape (..., 4).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)

    B = q.shape[:-1]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*B, 3, 3)

    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Args:
        R: Rotation matrices of shape (..., 3, 3).

    Returns:
        Quaternions of shape (..., 4).
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Case 2: R[0,0] is max diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is max diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is max diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    q = F.normalize(q, p=2, dim=-1)
    return q.reshape(*batch_shape, 4)


# ── Axis-angle ↔ Matrix ─────────────────────────────────────────────────────

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix via Rodrigues' formula.

    Args:
        axis_angle: Shape (..., 3).

    Returns:
        Rotation matrix of shape (..., 3, 3).
    """
    angle = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # (..., 1)
    axis = axis_angle / (angle + 1e-8)  # (..., 3)

    cos_a = torch.cos(angle).unsqueeze(-1)  # (..., 1, 1)
    sin_a = torch.sin(angle).unsqueeze(-1)  # (..., 1, 1)

    # Skew-symmetric matrix K
    kx, ky, kz = axis.unbind(-1)
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        zeros, -kz, ky,
        kz, zeros, -kx,
        -ky, kx, zeros,
    ], dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)

    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    I = I.expand_as(K)

    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    return R


# ── Triangle local coordinate system ────────────────────────────────────────

def compute_triangle_local_frame(
    v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute local coordinate frame for a triangle.

    Following GaussianAvatars convention:
      - t (translation): centroid of the triangle
      - R (rotation): orthonormal frame from edges
      - eta (scale): sqrt of triangle area

    Args:
        v0, v1, v2: Triangle vertices, each shape (..., 3).

    Returns:
        t: Centroid (..., 3)
        R: Rotation matrix (..., 3, 3)
        eta: Scale factor (..., 1)
    """
    # Centroid
    t = (v0 + v1 + v2) / 3.0

    # Edge vectors
    e1 = v1 - v0  # (..., 3)
    e2 = v2 - v0  # (..., 3)

    # Normal
    normal = torch.cross(e1, e2, dim=-1)
    area = torch.norm(normal, dim=-1, keepdim=True)
    normal = normal / (area + 1e-8)

    # Tangent (e1 normalized)
    tangent = F.normalize(e1, dim=-1)

    # Bitangent
    bitangent = torch.cross(normal, tangent, dim=-1)
    bitangent = F.normalize(bitangent, dim=-1)

    # Rotation matrix: columns are tangent, bitangent, normal
    R = torch.stack([tangent, bitangent, normal], dim=-1)  # (..., 3, 3)

    # Scale: sqrt of area
    eta = torch.sqrt(area / 2.0 + 1e-8)

    return t, R, eta


# ── Point-in-mesh test ───────────────────────────────────────────────────────

def signed_distance_to_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Approximate signed distance from points to a triangle mesh.

    Uses trimesh for the actual computation (CPU).

    Args:
        points: (N, 3) query points.
        vertices: (V, 3) mesh vertices.
        faces: (F, 3) mesh face indices.

    Returns:
        Signed distances (N,). Negative means inside.
    """
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
    )
    pts_np = points.detach().cpu().numpy()
    # Use proximity query
    closest, dist, face_id = trimesh.proximity.closest_point(mesh, pts_np)
    # Determine sign
    normals = mesh.face_normals[face_id]
    direction = pts_np - closest
    sign = np.sign(np.sum(direction * normals, axis=-1))
    sign[sign == 0] = 1.0

    signed_dist = dist * sign
    return torch.tensor(signed_dist, device=points.device, dtype=points.dtype)


# ── Camera utilities ─────────────────────────────────────────────────────────

def create_camera_poses_on_sphere(
    num_views: int,
    radius: float = 2.5,
    elevation_range: tuple[float, float] = (-30.0, 30.0),
    device: str = "cpu",
) -> torch.Tensor:
    """Generate camera poses uniformly distributed on a sphere.

    Args:
        num_views: Number of viewpoints K.
        radius: Distance from origin.
        elevation_range: (min, max) elevation in degrees.
        device: Torch device.

    Returns:
        Camera extrinsic matrices (K, 4, 4), world-to-camera.
    """
    poses = []
    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        t = i / max(num_views - 1, 1)
        elev_deg = elevation_range[0] + t * (elevation_range[1] - elevation_range[0])
        elev = np.radians(elev_deg)

        # Camera position
        x = radius * np.cos(elev) * np.sin(azimuth)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.cos(azimuth)
        cam_pos = np.array([x, y, z])

        # Look-at matrix
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)  # 3x3
        t_vec = -R.T @ cam_pos

        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, 3] = t_vec
        poses.append(pose)

    return torch.tensor(np.stack(poses), dtype=torch.float32, device=device)


def perspective_projection_matrix(
    fov_deg: float, aspect: float, near: float = 0.01, far: float = 100.0
) -> torch.Tensor:
    """Create a perspective projection matrix.

    Args:
        fov_deg: Vertical field of view in degrees.
        aspect: Width / height.
        near: Near clip plane.
        far: Far clip plane.

    Returns:
        Projection matrix (4, 4).
    """
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    P = torch.zeros(4, 4)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1.0
    return P
