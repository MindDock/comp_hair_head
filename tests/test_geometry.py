"""Unit tests for geometry utilities."""

import torch
import pytest
from comp_hair_head.utils.geometry import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
    axis_angle_to_matrix,
    compute_triangle_local_frame,
    create_camera_poses_on_sphere,
)


class TestQuaternion:
    def test_identity_quaternion_to_matrix(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_matrix(q)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_roundtrip_quaternion_matrix(self):
        # Random rotation
        q = torch.randn(4)
        q = q / q.norm()
        R = quaternion_to_matrix(q)
        q_back = matrix_to_quaternion(R)
        # Quaternions can differ by sign
        assert torch.allclose(q.abs(), q_back.abs(), atol=1e-4)

    def test_batch_quaternion(self):
        q = torch.randn(10, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_matrix(q)
        assert R.shape == (10, 3, 3)

    def test_rotation_matrix_orthogonality(self):
        q = torch.randn(5, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = quaternion_to_matrix(q)
        I = torch.eye(3).unsqueeze(0).expand(5, -1, -1)
        RtR = R.transpose(-1, -2) @ R
        assert torch.allclose(RtR, I, atol=1e-5)

    def test_quaternion_multiply_identity(self):
        q = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q2 = torch.randn(4)
        q2 = q2 / q2.norm()
        result = quaternion_multiply(q, q2)
        assert torch.allclose(result, q2, atol=1e-6)


class TestAxisAngle:
    def test_zero_rotation(self):
        aa = torch.zeros(3)
        R = axis_angle_to_matrix(aa)
        assert torch.allclose(R, torch.eye(3), atol=1e-6)

    def test_90deg_z_rotation(self):
        import math
        aa = torch.tensor([0.0, 0.0, math.pi / 2])
        R = axis_angle_to_matrix(aa)
        # Should map x→y, y→-x
        x = torch.tensor([1.0, 0.0, 0.0])
        rotated = R @ x
        assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


class TestTriangleFrame:
    def test_simple_triangle(self):
        v0 = torch.tensor([0.0, 0.0, 0.0])
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])

        t, R, eta = compute_triangle_local_frame(v0, v1, v2)

        # Centroid should be (1/3, 1/3, 0)
        assert torch.allclose(t, torch.tensor([1/3, 1/3, 0.0]), atol=1e-5)

        # R should be orthogonal
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


class TestCameraPoses:
    def test_num_views(self):
        poses = create_camera_poses_on_sphere(8)
        assert poses.shape == (8, 4, 4)

    def test_valid_transform(self):
        poses = create_camera_poses_on_sphere(4)
        for i in range(4):
            # Check it's a valid rigid transform
            R = poses[i, :3, :3]
            assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
