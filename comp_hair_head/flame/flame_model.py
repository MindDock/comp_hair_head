"""FLAME parametric head model.

Implements the FLAME (Faces Learned with an Articulated Model and Expressions)
head model for mesh generation, deformation, and linear blend skinning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logging import get_logger
from ..utils.geometry import axis_angle_to_matrix

logger = get_logger("flame.model")


def _load_flame_data(path: str | Path) -> dict:
    """Load FLAME model data from .npz or .pkl file.

    Prefers .npz format (no chumpy dependency). Falls back to .pkl
    with a custom unpickler that handles chumpy objects.
    """
    path = Path(path)

    if path.suffix == ".npz":
        return dict(np.load(str(path)))

    import pickle

    class _FlameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return _ChumpyNumpy
            if module in ("scipy.sparse.csc", "scipy.sparse._csc"):
                from scipy.sparse import csc_matrix
                return csc_matrix
            if module in ("scipy.sparse.csr", "scipy.sparse._csr"):
                from scipy.sparse import csr_matrix
                return csr_matrix
            return super().find_class(module, name)

    with open(path, "rb") as f:
        data = _FlameUnpickler(f, encoding="latin1", errors="ignore").load()
    return _deep_to_numpy(data)


class _ChumpyNumpy:
    """Placeholder that converts chumpy objects to numpy during unpickling."""

    def __init__(self, *args, **kwargs):
        self._data = None
        if args:
            obj = args[0]
            if isinstance(obj, np.ndarray):
                self._data = obj
            elif isinstance(obj, _ChumpyNumpy):
                self._data = obj._data
            elif hasattr(obj, "r"):
                try:
                    val = obj.r
                    self._data = np.array(val, dtype=np.float64) if isinstance(val, np.ndarray) else None
                except Exception:
                    self._data = None
            else:
                try:
                    self._data = np.array(obj, dtype=np.float64)
                except Exception:
                    self._data = None

    def __array__(self, dtype=None):
        if self._data is not None:
            return self._data if dtype is None else self._data.astype(dtype)
        return np.array([], dtype=dtype or np.float64)

    @property
    def r(self):
        return self._data if self._data is not None else np.array([])

    @property
    def shape(self):
        return self._data.shape if self._data is not None else (0,)

    @property
    def T(self):
        return self._data.T if self._data is not None else None

    def __reduce__(self):
        return (np.array, (self._data if self._data is not None else np.array([]),))


def _deep_to_numpy(obj):
    """Recursively convert chumpy-like objects to numpy arrays."""
    if isinstance(obj, _ChumpyNumpy):
        return obj._data if obj._data is not None else np.array([])
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: _deep_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_deep_to_numpy(v) for v in obj]
        return type(obj)(converted)
    if hasattr(obj, "toarray"):
        try:
            return obj.toarray().astype(np.float64)
        except Exception:
            pass
    return obj


class FLAMEModel(nn.Module):
    """FLAME parametric head model.

    Given shape (β), expression (ψ), and pose (θ) parameters,
    produces a deformed 3D head mesh with vertices and faces.

    The model supports:
    - Shape variations via shape blend shapes
    - Expression deformations via expression blend shapes
    - Articulated motion via joint rotations and LBS
    """

    # FLAME face region indices (approximate segmentation)
    SCALP_FACE_IDS: list[int] = []  # Will be populated from model data
    TEETH_FACE_IDS: list[int] = []
    EYEBALL_FACE_IDS: list[int] = []

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_shape_params: int = 100,
        num_expression_params: int = 50,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_shape_params = num_shape_params
        self.num_expression_params = num_expression_params
        self.device_str = device

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            self._init_dummy(device)
            logger.warning(
                "FLAME model file not found. Using dummy model. "
                "Download from https://flame.is.tue.mpg.de/"
            )

    def _load_model(self, path: str | Path) -> None:
        """Load FLAME model from .npz or .pkl file."""
        flame_data = _load_flame_data(path)

        # Template mesh
        self.register_buffer(
            "v_template",
            torch.tensor(flame_data["v_template"], dtype=torch.float32),
        )
        self.register_buffer(
            "faces",
            torch.tensor(flame_data["f"].astype(np.int64), dtype=torch.long),
        )

        # Shape blend shapes: (V, 3, num_shape_params)
        shapedirs = torch.tensor(flame_data["shapedirs"], dtype=torch.float32)
        self.register_buffer("shapedirs", shapedirs[:, :, :self.num_shape_params])

        # Expression blend shapes
        # In FLAME, expression blendshapes may be in 'shapedirs' after shape ones
        # or in a separate 'exprdirs' field
        if "exprdirs" in flame_data:
            exprdirs = torch.tensor(flame_data["exprdirs"], dtype=torch.float32)
        else:
            # Fallback: use remaining shapedirs
            exprdirs = shapedirs[:, :, self.num_shape_params:
                                 self.num_shape_params + self.num_expression_params]
        self.register_buffer("exprdirs", exprdirs[:, :, :self.num_expression_params])

        # Pose blend shapes
        if "posedirs" in flame_data:
            posedirs = torch.tensor(flame_data["posedirs"], dtype=torch.float32)
            self.register_buffer("posedirs", posedirs)
        else:
            self.register_buffer(
                "posedirs",
                torch.zeros(self.v_template.shape[0], 3, 1, dtype=torch.float32),
            )

        # Joint regressor
        J_regressor = flame_data["J_regressor"]
        if hasattr(J_regressor, "toarray"):
            J_regressor = J_regressor.toarray()
        self.register_buffer(
            "J_regressor",
            torch.tensor(J_regressor, dtype=torch.float32),
        )

        # LBS weights
        self.register_buffer(
            "lbs_weights",
            torch.tensor(flame_data["weights"], dtype=torch.float32),
        )

        # Kinematic tree
        kintree = flame_data["kintree_table"].astype(np.int64)
        self.register_buffer(
            "kintree_table",
            torch.tensor(kintree, dtype=torch.long),
        )

        self.num_vertices = self.v_template.shape[0]
        self.num_faces = self.faces.shape[0]
        self.num_joints = self.J_regressor.shape[0]

        logger.info(
            f"FLAME loaded: {self.num_vertices} verts, "
            f"{self.num_faces} faces, {self.num_joints} joints"
        )

    def _init_dummy(self, device: str) -> None:
        """Initialize a simple dummy head model for testing."""
        # Create a simple sphere mesh as placeholder
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
        V = sphere.vertices.shape[0]

        self.register_buffer(
            "v_template",
            torch.tensor(sphere.vertices, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "faces",
            torch.tensor(sphere.faces, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "shapedirs",
            torch.zeros(V, 3, self.num_shape_params, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "exprdirs",
            torch.zeros(V, 3, self.num_expression_params, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "posedirs",
            torch.zeros(V, 3, 1, dtype=torch.float32, device=device),
        )
        # 5 joints for FLAME (global, neck, jaw, left eye, right eye)
        num_joints = 5
        J_reg = torch.zeros(num_joints, V, dtype=torch.float32, device=device)
        J_reg[0, 0] = 1.0  # root joint at first vertex
        self.register_buffer("J_regressor", J_reg)
        self.register_buffer(
            "lbs_weights",
            torch.ones(V, num_joints, dtype=torch.float32, device=device) / num_joints,
        )
        self.register_buffer(
            "kintree_table",
            torch.tensor([[-1, 0, 0, 0, 0], [0, 1, 2, 3, 4]], dtype=torch.long, device=device),
        )

        self.num_vertices = V
        self.num_faces = sphere.faces.shape[0]
        self.num_joints = num_joints

    def forward(
        self,
        shape_params: torch.Tensor | None = None,
        expression_params: torch.Tensor | None = None,
        pose_params: torch.Tensor | None = None,
        global_rotation: torch.Tensor | None = None,
        global_translation: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: parameters → deformed mesh.

        Args:
            shape_params: (B, num_shape_params) or (num_shape_params,)
            expression_params: (B, num_expression_params) or (num_expression_params,)
            pose_params: (B, (num_joints-1)*3) axis-angle rotations
            global_rotation: (B, 3) global rotation axis-angle
            global_translation: (B, 3) global translation

        Returns:
            Dict with keys:
                - "vertices": (B, V, 3) deformed vertices
                - "faces": (F, 3) face indices
                - "joints": (B, J, 3) joint locations
                - "lbs_weights": (V, J) skinning weights
        """
        batch_size = 1
        device = self.v_template.device

        # Default parameters
        if shape_params is None:
            shape_params = torch.zeros(batch_size, self.num_shape_params, device=device)
        if expression_params is None:
            expression_params = torch.zeros(
                batch_size, self.num_expression_params, device=device
            )

        # Ensure batch dimension
        if shape_params.ndim == 1:
            shape_params = shape_params.unsqueeze(0)
        if expression_params.ndim == 1:
            expression_params = expression_params.unsqueeze(0)

        batch_size = shape_params.shape[0]

        # Apply blend shapes
        # v_shaped = v_template + shapedirs @ beta + exprdirs @ psi
        v_shaped = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Shape blendshapes: (B, V, 3)
        shape_offset = torch.einsum("vcd,bd->bvc", self.shapedirs, shape_params)
        v_shaped = v_shaped + shape_offset

        # Expression blendshapes
        expr_offset = torch.einsum("vcd,bd->bvc", self.exprdirs, expression_params)
        v_shaped = v_shaped + expr_offset

        # Compute joint locations
        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)

        # Apply LBS if pose is given
        if pose_params is not None:
            if pose_params.ndim == 1:
                pose_params = pose_params.unsqueeze(0)

            vertices = self._lbs(v_shaped, joints, pose_params)
        else:
            vertices = v_shaped

        # Apply global transform
        if global_rotation is not None:
            if global_rotation.ndim == 1:
                global_rotation = global_rotation.unsqueeze(0)
            R = axis_angle_to_matrix(global_rotation)  # (B, 3, 3)
            vertices = torch.bmm(vertices, R.transpose(1, 2))
            joints = torch.bmm(joints, R.transpose(1, 2))

        if global_translation is not None:
            if global_translation.ndim == 1:
                global_translation = global_translation.unsqueeze(0)
            vertices = vertices + global_translation.unsqueeze(1)
            joints = joints + global_translation.unsqueeze(1)

        return {
            "vertices": vertices,
            "faces": self.faces,
            "joints": joints,
            "lbs_weights": self.lbs_weights,
        }

    def _lbs(
        self,
        v_shaped: torch.Tensor,
        joints: torch.Tensor,
        pose_params: torch.Tensor,
    ) -> torch.Tensor:
        """Linear Blend Skinning.

        Args:
            v_shaped: (B, V, 3) shaped vertices.
            joints: (B, J, 3) joint locations.
            pose_params: (B, (J-1)*3) local joint rotations as axis-angles.

        Returns:
            (B, V, 3) posed vertices.
        """
        B = v_shaped.shape[0]
        J = self.num_joints
        device = v_shaped.device

        # Parse pose into per-joint rotations
        # First joint is identity (global), rest from pose_params
        num_pose_joints = min(pose_params.shape[1] // 3, J - 1)
        rot_mats = [torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)]

        for j in range(num_pose_joints):
            aa = pose_params[:, j * 3:(j + 1) * 3]
            rot_mats.append(axis_angle_to_matrix(aa))

        # Pad with identity if needed
        while len(rot_mats) < J:
            rot_mats.append(torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1))

        rot_mats = torch.stack(rot_mats, dim=1)  # (B, J, 3, 3)

        # Build world transforms via kinematic chain
        transforms = self._compute_world_transforms(joints, rot_mats)  # (B, J, 4, 4)

        # Skinning: weighted combination of transforms
        W = self.lbs_weights  # (V, J)

        # (B, V, 4, 4) = sum over J of W[v,j] * T[b,j]
        T = torch.einsum("vj,bjmn->bvmn", W, transforms)

        # Apply to homogeneous vertices
        v_h = torch.cat([
            v_shaped,
            torch.ones(*v_shaped.shape[:-1], 1, device=device),
        ], dim=-1)  # (B, V, 4)

        v_posed = torch.einsum("bvmn,bvn->bvm", T, v_h)[:, :, :3]

        return v_posed

    def _compute_world_transforms(
        self,
        joints: torch.Tensor,
        rot_mats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute world-space 4x4 transforms for each joint.

        Args:
            joints: (B, J, 3) joint positions in rest pose.
            rot_mats: (B, J, 3, 3) local rotation matrices.

        Returns:
            (B, J, 4, 4) world-space transforms.
        """
        B, J = joints.shape[:2]
        device = joints.device

        parents = self.kintree_table[0].long()  # (J,)

        transforms = []
        for j in range(J):
            # Local transform
            local_T = torch.zeros(B, 4, 4, device=device)
            local_T[:, :3, :3] = rot_mats[:, j]
            local_T[:, :3, 3] = joints[:, j]
            local_T[:, 3, 3] = 1.0

            parent = parents[j].item()
            if parent < 0 or j == 0:
                transforms.append(local_T)
            else:
                # Relative joint position
                rel_T = local_T.clone()
                rel_T[:, :3, 3] = joints[:, j] - joints[:, parent]
                world_T = transforms[parent] @ rel_T
                transforms.append(world_T)

        return torch.stack(transforms, dim=1)  # (B, J, 4, 4)

    def get_triangle_vertices(
        self, vertices: torch.Tensor, face_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the three vertex positions for each triangle.

        Args:
            vertices: (B, V, 3) or (V, 3) mesh vertices.
            face_indices: Optional subset of face indices (F',).

        Returns:
            v0, v1, v2: Each (B, F, 3) or (F, 3).
        """
        faces = face_indices if face_indices is not None else self.faces
        v0 = vertices[..., faces[:, 0], :]
        v1 = vertices[..., faces[:, 1], :]
        v2 = vertices[..., faces[:, 2], :]
        return v0, v1, v2
