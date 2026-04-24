"""File I/O utilities for CompHairHead."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, size: int | None = None) -> torch.Tensor:
    """Load an image as a float32 tensor (C, H, W) in [0, 1].

    Args:
        path: Path to image file.
        size: If provided, resize to (size, size).

    Returns:
        Image tensor of shape (3, H, W).
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a (3, H, W) or (H, W, 3) float tensor as an image.

    Args:
        tensor: Image tensor in [0, 1].
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = tensor.detach().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):
        t = t.permute(1, 2, 0)  # (H, W, C)

    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    Image.fromarray(arr).save(path)


def save_pointcloud_ply(
    path: str | Path,
    positions: torch.Tensor,
    colors: torch.Tensor | None = None,
) -> None:
    """Save a point cloud as PLY file.

    Args:
        path: Output .ply path.
        positions: (N, 3) point positions.
        colors: Optional (N, 3) RGB colors in [0, 1].
    """
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pos = positions.detach().cpu().numpy()
    N = pos.shape[0]

    if colors is not None:
        col = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
        data = np.zeros(N, dtype=dtype)
        data["x"], data["y"], data["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
        data["red"], data["green"], data["blue"] = col[:, 0], col[:, 1], col[:, 2]
    else:
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        data = np.zeros(N, dtype=dtype)
        data["x"], data["y"], data["z"] = pos[:, 0], pos[:, 1], pos[:, 2]

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=True).write(str(path))


def load_pointcloud_ply(path: str | Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load a PLY point cloud.

    Returns:
        positions: (N, 3) tensor.
        colors: (N, 3) tensor in [0, 1] or None.
    """
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    pos = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
    positions = torch.from_numpy(pos.astype(np.float32))

    colors = None
    if "red" in vertex.data.dtype.names:
        col = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=-1)
        colors = torch.from_numpy(col.astype(np.float32)) / 255.0

    return positions, colors
