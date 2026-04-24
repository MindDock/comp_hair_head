"""Configuration management for CompHairHead."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeviceConfig:
    device: str = "mps"
    taichi_arch: str = "metal"

    def resolve(self) -> str:
        """Auto-detect best available device."""
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def resolve_taichi(self) -> str:
        """Auto-detect best Taichi arch."""
        if self.taichi_arch == "auto":
            device = self.resolve()
            return {"cuda": "cuda", "mps": "metal"}.get(device, "cpu")
        return self.taichi_arch


@dataclass
class PreprocessConfig:
    image_size: int = 512
    num_views: int = 16


@dataclass
class GaussianConfig:
    sh_degree: int = 3
    feature_dim: int = 2
    init_opacity: float = 0.8
    min_opacity: float = 0.005


@dataclass
class FlameConfig:
    model_path: str = "assets/flame/generic_model.npz"
    landmark_path: str = "assets/flame/landmark_embedding.npy"
    num_shape_params: int = 100
    num_expression_params: int = 50


@dataclass
class SegmentationConfig:
    sam2_checkpoint: str = "assets/sam2/sam2_hiera_small.pt"
    seg_lr: float = 0.01
    seg_iterations: int = 200
    boundary_threshold: int = 5


@dataclass
class RegistrationConfig:
    lr_latent: float = 0.001
    lr_flame: float = 0.0001
    lr_gaussian: float = 0.0005
    iterations: int = 500
    lambda_rec: float = 1.0
    lambda_chamfer: float = 0.1
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.1


@dataclass
class AssemblyConfig:
    lr_hair: float = 0.0005
    lr_theta: float = 0.0001
    iterations: int = 300
    lambda_geo: float = 0.5
    collision_margin: float = 0.002


@dataclass
class CageConfig:
    voxel_resolution: int = 64
    target_vertices: int = 400
    simplify_ratio: float = 0.1


@dataclass
class PBDConfig:
    dt: float = 0.016
    gravity: list[float] = field(default_factory=lambda: [0.0, -9.81, 0.0])
    num_iterations: int = 15
    stretch_compliance: float = 0.0001
    bending_compliance: float = 0.001
    volume_compliance: float = 0.0
    damping: float = 0.99
    collision_margin: float = 0.001


@dataclass
class RenderConfig:
    width: int = 512
    height: int = 512
    background_color: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    fov: float = 20.0


@dataclass
class OutputConfig:
    save_dir: str = "outputs"
    save_interval: int = 50


@dataclass
class CompHairHeadConfig:
    """Master configuration for CompHairHead."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    flame: FlameConfig = field(default_factory=FlameConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    cage: CageConfig = field(default_factory=CageConfig)
    pbd: PBDConfig = field(default_factory=PBDConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompHairHeadConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> CompHairHeadConfig:
        """Build config from a flat/nesting dictionary."""
        cfg = cls()
        field_map = {
            "device": (DeviceConfig, cfg.device),
            "preprocess": (PreprocessConfig, cfg.preprocess),
            "gaussian": (GaussianConfig, cfg.gaussian),
            "flame": (FlameConfig, cfg.flame),
            "segmentation": (SegmentationConfig, cfg.segmentation),
            "registration": (RegistrationConfig, cfg.registration),
            "assembly": (AssemblyConfig, cfg.assembly),
            "cage": (CageConfig, cfg.cage),
            "pbd": (PBDConfig, cfg.pbd),
            "render": (RenderConfig, cfg.render),
            "output": (OutputConfig, cfg.output),
        }

        # Handle top-level device/taichi_arch shorthand
        if "device" in d and isinstance(d["device"], str):
            cfg.device.device = d["device"]
        if "taichi_arch" in d and isinstance(d["taichi_arch"], str):
            cfg.device.taichi_arch = d["taichi_arch"]

        for key, (dc_class, dc_instance) in field_map.items():
            if key in d and isinstance(d[key], dict):
                for k, v in d[key].items():
                    if hasattr(dc_instance, k):
                        setattr(dc_instance, k, v)
                setattr(cfg, key, dc_instance)

        return cfg

    def get_device(self) -> str:
        """Get resolved PyTorch device string."""
        return self.device.resolve()

    def get_taichi_arch(self) -> str:
        """Get resolved Taichi architecture string."""
        return self.device.resolve_taichi()


def load_config(path: str | Path | None = None) -> CompHairHeadConfig:
    """Load config, falling back to defaults."""
    if path is None:
        default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if default_path.exists():
            return CompHairHeadConfig.from_yaml(default_path)
        return CompHairHeadConfig()
    return CompHairHeadConfig.from_yaml(path)
