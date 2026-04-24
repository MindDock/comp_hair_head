"""Unit tests for config system."""

import pytest
from comp_hair_head.config import CompHairHeadConfig, load_config


class TestConfig:
    def test_defaults(self):
        cfg = CompHairHeadConfig()
        assert cfg.gaussian.sh_degree == 3
        assert cfg.pbd.num_iterations == 15
        assert cfg.cage.target_vertices == 400

    def test_device_resolve_cpu(self):
        cfg = CompHairHeadConfig()
        cfg.device.device = "cpu"
        assert cfg.get_device() == "cpu"

    def test_taichi_arch_resolve(self):
        cfg = CompHairHeadConfig()
        cfg.device.device = "mps"
        cfg.device.taichi_arch = "auto"
        assert cfg.get_taichi_arch() == "metal"

    def test_from_dict(self):
        d = {
            "device": "cpu",
            "gaussian": {"sh_degree": 2, "feature_dim": 4},
            "pbd": {"num_iterations": 20},
        }
        cfg = CompHairHeadConfig._from_dict(d)
        assert cfg.gaussian.sh_degree == 2
        assert cfg.gaussian.feature_dim == 4
        assert cfg.pbd.num_iterations == 20
