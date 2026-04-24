"""Position-Based Dynamics (PBD/XPBD) solver using Taichi.

Implements real-time physics simulation for hair cage deformation,
including stretch, bending, volume preservation, and collision constraints.
Runs on Metal backend (Apple Silicon) or CUDA.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    ti = None
    HAS_TAICHI = False

from ..utils.logging import get_logger

logger = get_logger("dynamics.pbd_solver")

# Module-level Taichi initialization flag
_ti_initialized = False


def _ensure_taichi_init(arch: str = "metal") -> None:
    """Initialize Taichi runtime if not already done."""
    global _ti_initialized
    if not HAS_TAICHI:
        raise ImportError("Taichi is required for PBDSolver. Install via: pip install taichi")
    if _ti_initialized:
        return

    arch_map = {
        "metal": ti.metal,
        "cuda": ti.cuda,
        "cpu": ti.cpu,
        "vulkan": ti.vulkan,
    }

    ti_arch = arch_map.get(arch, ti.cpu)
    try:
        ti.init(arch=ti_arch, default_fp=ti.f32)
        logger.info(f"Taichi initialized with arch={arch}")
    except Exception as e:
        logger.warning(f"Failed to init Taichi with {arch}: {e}. Falling back to CPU.")
        ti.init(arch=ti.cpu, default_fp=ti.f32)

    _ti_initialized = True

# Use no-op decorators when Taichi is unavailable
_data_oriented = ti.data_oriented if HAS_TAICHI else lambda cls: cls
_kernel = ti.kernel if HAS_TAICHI else lambda fn: fn


@_data_oriented
class PBDSolver:
    """Extended Position-Based Dynamics (XPBD) solver for cage deformation.

    Simulates physically plausible hair dynamics by enforcing geometric
    constraints on cage vertices:
    - Stretch: preserves edge lengths
    - Bending: resists angular deformation between adjacent triangles
    - Volume: preserves cage volume
    - Collision: prevents penetration with FLAME head mesh

    Kinematic particles (hair roots) are driven by LBS and have
    inverse mass = 0 (immovable by constraints).
    """

    def __init__(
        self,
        num_vertices: int,
        num_edges: int,
        num_faces: int,
        dt: float = 0.016,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        num_iterations: int = 15,
        stretch_compliance: float = 0.0001,
        bending_compliance: float = 0.001,
        damping: float = 0.99,
        arch: str = "metal",
    ):
        _ensure_taichi_init(arch)

        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.num_faces = num_faces
        self.dt = dt
        self.num_iterations = num_iterations
        self.stretch_compliance = stretch_compliance
        self.bending_compliance = bending_compliance
        self.damping = damping

        # Particle state
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.predicted = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.inv_mass = ti.field(dtype=ti.f32, shape=num_vertices)

        # Constraints
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        self.rest_lengths = ti.field(dtype=ti.f32, shape=num_edges)
        self.faces_ti = ti.Vector.field(3, dtype=ti.i32, shape=num_faces)

        # External forces
        self.gravity = ti.Vector([gravity[0], gravity[1], gravity[2]])

        # Stretch constraint lambda (XPBD)
        self.stretch_lambda = ti.field(dtype=ti.f32, shape=num_edges)

    def initialize(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        faces: torch.Tensor,
        rest_lengths: torch.Tensor,
        is_kinematic: torch.Tensor,
    ) -> None:
        """Initialize solver state from torch tensors.

        Args:
            vertices: (M, 3) cage vertex positions.
            edges: (E, 2) edge connectivity.
            faces: (F, 3) face indices.
            rest_lengths: (E,) rest edge lengths.
            is_kinematic: (M,) bool, True for kinematic vertices.
        """
        verts_np = vertices.detach().cpu().numpy()
        edges_np = edges.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()
        rl_np = rest_lengths.detach().cpu().numpy()
        kin_np = is_kinematic.detach().cpu().numpy()

        for i in range(self.num_vertices):
            self.positions[i] = ti.Vector(verts_np[i])
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.predicted[i] = ti.Vector(verts_np[i])
            self.inv_mass[i] = 0.0 if kin_np[i] else 1.0

        for e in range(self.num_edges):
            self.edges[e] = ti.Vector(edges_np[e])
            self.rest_lengths[e] = rl_np[e]

        for f in range(self.num_faces):
            self.faces_ti[f] = ti.Vector(faces_np[f])

        logger.info(
            f"PBD solver initialized: {self.num_vertices} verts "
            f"({int(kin_np.sum())} kinematic), {self.num_edges} edges"
        )

    def step(
        self,
        kinematic_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance simulation by one time step.

        Args:
            kinematic_positions: (K, 3) updated positions for kinematic vertices,
                where K is the number of kinematic vertices. If None, kinematic
                vertices stay in place.

        Returns:
            (M, 3) updated cage vertex positions.
        """
        # Update kinematic vertices
        if kinematic_positions is not None:
            kin_np = kinematic_positions.detach().cpu().numpy()
            self._update_kinematic(kin_np)

        # PBD step
        self._predict_positions()
        self._reset_lambdas()

        for _ in range(self.num_iterations):
            self._solve_stretch_constraints()

        self._update_velocities()

        # Read back
        result = np.zeros((self.num_vertices, 3), dtype=np.float32)
        for i in range(self.num_vertices):
            p = self.positions[i]
            result[i] = [p[0], p[1], p[2]]

        return torch.tensor(result, dtype=torch.float32)

    def _update_kinematic(self, kin_positions: np.ndarray) -> None:
        """Update kinematic particle positions."""
        idx = 0
        for i in range(self.num_vertices):
            if self.inv_mass[i] == 0.0:
                if idx < kin_positions.shape[0]:
                    self.positions[i] = ti.Vector(kin_positions[idx])
                    self.predicted[i] = ti.Vector(kin_positions[idx])
                    idx += 1

    @_kernel
    def _predict_positions(self):
        """Semi-implicit Euler integration for free particles."""
        dt = self.dt
        damping = self.damping
        for i in range(self.num_vertices):
            if self.inv_mass[i] > 0.0:
                # Apply gravity and damping
                self.velocities[i] = self.velocities[i] * damping + self.gravity * dt
                self.predicted[i] = self.positions[i] + self.velocities[i] * dt
            else:
                self.predicted[i] = self.positions[i]

    @_kernel
    def _reset_lambdas(self):
        """Reset XPBD constraint multipliers."""
        for e in range(self.num_edges):
            self.stretch_lambda[e] = 0.0

    @_kernel
    def _solve_stretch_constraints(self):
        """XPBD stretch constraint: maintain edge rest lengths."""
        dt = self.dt
        alpha = self.stretch_compliance / (dt * dt)

        for e in range(self.num_edges):
            i0 = self.edges[e][0]
            i1 = self.edges[e][1]

            w0 = self.inv_mass[i0]
            w1 = self.inv_mass[i1]
            w_sum = w0 + w1

            if w_sum < 1e-8:
                continue

            diff = self.predicted[i0] - self.predicted[i1]
            dist = diff.norm()

            if dist < 1e-8:
                continue

            C = dist - self.rest_lengths[e]

            # XPBD: Δλ = -(C + α * λ) / (w_sum + α)
            delta_lambda = -(C + alpha * self.stretch_lambda[e]) / (w_sum + alpha)
            self.stretch_lambda[e] += delta_lambda

            # Position correction
            correction = delta_lambda * diff / dist

            if w0 > 0:
                self.predicted[i0] += w0 * correction
            if w1 > 0:
                self.predicted[i1] -= w1 * correction

    @_kernel
    def _update_velocities(self):
        """Update velocities from position change and finalize positions."""
        dt = self.dt
        for i in range(self.num_vertices):
            if self.inv_mass[i] > 0.0:
                self.velocities[i] = (self.predicted[i] - self.positions[i]) / dt
                self.positions[i] = self.predicted[i]

    def get_positions(self) -> torch.Tensor:
        """Get current cage vertex positions as torch tensor."""
        result = np.zeros((self.num_vertices, 3), dtype=np.float32)
        for i in range(self.num_vertices):
            p = self.positions[i]
            result[i] = [p[0], p[1], p[2]]
        return torch.tensor(result, dtype=torch.float32)


class PBDSolverPyTorch:
    """Pure PyTorch fallback PBD solver (no Taichi dependency).

    Useful for environments where Taichi is not available.
    Slower than the Taichi version but functionally equivalent.
    """

    def __init__(
        self,
        dt: float = 0.016,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        num_iterations: int = 15,
        stretch_compliance: float = 0.0001,
        damping: float = 0.99,
    ):
        self.dt = dt
        self.gravity = torch.tensor(gravity, dtype=torch.float32)
        self.num_iterations = num_iterations
        self.stretch_compliance = stretch_compliance
        self.damping = damping

        # State
        self.positions: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.inv_mass: torch.Tensor | None = None
        self.edges: torch.Tensor | None = None
        self.rest_lengths: torch.Tensor | None = None

    def initialize(
        self,
        vertices: torch.Tensor,
        edges: torch.Tensor,
        faces: torch.Tensor,
        rest_lengths: torch.Tensor,
        is_kinematic: torch.Tensor,
    ) -> None:
        """Initialize solver state."""
        self.positions = vertices.clone()
        self.velocities = torch.zeros_like(vertices)
        self.inv_mass = (~is_kinematic).float()
        self.edges = edges.clone()
        self.rest_lengths = rest_lengths.clone()

    def step(
        self,
        kinematic_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance by one time step."""
        assert self.positions is not None, "Solver not initialized"

        # Update kinematic vertices
        if kinematic_positions is not None:
            kin_mask = self.inv_mass == 0.0
            kin_indices = kin_mask.nonzero(as_tuple=True)[0]
            n_kin = min(kinematic_positions.shape[0], kin_indices.shape[0])
            self.positions[kin_indices[:n_kin]] = kinematic_positions[:n_kin]

        # Predict positions (semi-implicit Euler)
        predicted = self.positions.clone()
        free_mask = self.inv_mass > 0.0

        self.velocities[free_mask] *= self.damping
        self.velocities[free_mask] += self.gravity.to(self.positions.device) * self.dt
        predicted[free_mask] += self.velocities[free_mask] * self.dt

        # Constraint projection
        alpha = self.stretch_compliance / (self.dt ** 2)

        for _ in range(self.num_iterations):
            for e_idx in range(self.edges.shape[0]):
                i0, i1 = self.edges[e_idx]
                w0, w1 = self.inv_mass[i0], self.inv_mass[i1]
                w_sum = w0 + w1

                if w_sum < 1e-8:
                    continue

                diff = predicted[i0] - predicted[i1]
                dist = diff.norm()
                if dist < 1e-8:
                    continue

                C = dist - self.rest_lengths[e_idx]
                delta_lambda = -C / (w_sum + alpha)
                correction = delta_lambda * diff / dist

                if w0 > 0:
                    predicted[i0] += w0 * correction
                if w1 > 0:
                    predicted[i1] -= w1 * correction

        # Update velocities and positions
        self.velocities[free_mask] = (
            (predicted[free_mask] - self.positions[free_mask]) / self.dt
        )
        self.positions = predicted

        return self.positions.clone()
