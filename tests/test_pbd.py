"""Unit tests for PBD solver."""

import torch
import pytest


class TestPBDSolverPyTorch:
    """Test the PyTorch fallback PBD solver."""

    def _make_simple_chain(self, n_verts=5, edge_length=0.1):
        """Create a simple chain of vertices for testing."""
        # Vertical chain
        vertices = torch.zeros(n_verts, 3)
        for i in range(n_verts):
            vertices[i, 1] = -i * edge_length  # hanging down

        edges = torch.tensor([[i, i + 1] for i in range(n_verts - 1)], dtype=torch.long)
        faces = torch.zeros(0, 3, dtype=torch.long)  # no faces for chain
        rest_lengths = torch.full((n_verts - 1,), edge_length)
        is_kinematic = torch.zeros(n_verts, dtype=torch.bool)
        is_kinematic[0] = True  # Pin the top

        return vertices, edges, faces, rest_lengths, is_kinematic

    def test_initialization(self):
        from comp_hair_head.dynamics.pbd_solver import PBDSolverPyTorch

        solver = PBDSolverPyTorch(dt=0.016, num_iterations=10)
        verts, edges, faces, rest, kin = self._make_simple_chain()
        solver.initialize(verts, edges, faces, rest, kin)

        assert solver.positions is not None
        assert solver.positions.shape == (5, 3)

    def test_kinematic_vertex_stays(self):
        from comp_hair_head.dynamics.pbd_solver import PBDSolverPyTorch

        solver = PBDSolverPyTorch(dt=0.016, gravity=(0, -9.81, 0), num_iterations=10)
        verts, edges, faces, rest, kin = self._make_simple_chain()
        solver.initialize(verts, edges, faces, rest, kin)

        initial_pos_0 = solver.positions[0].clone()

        # Step 10 times
        for _ in range(10):
            solver.step()

        # Kinematic vertex should not move
        assert torch.allclose(solver.positions[0], initial_pos_0, atol=1e-6)

    def test_free_vertex_falls(self):
        from comp_hair_head.dynamics.pbd_solver import PBDSolverPyTorch

        solver = PBDSolverPyTorch(
            dt=0.016, gravity=(0, -9.81, 0), num_iterations=5, damping=1.0
        )
        verts, edges, faces, rest, kin = self._make_simple_chain()
        initial_y = verts[-1, 1].item()
        solver.initialize(verts, edges, faces, rest, kin)

        for _ in range(20):
            solver.step()

        # Last vertex should have moved down (or stayed due to stretch constraint)
        # The y position should have changed
        final_y = solver.positions[-1, 1].item()
        assert final_y != pytest.approx(initial_y, abs=1e-4)

    def test_stretch_constraint(self):
        from comp_hair_head.dynamics.pbd_solver import PBDSolverPyTorch

        solver = PBDSolverPyTorch(
            dt=0.016, gravity=(0, 0, 0), num_iterations=15,
            stretch_compliance=0.0001,
        )
        verts, edges, faces, rest, kin = self._make_simple_chain()
        solver.initialize(verts, edges, faces, rest, kin)

        # Perturb a vertex
        solver.positions[2] += torch.tensor([0.05, 0.0, 0.0])

        # Step many times
        for _ in range(50):
            solver.step()

        # Check edge lengths are approximately preserved
        for e_idx in range(edges.shape[0]):
            i0, i1 = edges[e_idx]
            length = (solver.positions[i0] - solver.positions[i1]).norm().item()
            assert abs(length - rest[e_idx].item()) < 0.02  # Allow some tolerance
