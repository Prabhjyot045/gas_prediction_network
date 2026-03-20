"""
Unit tests for Block 1 — World (3D FTCS Diffusion).

Run with:
    python -m pytest blocks/world/test_world.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blocks.world.stability import compute_max_dt, compute_stable_dt, validate_dt, fourier_number
from blocks.world.environment import Environment
from blocks.world.world import World


# ── Helpers ───────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    """Write a config dict to a temp JSON file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _minimal_config(**overrides) -> dict:
    """A minimal valid config: 10x10x3 box, one room filling most of it."""
    cfg = {
        "grid": {"nx": 10, "ny": 10, "nz": 3, "dx": 1.0},
        "physics": {"diffusion_coefficient": 0.05, "dt": None, "safety_factor": 0.4},
        "rooms": [
            {"name": "open_box", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }}
        ],
        "doors": [],
        "sources": [],
        "noise": {"sensor_sigma": 0.0, "source_rate_sigma": 0.0},
    }
    cfg.update(overrides)
    return cfg


# ══════════════════════════════════════════════════════════════════════════
# Stability tests
# ══════════════════════════════════════════════════════════════════════════

class TestStability:
    def test_max_dt_basic(self):
        # dx=1, D=0.05 → dt_max = 1/(6*0.05) = 3.333...
        assert abs(compute_max_dt(1.0, 0.05) - 10 / 3) < 1e-10

    def test_stable_dt_with_safety(self):
        dt = compute_stable_dt(1.0, 0.05, safety_factor=0.5)
        assert abs(dt - 0.5 * 10 / 3) < 1e-10

    def test_validate_dt_passes(self):
        validate_dt(1.0, 1.0, 0.05)  # 1.0 < 3.33

    def test_validate_dt_raises(self):
        with pytest.raises(ValueError, match="exceeds"):
            validate_dt(5.0, 1.0, 0.05)  # 5.0 > 3.33

    def test_invalid_dx(self):
        with pytest.raises(ValueError):
            compute_max_dt(0, 0.05)

    def test_invalid_D(self):
        with pytest.raises(ValueError):
            compute_max_dt(1.0, -0.1)

    def test_fourier_number(self):
        fo = fourier_number(1.0, 1.0, 0.05)
        assert abs(fo - 0.05) < 1e-10


# ══════════════════════════════════════════════════════════════════════════
# Environment tests
# ══════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    def test_grid_shape(self):
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        assert env.grid_shape == (10, 10, 3)

    def test_walls_carved(self):
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        # Inside room should be open
        assert not env.walls[5, 5, 1]
        # Outer boundary row x=0 should be wall
        assert env.walls[0, 5, 1]

    def test_auto_dt(self):
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        expected = 0.4 * (1.0**2 / (6 * 0.05))
        assert abs(env.dt - expected) < 1e-10

    def test_door_initially_closed(self):
        config = _minimal_config()
        config["doors"] = [
            {"name": "test_door", "bounds": {
                "x_min": 4, "x_max": 6, "y_min": 0, "y_max": 1,
                "z_min": 0, "z_max": 2
            }, "state": "closed"}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        assert env.walls[5, 0, 1]  # Door region is wall
        assert env.get_door_state("test_door") == "closed"

    def test_door_open_close(self):
        config = _minimal_config()
        config["doors"] = [
            {"name": "d1", "bounds": {
                "x_min": 4, "x_max": 6, "y_min": 0, "y_max": 1,
                "z_min": 0, "z_max": 2
            }, "state": "open"}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        assert not env.walls[5, 0, 1]  # Open

        env.close_door("d1")
        assert env.walls[5, 0, 1]  # Now wall

        env.open_door("d1")
        assert not env.walls[5, 0, 1]  # Open again

    def test_source_in_wall_raises(self):
        config = _minimal_config()
        config["sources"] = [
            {"name": "bad", "position": {"x": 0, "y": 0, "z": 0}, "rate": 1.0}
        ]
        cfg = _write_config(config)
        with pytest.raises(ValueError, match="inside a wall"):
            Environment(cfg)


# ══════════════════════════════════════════════════════════════════════════
# World tests
# ══════════════════════════════════════════════════════════════════════════

class TestWorld:
    def test_initial_state(self):
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        world = World(env)
        assert world.phi.shape == (10, 10, 3)
        assert world.total_mass() == 0.0

    def test_source_injection(self):
        config = _minimal_config()
        config["sources"] = [
            {"name": "src", "position": {"x": 5, "y": 5, "z": 1}, "rate": 10.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        world.step()
        # After one step, source cell should have rate * dt concentration
        expected = 10.0 * env.dt
        assert world.phi[5, 5, 1] > 0
        # Allow tolerance for diffusion spreading within the same step
        assert abs(world.phi[5, 5, 1] - expected) < expected * 0.5

    def test_mass_increases_with_source(self):
        config = _minimal_config()
        config["sources"] = [
            {"name": "src", "position": {"x": 5, "y": 5, "z": 1}, "rate": 5.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        world.run(10)
        assert world.total_mass() > 0

    def test_diffusion_spreads(self):
        """Gas at center should spread to neighbors after several steps."""
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        world = World(env)

        # Manually inject a blob at center
        world.phi[5, 5, 1] = 100.0

        world.run(50)

        # Neighbors should have received some concentration
        assert world.phi[4, 5, 1] > 0
        assert world.phi[6, 5, 1] > 0
        assert world.phi[5, 4, 1] > 0
        assert world.phi[5, 6, 1] > 0

    def test_walls_block_diffusion(self):
        """Walls should always have zero concentration."""
        config = _minimal_config()
        config["sources"] = [
            {"name": "src", "position": {"x": 5, "y": 5, "z": 1}, "rate": 10.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        world.run(100)

        # All wall cells should be zero
        assert np.all(world.phi[env.walls] == 0.0)

    def test_door_blocks_diffusion(self):
        """Closing a door should prevent gas from spreading through it."""
        config = _minimal_config()
        # Two rooms connected by a door
        config["rooms"] = [
            {"name": "left", "bounds": {
                "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
            {"name": "right", "bounds": {
                "x_min": 6, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
        ]
        config["doors"] = [
            {"name": "mid_door", "bounds": {
                "x_min": 5, "x_max": 6, "y_min": 4, "y_max": 6, "z_min": 0, "z_max": 3
            }, "state": "open"}
        ]
        config["sources"] = [
            {"name": "src", "position": {"x": 2, "y": 5, "z": 1}, "rate": 10.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        # Close the door immediately
        world.close_door("mid_door")

        world.run(200)

        # Right room should have (near) zero concentration
        right_max = np.max(world.phi[6:9, 1:9, :])
        assert right_max < 1e-10, f"Gas leaked through closed door: max={right_max}"

    def test_gas_flows_through_open_door(self):
        """Gas should flow from one room through an open door into an adjacent room."""
        config = _minimal_config()
        config["rooms"] = [
            {"name": "left", "bounds": {
                "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
            {"name": "right", "bounds": {
                "x_min": 6, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
        ]
        config["doors"] = [
            {"name": "mid_door", "bounds": {
                "x_min": 5, "x_max": 6, "y_min": 4, "y_max": 6, "z_min": 0, "z_max": 3
            }, "state": "open"}
        ]
        config["sources"] = [
            {"name": "src", "position": {"x": 2, "y": 5, "z": 1}, "rate": 10.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        world.run(200)

        # Right room should have received gas through the open door
        right_max = np.max(world.phi[6:9, 1:9, :])
        assert right_max > 0.1, f"Gas did not flow through open door: max={right_max}"

    def test_contamination_integral(self):
        """Contamination integral should accumulate over time."""
        config = _minimal_config()
        config["sources"] = [
            {"name": "src", "position": {"x": 5, "y": 5, "z": 1}, "rate": 10.0}
        ]
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        total_contam = 0.0
        for _ in range(50):
            world.step()
            total_contam += world.contamination_integral(threshold=0.1)

        assert total_contam > 0

    def test_symmetry(self):
        """A centered source in a symmetric box should diffuse symmetrically."""
        config = {
            "grid": {"nx": 11, "ny": 11, "nz": 3, "dx": 1.0},
            "physics": {"diffusion_coefficient": 0.05},
            "rooms": [
                {"name": "box", "bounds": {
                    "x_min": 1, "x_max": 10, "y_min": 1, "y_max": 10,
                    "z_min": 0, "z_max": 3
                }}
            ],
            "sources": [],
            "doors": [],
        }
        cfg = _write_config(config)
        env = Environment(cfg)
        world = World(env)

        # Place symmetric blob
        world.phi[5, 5, 1] = 100.0

        world.run(30)

        # Check X-symmetry: phi[5+d, 5, 1] ≈ phi[5-d, 5, 1]
        for d in range(1, 4):
            assert abs(world.phi[5+d, 5, 1] - world.phi[5-d, 5, 1]) < 1e-10

        # Check Y-symmetry
        for d in range(1, 4):
            assert abs(world.phi[5, 5+d, 1] - world.phi[5, 5-d, 1]) < 1e-10

    def test_no_negative_concentration(self):
        """Concentration should never go negative."""
        cfg = _write_config(_minimal_config())
        env = Environment(cfg)
        world = World(env)
        world.phi[5, 5, 1] = 1.0

        world.run(100)
        assert np.all(world.phi >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
