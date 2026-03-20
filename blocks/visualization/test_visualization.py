"""
Unit tests for Block 2 — 3D Visualization.

These tests verify mesh construction and rendering logic without
displaying windows (uses off-screen rendering).

Run with:
    python -m pytest blocks/visualization/test_visualization.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.visualization.renderer import Renderer


# ── Helpers ───────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _test_config() -> dict:
    return {
        "grid": {"nx": 10, "ny": 10, "nz": 3, "dx": 1.0},
        "physics": {"diffusion_coefficient": 0.05},
        "rooms": [
            {"name": "room", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }}
        ],
        "doors": [
            {"name": "test_door", "bounds": {
                "x_min": 0, "x_max": 1, "y_min": 4, "y_max": 6, "z_min": 0, "z_max": 2
            }, "state": "open"}
        ],
        "sources": [
            {"name": "src", "position": {"x": 5, "y": 5, "z": 1}, "rate": 5.0}
        ],
    }


@pytest.fixture
def env_and_world():
    cfg = _write_config(_test_config())
    env = Environment(cfg)
    world = World(env)
    return env, world


# ══════════════════════════════════════════════════════════════════════════
# Renderer construction tests
# ══════════════════════════════════════════════════════════════════════════

class TestRendererConstruction:
    def test_creates_wall_mesh(self, env_and_world):
        env, _ = env_and_world
        renderer = Renderer(env)
        assert renderer._wall_mesh.n_points > 0

    def test_creates_door_meshes(self, env_and_world):
        env, _ = env_and_world
        renderer = Renderer(env)
        assert "test_door" in renderer._door_meshes
        assert renderer._door_meshes["test_door"].n_points > 0

    def test_creates_source_markers(self, env_and_world):
        env, world = env_and_world
        renderer = Renderer(env)
        markers = renderer._build_source_markers()
        assert markers.n_points > 0


# ══════════════════════════════════════════════════════════════════════════
# Volume construction tests
# ══════════════════════════════════════════════════════════════════════════

class TestVolumeConstruction:
    def test_volume_shape(self, env_and_world):
        env, world = env_and_world
        renderer = Renderer(env)
        vol = renderer._build_concentration_volume(world.get_concentration())
        assert isinstance(vol, pv.ImageData)
        # Dimensions are shape + 1 (cell vs point data)
        assert vol.dimensions == (11, 11, 4)

    def test_volume_data_matches_phi(self, env_and_world):
        env, world = env_and_world
        world.phi[5, 5, 1] = 42.0
        renderer = Renderer(env)
        vol = renderer._build_concentration_volume(world.get_concentration())
        data = vol.cell_data["concentration"]
        assert 42.0 in data

    def test_volume_after_stepping(self, env_and_world):
        env, world = env_and_world
        world.run(10)
        renderer = Renderer(env)
        vol = renderer._build_concentration_volume(world.get_concentration())
        data = vol.cell_data["concentration"]
        assert np.max(data) > 0


# ══════════════════════════════════════════════════════════════════════════
# Off-screen rendering tests
# ══════════════════════════════════════════════════════════════════════════

class TestOffScreenRendering:
    def test_snapshot_creates_plotter(self, env_and_world):
        env, world = env_and_world
        world.run(5)
        renderer = Renderer(env)
        # Use off-screen to avoid opening a window
        pv.OFF_SCREEN = True
        pl = renderer.snapshot(world, title="Test")
        assert isinstance(pl, pv.Plotter)
        pl.close()

    def test_snapshot_with_closed_door(self, env_and_world):
        env, world = env_and_world
        world.close_door("test_door")
        world.run(5)
        renderer = Renderer(env)
        pv.OFF_SCREEN = True
        pl = renderer.snapshot(world, title="Door Closed")
        assert isinstance(pl, pv.Plotter)
        pl.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
