"""
Unit tests for Block 2 — 3D Visualization (HVAC).

Tests verify mesh construction and rendering logic without
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
        "physics": {"thermal_diffusivity": 0.02, "ambient_temperature": 20.0},
        "rooms": [
            {"name": "room_A", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }, "setpoint": 22.0}
        ],
        "vav_dampers": [
            {"name": "vav_A", "zone": "room_A",
             "position": {"x": 5, "y": 5, "z": 1},
             "max_flow": 1.0, "initial_opening": 0.5}
        ],
        "heat_sources": [
            {"name": "heat_A", "zone": "room_A", "rate": 0.3,
             "schedule": {"start": 0, "end": None}}
        ],
        "cooling_plant": {"Q_total": 5.0, "supply_temperature": 12.0},
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

    def test_creates_damper_markers(self, env_and_world):
        env, _ = env_and_world
        renderer = Renderer(env)
        assert renderer._damper_markers.n_points > 0


# ══════════════════════════════════════════════════════════════════════════
# Volume construction tests
# ══════════════════════════════════════════════════════════════════════════

class TestVolumeConstruction:
    def test_volume_shape(self, env_and_world):
        env, world = env_and_world
        renderer = Renderer(env)
        vol = renderer._build_temperature_volume(world.T)
        assert isinstance(vol, pv.ImageData)
        assert vol.dimensions == (11, 11, 4)

    def test_volume_data_matches_T(self, env_and_world):
        env, world = env_and_world
        world.T[5, 5, 1] = 42.0
        renderer = Renderer(env)
        vol = renderer._build_temperature_volume(world.T)
        data = vol.cell_data["temperature"]
        assert 42.0 in data

    def test_volume_after_stepping(self, env_and_world):
        env, world = env_and_world
        world.run(10)
        renderer = Renderer(env)
        vol = renderer._build_temperature_volume(world.T)
        data = vol.cell_data["temperature"]
        # After stepping with heat source, max should exceed ambient
        assert np.max(data) >= 20.0


# ══════════════════════════════════════════════════════════════════════════
# Off-screen rendering tests
# ══════════════════════════════════════════════════════════════════════════

class TestOffScreenRendering:
    def test_snapshot_creates_plotter(self, env_and_world):
        env, world = env_and_world
        world.run(5)
        renderer = Renderer(env)
        pv.OFF_SCREEN = True
        pl = renderer.snapshot(world, title="Test")
        assert isinstance(pl, pv.Plotter)
        pl.close()

    def test_snapshot_with_hot_zone(self, env_and_world):
        env, world = env_and_world
        room = env.rooms["room_A"]
        world.T[room.slices] = 30.0
        renderer = Renderer(env)
        pv.OFF_SCREEN = True
        pl = renderer.snapshot(world, title="Hot Zone")
        assert isinstance(pl, pv.Plotter)
        pl.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
