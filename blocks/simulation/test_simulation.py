"""
Unit tests for Block 6 — Simulation.

Run with:
    python -m pytest blocks/simulation/test_simulation.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blocks.simulation.simulation import Simulation


# ── Helpers ────────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _two_room_config() -> dict:
    return {
        "grid": {"nx": 12, "ny": 6, "nz": 3, "dx": 1.0},
        "physics": {"diffusion_coefficient": 0.05},
        "rooms": [
            {"name": "left", "bounds": {
                "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
            }},
            {"name": "right", "bounds": {
                "x_min": 7, "x_max": 11, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
            }},
        ],
        "doors": [
            {"name": "mid_door", "bounds": {
                "x_min": 5, "x_max": 7, "y_min": 2, "y_max": 4, "z_min": 0, "z_max": 3
            }, "state": "open"}
        ],
        "sources": [
            {"name": "leak", "position": {"x": 2, "y": 3, "z": 1}, "rate": 5.0}
        ],
        "sensors": {
            "placement": "manual",
            "communication_radius": 6.0,
            "nodes": [
                {"name": "s_left", "position": {"x": 2, "y": 3, "z": 1}},
                {"name": "s_door", "position": {"x": 4, "y": 3, "z": 1}},
                {"name": "s_right", "position": {"x": 9, "y": 3, "z": 1}},
            ],
        },
        "noise": {"sensor_sigma": 0.0},
    }


@pytest.fixture
def env_path():
    return _write_config(_two_room_config())


# ══════════════════════════════════════════════════════════════════════════
# Initialization
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationInit:
    def test_creates_all_blocks(self, env_path):
        sim = Simulation(env_path, seed=42)
        assert sim.world is not None
        assert sim.network is not None
        assert sim.sensor_field is not None
        assert sim.actuator is not None
        assert sim.collector is not None

    def test_shared_environment(self, env_path):
        sim = Simulation(env_path, seed=42)
        assert sim.world.env is sim.env
        assert sim.sensor_field.env is sim.env
        assert sim.actuator.env is sim.env
        assert sim.network.env is sim.env

    def test_predictive_policy(self, env_path):
        sim = Simulation(env_path, actuator_policy="predictive", seed=42)
        assert sim.actuator.policy == "predictive"

    def test_reactive_policy(self, env_path):
        sim = Simulation(env_path, actuator_policy="reactive", seed=42)
        assert sim.actuator.policy == "reactive"


# ══════════════════════════════════════════════════════════════════════════
# Running
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationRun:
    def test_step_advances_world(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.step()
        assert sim.world.step_count == 1
        assert sim.world.time > 0

    def test_run_n_steps(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.run(50, record_every=10)
        assert sim.world.step_count == 50

    def test_records_metrics(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.run(50, record_every=10)
        # Should have 5 records (steps 10, 20, 30, 40, 50)
        assert len(sim.collector.records) == 5

    def test_callback_called(self, env_path):
        sim = Simulation(env_path, seed=42)
        calls = []
        sim.run(10, record_every=5, step_callback=lambda s, i: calls.append(i))
        assert len(calls) == 10

    def test_cumulative_contamination_increases(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.run(100, record_every=100)
        assert sim.cumulative_contamination > 0

    def test_collector_has_scalars(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.run(20, record_every=5)
        steps, mass = sim.collector.scalar_series("total_mass")
        assert len(steps) == 4
        assert all(m >= 0 for m in mass)


# ══════════════════════════════════════════════════════════════════════════
# Metadata and output
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationOutput:
    def test_summary(self, env_path):
        sim = Simulation(env_path, seed=42)
        sim.run(10)
        s = sim.summary()
        assert "10 steps" in s
        assert "Policy" in s

    def test_metadata(self, env_path):
        sim = Simulation(env_path, actuator_policy="predictive", seed=42)
        snap = sim.collector.snapshot()
        assert snap["metadata"]["policy"] == "predictive"
        assert snap["metadata"]["seed"] == 42

    def test_save_json(self, env_path, tmp_path):
        sim = Simulation(env_path, seed=42)
        sim.run(10, record_every=5)
        path = sim.collector.save_json(tmp_path / "test.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["n_records"] == 2

    def test_from_config(self, env_path, tmp_path):
        sim_cfg = {
            "environment": str(env_path),
            "actuator": {"policy": "reactive", "concentration_threshold": 0.3},
            "sensor_field": {"gossip_rounds": 2},
            "simulation": {"seed": 99, "name": "test_run"},
        }
        cfg_path = tmp_path / "sim_config.json"
        cfg_path.write_text(json.dumps(sim_cfg))

        sim = Simulation.from_config(cfg_path)
        assert sim.actuator.policy == "reactive"
        assert sim.actuator.concentration_threshold == 0.3
