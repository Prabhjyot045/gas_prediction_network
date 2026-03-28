"""
Unit tests for Block 6 — Full HVAC Simulation.

Run with:
    python -m pytest blocks/simulation/test_simulation.py -v
"""

from __future__ import annotations

import json
import tempfile

import pytest

from blocks.simulation.simulation import Simulation


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config() -> dict:
    return {
        "grid": {"nx": 12, "ny": 12, "nz": 3, "dx": 1.0},
        "physics": {"thermal_diffusivity": 0.02, "ambient_temperature": 20.0},
        "rooms": [
            {"name": "room_A", "bounds": {
                "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
            }, "setpoint": 22.0},
            {"name": "room_B", "bounds": {
                "x_min": 7, "x_max": 11, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
            }, "setpoint": 22.0},
        ],
        "hallways": [
            {"name": "corridor", "bounds": {
                "x_min": 5, "x_max": 7, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
            }},
        ],
        "vav_dampers": [
            {"name": "vav_A", "zone": "room_A",
             "position": {"x": 3, "y": 3, "z": 1},
             "max_flow": 1.0, "initial_opening": 0.5},
        ],
        "heat_sources": [
            {"name": "heat_A", "zone": "room_A", "rate": 0.5,
             "schedule": {"start": 0, "end": None}},
        ],
        "cooling_plant": {"Q_total": 5.0, "supply_temperature": 12.0},
        "sensors": {"placement": "grid", "spacing": 3, "z_levels": [1],
                     "communication_radius": 5.0},
        "noise": {"sensor_sigma": 0.0},
        "network": {"polling_interval": 5.0, "jitter_sigma": 0.5, "compute_delay": 1.0},
    }


def _write_config(cfg: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


def _make_sim(policy="edge") -> Simulation:
    path = _write_config(_make_config())
    return Simulation(path, actuator_policy=policy, seed=42)


# ══════════════════════════════════════════════════════════════════════════
# Init tests
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationInit:
    def test_creates_all_blocks(self):
        sim = _make_sim()
        assert sim.world is not None
        assert sim.sensor_field is not None
        assert sim.interface is not None
        assert sim.collector is not None

    def test_shared_env(self):
        sim = _make_sim()
        assert sim.world.env is sim.env
        assert sim.sensor_field.env is sim.env
        assert sim.interface.env is sim.env


# ══════════════════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationRun:
    def test_step_advances_time(self):
        sim = _make_sim()
        sim.step()
        assert sim.world.step_count == 1
        assert sim.world.time > 0

    def test_run_multiple_steps(self):
        sim = _make_sim()
        sim.run(20, record_every=5)
        assert sim.world.step_count == 20

    def test_comfort_violation_accumulates(self):
        sim = _make_sim()
        sim.run(50)
        assert sim.cumulative_comfort_violation >= 0

    def test_energy_accumulates(self):
        sim = _make_sim()
        sim.run(20)
        assert sim.cumulative_energy > 0

    def test_step_callback(self):
        sim = _make_sim()
        called = []
        sim.run(5, step_callback=lambda s, i: called.append(i))
        assert len(called) == 5

    def test_centralized_policy_runs(self):
        sim = _make_sim("centralized")
        sim.run(20)
        assert sim.world.step_count == 20


# ══════════════════════════════════════════════════════════════════════════
# Output tests
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationOutput:
    def test_collector_has_records(self):
        sim = _make_sim()
        sim.run(10, record_every=5)
        assert len(sim.collector.records) >= 2

    def test_records_contain_expected_keys(self):
        sim = _make_sim()
        sim.run(10, record_every=10)
        rec = sim.collector.records[-1]
        assert "max_overshoot" in rec
        assert "cumulative_comfort_violation" in rec
        assert "cumulative_energy" in rec

    def test_scalar_series_recorded(self):
        sim = _make_sim()
        sim.run(10, record_every=5)
        steps, values = sim.collector.scalar_series("max_overshoot")
        assert len(steps) >= 2

    def test_summary_string(self):
        sim = _make_sim()
        sim.run(5)
        s = sim.summary()
        assert "Policy" in s
        assert "overshoot" in s
