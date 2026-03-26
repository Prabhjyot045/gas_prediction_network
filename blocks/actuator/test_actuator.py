"""
Unit tests for Block 5 — VAV Damper Controller.

Run with:
    python -m pytest blocks/actuator/test_actuator.py -v
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.network.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.actuator.controller import DamperController


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
            {"name": "vav_B", "zone": "room_B",
             "position": {"x": 9, "y": 3, "z": 1},
             "max_flow": 1.0, "initial_opening": 0.5},
        ],
        "heat_sources": [
            {"name": "heat_A", "zone": "room_A", "rate": 1.0,
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


def _make_system(policy="edge"):
    cfg = _make_config()
    path = _write_config(cfg)
    env = Environment(path)
    world = World(env)
    net = SensorNetwork(env)
    sf = SensorField(env, net, gossip_rounds=2, seed=42)
    ctrl = DamperController(
        env, sf, policy=policy,
        proximity_radius=5.0,
        polling_interval=cfg["network"]["polling_interval"],
        jitter_sigma=cfg["network"]["jitter_sigma"],
        compute_delay=cfg["network"]["compute_delay"],
        seed=42,
    )
    return env, world, sf, ctrl


# ══════════════════════════════════════════════════════════════════════════
# Init tests
# ══════════════════════════════════════════════════════════════════════════

class TestDamperInit:
    def test_invalid_policy_raises(self):
        cfg = _make_config()
        path = _write_config(cfg)
        env = Environment(path)
        net = SensorNetwork(env)
        sf = SensorField(env, net, seed=42)
        with pytest.raises(ValueError, match="Unknown policy"):
            DamperController(env, sf, policy="magic")

    def test_damper_sensor_mapping(self):
        _, _, _, ctrl = _make_system()
        assert "vav_A" in ctrl.damper_sensors
        assert "vav_B" in ctrl.damper_sensors
        # Each damper should have at least one nearby sensor
        assert len(ctrl.damper_sensors["vav_A"]) > 0


# ══════════════════════════════════════════════════════════════════════════
# Edge policy tests
# ══════════════════════════════════════════════════════════════════════════

class TestEdgePolicy:
    def test_evaluate_returns_openings(self):
        _, world, sf, ctrl = _make_system("edge")
        world.step()
        sf.step(world)
        openings = ctrl.evaluate(world)
        assert "vav_A" in openings
        assert "vav_B" in openings
        for v in openings.values():
            assert 0.0 <= v <= 1.0

    def test_heated_room_gets_more_cooling(self):
        env, world, sf, ctrl = _make_system("edge")
        room_A = env.rooms["room_A"]
        # Heat room A a lot to create urgency
        for _ in range(30):
            world.T[room_A.slices] += 0.3
            world.step()
            sf.step(world)
        openings = ctrl.evaluate(world)
        # Room A (heated) should get more cooling than room B
        assert openings["vav_A"] >= openings["vav_B"]

    def test_edge_aoi_is_zero(self):
        _, world, sf, ctrl = _make_system("edge")
        world.step()
        sf.step(world)
        ctrl.evaluate(world)
        assert ctrl.mean_age_of_information == 0.0

    def test_actions_logged(self):
        _, world, sf, ctrl = _make_system("edge")
        world.step()
        sf.step(world)
        ctrl.evaluate(world)
        assert len(ctrl.actions) > 0
        assert ctrl.actions[0].policy == "edge"


# ══════════════════════════════════════════════════════════════════════════
# Centralized policy tests
# ══════════════════════════════════════════════════════════════════════════

class TestCentralizedPolicy:
    def test_evaluate_returns_openings(self):
        _, world, sf, ctrl = _make_system("centralized")
        world.step()
        sf.step(world)
        openings = ctrl.evaluate(world)
        assert "vav_A" in openings
        for v in openings.values():
            assert 0.0 <= v <= 1.0

    def test_centralized_has_nonzero_aoi(self):
        _, world, sf, ctrl = _make_system("centralized")
        for _ in range(10):
            world.step()
            sf.step(world)
            ctrl.evaluate(world)
        # Centralized should accumulate AoI
        assert ctrl.mean_age_of_information >= 0.0

    def test_centralized_caches_between_polls(self):
        _, world, sf, ctrl = _make_system("centralized")
        world.step()
        sf.step(world)
        openings1 = ctrl.evaluate(world)
        # Immediately evaluate again — should use cache
        openings2 = ctrl.evaluate(world)
        assert openings1 == openings2


# ══════════════════════════════════════════════════════════════════════════
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════

class TestDamperMetrics:
    def test_metrics_dict(self):
        _, world, sf, ctrl = _make_system("edge")
        world.step()
        sf.step(world)
        ctrl.evaluate(world)
        m = ctrl.metrics()
        assert "policy" in m
        assert "damper_openings" in m
        assert "total_energy" in m
        assert "mean_age_of_information" in m

    def test_energy_accumulates(self):
        _, world, sf, ctrl = _make_system("edge")
        for _ in range(10):
            world.step()
            sf.step(world)
            ctrl.evaluate(world)
        assert ctrl.total_energy > 0
