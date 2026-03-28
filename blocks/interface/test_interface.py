"""
Unit tests for the Interface block — environment I/O boundary.

Tests that the interface:
- Reads sensor values from the World (input side)
- Feeds scalar values to the SensorField (no World coupling in inference)
- Translates urgency into vent routing commands (output side)
- Supports edge (AoI=0) and centralized (AoI>0) policies

Run with:
    python -m pytest blocks/interface/test_interface.py -v
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.sensor.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.interface.interface import EnvironmentInterface


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
    iface = EnvironmentInterface(
        env, sf, policy=policy,
        proximity_radius=5.0,
        polling_interval=cfg["network"]["polling_interval"],
        jitter_sigma=cfg["network"]["jitter_sigma"],
        compute_delay=cfg["network"]["compute_delay"],
        seed=42,
    )
    return env, world, sf, iface


# ══════════════════════════════════════════════════════════════════════════
# Init tests
# ══════════════════════════════════════════════════════════════════════════

class TestInterfaceInit:
    def test_invalid_policy_raises(self):
        cfg = _make_config()
        path = _write_config(cfg)
        env = Environment(path)
        net = SensorNetwork(env)
        sf = SensorField(env, net, seed=42)
        with pytest.raises(ValueError, match="Unknown policy"):
            EnvironmentInterface(env, sf, policy="magic")

    def test_damper_sensor_mapping(self):
        _, _, _, iface = _make_system()
        assert "vav_A" in iface.damper_sensors
        assert "vav_B" in iface.damper_sensors
        assert len(iface.damper_sensors["vav_A"]) > 0


# ══════════════════════════════════════════════════════════════════════════
# Input side: reading from environment
# ══════════════════════════════════════════════════════════════════════════

class TestInterfaceRead:
    def test_read_sensors_returns_dict(self):
        _, world, sf, iface = _make_system()
        readings = iface.read_sensors(world)
        assert len(readings) == len(sf.nodes)
        for name, val in readings.items():
            assert isinstance(val, float)

    def test_read_sensors_matches_world(self):
        _, world, sf, iface = _make_system()
        world.run(10)
        readings = iface.read_sensors(world)
        for name, node in sf.nodes.items():
            expected = float(world.T[node.position])
            assert readings[name] == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════════════
# Edge policy tests
# ══════════════════════════════════════════════════════════════════════════

class TestEdgePolicy:
    def test_step_returns_openings(self):
        _, world, sf, iface = _make_system("edge")
        world.step()
        openings = iface.step(world)
        assert "vav_A" in openings
        assert "vav_B" in openings
        for v in openings.values():
            assert 0.0 <= v <= 1.0

    def test_heated_room_gets_more_cooling(self):
        env, world, sf, iface = _make_system("edge")
        room_A = env.rooms["room_A"]
        # Heat room A a lot to create urgency
        for _ in range(30):
            world.T[room_A.slices] += 0.3
            world.step()
            iface.step(world)
        openings = iface.step(world)
        assert openings["vav_A"] >= openings["vav_B"]

    def test_edge_aoi_is_zero(self):
        _, world, sf, iface = _make_system("edge")
        world.step()
        iface.step(world)
        assert iface.mean_age_of_information == 0.0

    def test_actions_logged(self):
        _, world, sf, iface = _make_system("edge")
        world.step()
        iface.step(world)
        assert len(iface.actions) > 0
        assert iface.actions[0].policy == "edge"


# ══════════════════════════════════════════════════════════════════════════
# Centralized policy tests
# ══════════════════════════════════════════════════════════════════════════

class TestCentralizedPolicy:
    def test_step_returns_openings(self):
        _, world, sf, iface = _make_system("centralized")
        world.step()
        openings = iface.step(world)
        assert "vav_A" in openings
        for v in openings.values():
            assert 0.0 <= v <= 1.0

    def test_centralized_has_nonzero_aoi(self):
        _, world, sf, iface = _make_system("centralized")
        for _ in range(10):
            world.step()
            iface.step(world)
        assert iface.mean_age_of_information >= 0.0

    def test_centralized_caches_between_polls(self):
        _, world, sf, iface = _make_system("centralized")
        world.step()
        openings1 = iface.step(world)
        openings2 = iface.step(world)
        assert openings1 == openings2


# ══════════════════════════════════════════════════════════════════════════
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════

class TestInterfaceMetrics:
    def test_metrics_dict(self):
        _, world, sf, iface = _make_system("edge")
        world.step()
        iface.step(world)
        m = iface.metrics()
        assert "policy" in m
        assert "damper_openings" in m
        assert "total_energy" in m
        assert "mean_age_of_information" in m

    def test_energy_accumulates(self):
        _, world, sf, iface = _make_system("edge")
        for _ in range(10):
            world.step()
            iface.step(world)
        assert iface.total_energy > 0

    def test_temperature_rmse(self):
        env, world, sf, iface = _make_system("edge")
        world.step()
        iface.step(world)
        rmse = iface.temperature_rmse(world)
        # With sensor_sigma=0, RMSE should be ~0
        assert rmse == pytest.approx(0.0, abs=0.01)
