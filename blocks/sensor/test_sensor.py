"""
Unit tests for Block 4 — Sensor Nodes (Rolling Buffer, TTI, Gossip).

Run with:
    python -m pytest blocks/sensor/test_sensor.py -v
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from blocks.sensor.gossip import NegotiationMessage
from blocks.sensor.node import SensorNode, RollingBuffer
from blocks.sensor.sensor_field import SensorField
from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.network.sensor_network import SensorNetwork


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
    }


def _write_config(cfg: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


def _make_env_world():
    path = _write_config(_make_config())
    env = Environment(path)
    world = World(env)
    return env, world


def _make_field():
    env, world = _make_env_world()
    net = SensorNetwork(env)
    sf = SensorField(env, net, gossip_rounds=2, seed=42)
    return env, world, net, sf


# ══════════════════════════════════════════════════════════════════════════
# RollingBuffer tests
# ══════════════════════════════════════════════════════════════════════════

class TestRollingBuffer:
    def test_append_and_size(self):
        buf = RollingBuffer(max_samples=10)
        buf.append(0.0, 1.0)
        buf.append(1.0, 2.0)
        assert buf.size == 2

    def test_max_samples_enforced(self):
        buf = RollingBuffer(max_samples=3)
        for i in range(10):
            buf.append(float(i), float(i))
        assert buf.size == 3

    def test_slope_linear(self):
        buf = RollingBuffer(max_samples=100)
        for i in range(50):
            buf.append(float(i), 2.0 * i + 5.0)
        assert buf.slope() == pytest.approx(2.0, abs=1e-6)

    def test_slope_constant(self):
        buf = RollingBuffer(max_samples=100)
        for i in range(20):
            buf.append(float(i), 10.0)
        assert buf.slope() == pytest.approx(0.0, abs=1e-10)

    def test_slope_insufficient_data(self):
        buf = RollingBuffer(max_samples=10)
        buf.append(0.0, 1.0)
        assert buf.slope() == 0.0

    def test_latest_value(self):
        buf = RollingBuffer(max_samples=5)
        buf.append(0.0, 42.0)
        assert buf.latest_value == 42.0


# ══════════════════════════════════════════════════════════════════════════
# SensorNode tests
# ══════════════════════════════════════════════════════════════════════════

class TestSensorNodeSensing:
    def test_sense_reads_temperature(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=0.1, setpoint=22.0)
        T = np.full((12, 12, 3), 25.0)
        node.sense(T, timestamp=0.0)
        assert node.filtered_temperature == pytest.approx(25.0)

    def test_sense_with_noise(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=0.1,
                          sensor_sigma=0.5, rng=np.random.default_rng(42))
        T = np.full((12, 12, 3), 25.0)
        node.sense(T, timestamp=0.0)
        assert node.raw_reading != 25.0

    def test_buffer_fills_on_sense(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=0.1, buffer_seconds=1.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(5):
            node.sense(T, timestamp=i * 0.1)
        assert node.buffer.size == 5


class TestSensorNodeTTI:
    def test_tti_already_above_setpoint(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=22.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(10):
            T[3, 3, 1] = 20.0 + 0.5 * i
            node.sense(T, timestamp=float(i))
        assert node.tti == 0.0

    def test_tti_below_setpoint_heating(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=30.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(10):
            T[3, 3, 1] = 20.0 + 0.5 * i
            node.sense(T, timestamp=float(i))
        assert 5.0 < node.tti < 20.0

    def test_tti_stable(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=22.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(10):
            node.sense(T, timestamp=float(i))
        assert node.tti == float("inf")

    def test_urgency_from_tti(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=30.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(10):
            T[3, 3, 1] = 20.0 + 1.0 * i
            node.sense(T, timestamp=float(i))
        assert node.urgency > 0

    def test_urgency_zero_when_stable(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=22.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(10):
            node.sense(T, timestamp=float(i))
        assert node.urgency == 0.0


class TestSensorNodeGradient:
    def test_gradient_uniform(self):
        node = SensorNode("s", (5, 5, 1), dx=1.0, dt=0.1)
        T = np.full((12, 12, 3), 20.0)
        walls = np.zeros((12, 12, 3), dtype=bool)
        grad = node.compute_gradient(T, walls)
        np.testing.assert_allclose(grad, [0, 0, 0], atol=1e-10)

    def test_gradient_x_ramp(self):
        node = SensorNode("s", (5, 5, 1), dx=1.0, dt=0.1)
        T = np.zeros((12, 12, 3))
        for x in range(12):
            T[x, :, :] = float(x)
        walls = np.zeros((12, 12, 3), dtype=bool)
        grad = node.compute_gradient(T, walls)
        assert grad[0] == pytest.approx(1.0)
        assert abs(grad[1]) < 1e-10


class TestSensorNodeGossip:
    def test_create_message_when_heating(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=22.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(5):
            T[3, 3, 1] = 20.0 + 0.5 * i
            node.sense(T, timestamp=float(i))
        msg = node.create_negotiation_message(timestamp=4.0, talk_threshold=0.01)
        assert msg is not None
        assert msg.urgency > 0

    def test_no_message_when_stable(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0, setpoint=22.0)
        T = np.full((12, 12, 3), 20.0)
        for i in range(5):
            node.sense(T, timestamp=float(i))
        msg = node.create_negotiation_message(timestamp=4.0, talk_threshold=0.01)
        assert msg is None

    def test_receive_updates_urgencies(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0)
        msg = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=1.0,
            temperature=25.0, dT_dt=0.5,
            gradient=np.array([1.0, 0, 0]),
            urgency=0.5,
        )
        result = node.receive_negotiation(msg)
        assert "other" in node.neighbor_urgencies
        assert node.neighbor_urgencies["other"] == 0.5
        assert result is not None

    def test_receive_ignores_lower_urgency(self):
        node = SensorNode("s", (3, 3, 1), dx=1.0, dt=1.0)
        msg1 = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=1.0,
            temperature=25.0, dT_dt=0.5,
            gradient=np.zeros(3), urgency=0.8,
        )
        msg2 = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=2.0,
            temperature=25.0, dT_dt=0.5,
            gradient=np.zeros(3), urgency=0.3,
        )
        node.receive_negotiation(msg1)
        result = node.receive_negotiation(msg2)
        assert result is None
        assert node.neighbor_urgencies["other"] == 0.8

    def test_message_forward_increments_hops(self):
        msg = NegotiationMessage(
            origin_node="a", origin_position=(1, 1, 1),
            sender_node="a", timestamp=0.0,
            temperature=20.0, dT_dt=0.1,
            gradient=np.zeros(3), urgency=0.5,
        )
        fwd = msg.forward("b")
        assert fwd.hops == 1
        assert fwd.sender_node == "b"
        assert fwd.origin_node == "a"


# ══════════════════════════════════════════════════════════════════════════
# SensorField tests
# ══════════════════════════════════════════════════════════════════════════

class TestSensorFieldInit:
    def test_nodes_created(self):
        _, _, net, sf = _make_field()
        assert len(sf.nodes) == net.n_nodes
        assert len(sf.nodes) > 0

    def test_nodes_have_setpoints(self):
        _, _, _, sf = _make_field()
        for node in sf.nodes.values():
            assert node.setpoint in (22.0, 25.0)


class TestSensorFieldStep:
    def test_step_reads_temperature(self):
        _, world, _, sf = _make_field()
        world.run(5)
        sf.step(world)
        temps = [n.filtered_temperature for n in sf.nodes.values()]
        assert any(t != 0.0 for t in temps)

    def test_step_gossip_propagates(self):
        env, world, _, sf = _make_field()
        room = env.rooms["room_A"]
        for _ in range(20):
            world.T[room.slices] += 0.5
            world.step()
            sf.step(world)
        total_sent = sum(n.messages_sent for n in sf.nodes.values())
        assert total_sent > 0

    def test_step_increments_count(self):
        _, world, _, sf = _make_field()
        sf.step(world)
        sf.step(world)
        assert sf._step_count == 2


class TestSensorFieldMetrics:
    def test_metrics_dict(self):
        _, world, _, sf = _make_field()
        sf.step(world)
        m = sf.metrics(world)
        assert "n_nodes" in m
        assert "n_heating" in m
        assert "total_messages_sent" in m
        assert "temperature_rmse" in m

    def test_get_urgencies(self):
        _, world, _, sf = _make_field()
        sf.step(world)
        urgencies = sf.get_urgencies()
        assert len(urgencies) == len(sf.nodes)

    def test_get_ttis(self):
        _, world, _, sf = _make_field()
        sf.step(world)
        ttis = sf.get_ttis()
        assert len(ttis) == len(sf.nodes)
