"""
Unit tests for Block 4 — Sensor Nodes (Rolling Buffer, TTI, Gossip).

Tests the pure inference layer: nodes receive scalar (timestamp, value) pairs
and perform rate-of-change monitoring, TTI prediction, and gossip consensus.
No environment coupling — nodes never see World.T directly.

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
from blocks.sensor.sensor_network import SensorNetwork


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


def _make_field():
    cfg = _make_config()
    path = _write_config(cfg)
    env = Environment(path)
    net = SensorNetwork(env)
    sf = SensorField(env, net, gossip_rounds=2, seed=42)
    return env, net, sf


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
# SensorNode tests — pure inference, no environment arrays
# ══════════════════════════════════════════════════════════════════════════

class TestSensorNodeSensing:
    def test_sense_records_value(self):
        node = SensorNode("s", (3, 3, 1), dt=0.1, setpoint=22.0)
        node.sense(25.0, timestamp=0.0)
        assert node.filtered_value == pytest.approx(25.0)

    def test_sense_with_noise(self):
        node = SensorNode("s", (3, 3, 1), dt=0.1,
                          sensor_sigma=0.5, rng=np.random.default_rng(42))
        node.sense(25.0, timestamp=0.0)
        assert node.raw_reading != 25.0

    def test_buffer_fills_on_sense(self):
        node = SensorNode("s", (3, 3, 1), dt=0.1, buffer_seconds=1.0)
        for i in range(5):
            node.sense(20.0, timestamp=i * 0.1)
        assert node.buffer.size == 5


class TestSensorNodeTTI:
    def test_tti_already_above_setpoint(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=22.0)
        for i in range(10):
            node.sense(20.0 + 0.5 * i, timestamp=float(i))
        assert node.tti == 0.0

    def test_tti_below_setpoint_heating(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=30.0)
        for i in range(10):
            node.sense(20.0 + 0.5 * i, timestamp=float(i))
        assert 5.0 < node.tti < 20.0

    def test_tti_stable(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=22.0)
        for i in range(10):
            node.sense(20.0, timestamp=float(i))
        assert node.tti == float("inf")

    def test_urgency_from_tti(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=30.0)
        for i in range(10):
            node.sense(20.0 + 1.0 * i, timestamp=float(i))
        assert node.urgency > 0

    def test_urgency_zero_when_stable(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=22.0)
        for i in range(10):
            node.sense(20.0, timestamp=float(i))
        assert node.urgency == 0.0


class TestSensorNodeGossip:
    def test_create_message_when_heating(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=22.0)
        for i in range(5):
            node.sense(20.0 + 0.5 * i, timestamp=float(i))
        msg = node.create_negotiation_message(timestamp=4.0, talk_threshold=0.01)
        assert msg is not None
        assert msg.urgency > 0

    def test_no_message_when_stable(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0, setpoint=22.0)
        for i in range(5):
            node.sense(20.0, timestamp=float(i))
        msg = node.create_negotiation_message(timestamp=4.0, talk_threshold=0.01)
        assert msg is None

    def test_receive_updates_urgencies(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0)
        msg = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=1.0,
            value=25.0, dT_dt=0.5,
            urgency=0.5,
        )
        result = node.receive_negotiation(msg)
        assert "other" in node.neighbor_urgencies
        assert node.neighbor_urgencies["other"] == 0.5
        assert result is not None

    def test_receive_ignores_lower_urgency(self):
        node = SensorNode("s", (3, 3, 1), dt=1.0)
        msg1 = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=1.0,
            value=25.0, dT_dt=0.5,
            urgency=0.8,
        )
        msg2 = NegotiationMessage(
            origin_node="other", origin_position=(5, 5, 1),
            sender_node="other", timestamp=2.0,
            value=25.0, dT_dt=0.5,
            urgency=0.3,
        )
        node.receive_negotiation(msg1)
        result = node.receive_negotiation(msg2)
        assert result is None
        assert node.neighbor_urgencies["other"] == 0.8

    def test_message_forward_increments_hops(self):
        msg = NegotiationMessage(
            origin_node="a", origin_position=(1, 1, 1),
            sender_node="a", timestamp=0.0,
            value=20.0, dT_dt=0.1,
            urgency=0.5,
        )
        fwd = msg.forward("b")
        assert fwd.hops == 1
        assert fwd.sender_node == "b"
        assert fwd.origin_node == "a"


# ══════════════════════════════════════════════════════════════════════════
# SensorField tests — receives readings dict, not World
# ══════════════════════════════════════════════════════════════════════════

class TestSensorFieldInit:
    def test_nodes_created(self):
        _, net, sf = _make_field()
        assert len(sf.nodes) == net.n_nodes
        assert len(sf.nodes) > 0

    def test_nodes_have_setpoints(self):
        _, _, sf = _make_field()
        for node in sf.nodes.values():
            assert node.setpoint in (22.0, 25.0)


class TestSensorFieldStep:
    def test_step_processes_readings(self):
        _, _, sf = _make_field()
        # Feed scalar readings directly
        readings = {name: 25.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        for node in sf.nodes.values():
            assert node.filtered_value == pytest.approx(25.0)

    def test_step_gossip_propagates(self):
        _, _, sf = _make_field()
        # Feed rising values to generate urgency and gossip
        for step_i in range(20):
            readings = {
                name: 20.0 + 0.5 * step_i
                for name in sf.nodes
            }
            sf.step(readings, timestamp=float(step_i))
        total_sent = sum(n.messages_sent for n in sf.nodes.values())
        assert total_sent > 0

    def test_step_increments_count(self):
        _, _, sf = _make_field()
        readings = {name: 20.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        sf.step(readings, timestamp=1.0)
        assert sf._step_count == 2


class TestSensorFieldMetrics:
    def test_metrics_dict(self):
        _, _, sf = _make_field()
        readings = {name: 20.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        m = sf.metrics()
        assert "n_nodes" in m
        assert "n_heating" in m
        assert "total_messages_sent" in m

    def test_get_urgencies(self):
        _, _, sf = _make_field()
        readings = {name: 20.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        urgencies = sf.get_urgencies()
        assert len(urgencies) == len(sf.nodes)

    def test_get_ttis(self):
        _, _, sf = _make_field()
        readings = {name: 20.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        ttis = sf.get_ttis()
        assert len(ttis) == len(sf.nodes)
