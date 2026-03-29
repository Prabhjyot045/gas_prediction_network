"""
Unit tests for the Sensor Network block — topology, inference, and gossip.

Tests cover:
- Sensor placement strategies (grid, random, manual)
- Graph construction and topology metrics
- Rolling buffer and slope estimation
- SensorNode inference (TTI, urgency)
- Gossip message propagation
- SensorField integration

Run with:
    python -m pytest blocks/sensor_network/test_sensor_network.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blocks.world.environment import Environment
from blocks.sensor_network.gossip import NegotiationMessage
from blocks.sensor_network.node import SensorNode, RollingBuffer
from blocks.sensor_network.sensor_field import SensorField
from blocks.sensor_network.sensor_network import SensorNetwork
from blocks.sensor_network.placement import grid_placement, random_placement, manual_placement


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _box_config(**overrides) -> dict:
    """Simple open box 10x10x3 with grid sensors."""
    cfg = {
        "grid": {"nx": 10, "ny": 10, "nz": 3, "dx": 1.0},
        "physics": {"thermal_diffusivity": 0.02, "ambient_temperature": 20.0},
        "rooms": [
            {"name": "box", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }, "setpoint": 22.0}
        ],
        "sensors": {
            "placement": "grid",
            "spacing": 3,
            "z_levels": [1],
            "communication_radius": 5.0,
        },
    }
    cfg.update(overrides)
    return cfg


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


def _make_field():
    cfg = _make_config()
    path = _write_config(cfg)
    env = Environment(path)
    net = SensorNetwork(env)
    sf = SensorField(env, net, gossip_rounds=2, seed=42)
    return env, net, sf


@pytest.fixture
def box_env():
    return Environment(_write_config(_box_config()))


# ══════════════════════════════════════════════════════════════════════════
# Placement tests
# ══════════════════════════════════════════════════════════════════════════

class TestPlacement:
    def test_grid_placement_produces_nodes(self, box_env):
        positions = grid_placement(box_env, box_env.sensor_config)
        assert len(positions) > 0

    def test_grid_placement_all_non_wall(self, box_env):
        positions = grid_placement(box_env, box_env.sensor_config)
        for name, pos in positions:
            assert not box_env.walls[pos], f"Sensor {name} at {pos} is in a wall"

    def test_grid_spacing(self, box_env):
        positions = grid_placement(box_env, box_env.sensor_config)
        xs = sorted(set(p[0] for _, p in positions))
        for i in range(1, len(xs)):
            assert xs[i] - xs[i-1] == 3

    def test_random_placement_count(self, box_env):
        cfg = {"placement": "random", "count": 5, "seed": 42, "z_levels": [1]}
        positions = random_placement(box_env, cfg)
        assert len(positions) == 5

    def test_random_placement_reproducible(self, box_env):
        cfg = {"placement": "random", "count": 5, "seed": 42, "z_levels": [1]}
        p1 = random_placement(box_env, cfg)
        p2 = random_placement(box_env, cfg)
        assert p1 == p2

    def test_random_placement_all_non_wall(self, box_env):
        cfg = {"placement": "random", "count": 10, "seed": 123, "z_levels": [1]}
        positions = random_placement(box_env, cfg)
        for name, pos in positions:
            assert not box_env.walls[pos]

    def test_manual_placement(self):
        config = _box_config()
        config["sensors"] = {
            "placement": "manual",
            "communication_radius": 5.0,
            "nodes": [
                {"name": "a", "position": {"x": 3, "y": 3, "z": 1}},
                {"name": "b", "position": {"x": 6, "y": 6, "z": 1}},
            ]
        }
        env = Environment(_write_config(config))
        positions = manual_placement(env, env.sensor_config)
        assert len(positions) == 2
        assert positions[0] == ("a", (3, 3, 1))

    def test_manual_placement_wall_raises(self):
        config = _box_config()
        config["sensors"] = {
            "placement": "manual",
            "nodes": [
                {"name": "bad", "position": {"x": 0, "y": 0, "z": 0}},
            ]
        }
        env = Environment(_write_config(config))
        with pytest.raises(ValueError, match="inside a wall"):
            manual_placement(env, env.sensor_config)

    def test_no_sensors_config(self):
        config = _box_config()
        del config["sensors"]
        env = Environment(_write_config(config))
        net = SensorNetwork(env)
        assert net.n_nodes == 0


# ══════════════════════════════════════════════════════════════════════════
# Graph construction tests
# ══════════════════════════════════════════════════════════════════════════

class TestGraphConstruction:
    def test_node_count(self, box_env):
        net = SensorNetwork(box_env)
        assert net.n_nodes == len(net.positions)
        assert net.n_nodes > 0

    def test_edges_within_radius(self, box_env):
        net = SensorNetwork(box_env)
        dx = box_env.dx
        for u, v, data in net.graph.edges(data=True):
            p1 = np.array(net.positions[u], dtype=float)
            p2 = np.array(net.positions[v], dtype=float)
            dist = float(np.linalg.norm((p1 - p2) * dx))
            assert dist <= net.comm_radius * dx + 1e-10

    def test_no_edges_beyond_radius(self):
        config = _box_config()
        config["sensors"]["communication_radius"] = 0.1
        env = Environment(_write_config(config))
        net = SensorNetwork(env)
        assert net.n_edges == 0

    def test_edge_weights_are_distances(self, box_env):
        net = SensorNetwork(box_env)
        dx = box_env.dx
        for u, v, data in net.graph.edges(data=True):
            p1 = np.array(net.positions[u], dtype=float)
            p2 = np.array(net.positions[v], dtype=float)
            expected = float(np.linalg.norm((p1 - p2) * dx))
            assert abs(data["weight"] - expected) < 1e-10

    def test_positions_array_shape(self, box_env):
        net = SensorNetwork(box_env)
        arr = net.node_positions_array()
        assert arr.shape == (net.n_nodes, 3)


# ══════════════════════════════════════════════════════════════════════════
# Topology metrics tests
# ══════════════════════════════════════════════════════════════════════════

class TestTopologyMetrics:
    def test_connected_graph(self, box_env):
        net = SensorNetwork(box_env)
        assert net.is_connected()
        assert net.connected_components() == 1

    def test_disconnected_graph(self):
        config = _box_config()
        config["sensors"]["communication_radius"] = 0.1
        env = Environment(_write_config(config))
        net = SensorNetwork(env)
        assert not net.is_connected()
        assert net.diameter() == float("inf")

    def test_degree_distribution(self, box_env):
        net = SensorNetwork(box_env)
        dd = net.degree_distribution()
        assert sum(dd.values()) == net.n_nodes

    def test_average_degree(self, box_env):
        net = SensorNetwork(box_env)
        assert net.average_degree() == 2.0 * net.n_edges / net.n_nodes

    def test_diameter_finite(self, box_env):
        net = SensorNetwork(box_env)
        if net.is_connected():
            assert net.diameter() < float("inf")

    def test_clustering_range(self, box_env):
        net = SensorNetwork(box_env)
        cc = net.clustering_coefficient()
        assert 0.0 <= cc <= 1.0

    def test_coverage_range(self, box_env):
        net = SensorNetwork(box_env)
        cov = net.coverage()
        assert 0.0 <= cov <= 1.0

    def test_coverage_increases_with_radius(self):
        config = _box_config()
        config["sensors"]["communication_radius"] = 3.0
        env1 = Environment(_write_config(config))
        net1 = SensorNetwork(env1)

        config["sensors"]["communication_radius"] = 10.0
        env2 = Environment(_write_config(config))
        net2 = SensorNetwork(env2)

        assert net2.coverage() >= net1.coverage()

    def test_metrics_dict_keys(self, box_env):
        net = SensorNetwork(box_env)
        m = net.metrics()
        expected_keys = {
            "n_nodes", "n_edges", "communication_radius", "is_connected",
            "connected_components", "average_degree", "diameter",
            "average_path_length", "clustering_coefficient", "coverage",
            "degree_distribution",
        }
        assert set(m.keys()) == expected_keys

    def test_empty_network_metrics(self):
        config = _box_config()
        del config["sensors"]
        env = Environment(_write_config(config))
        net = SensorNetwork(env)
        m = net.metrics()
        assert m["n_nodes"] == 0
        assert m["n_edges"] == 0
        assert m["coverage"] == 0.0


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
        readings = {name: 25.0 for name in sf.nodes}
        sf.step(readings, timestamp=0.0)
        for node in sf.nodes.values():
            assert node.filtered_value == pytest.approx(25.0)

    def test_step_gossip_propagates(self):
        _, _, sf = _make_field()
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
