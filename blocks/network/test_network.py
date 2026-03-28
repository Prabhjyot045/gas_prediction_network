"""
Unit tests for Block 3 — Sensor Mesh Topology.

Run with:
    python -m pytest blocks/network/test_network.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blocks.world.environment import Environment
from blocks.sensor.sensor_network import SensorNetwork
from blocks.sensor.placement import grid_placement, random_placement, manual_placement


# ── Helpers ───────────────────────────────────────────────────────────────

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
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════

class TestMetrics:
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
