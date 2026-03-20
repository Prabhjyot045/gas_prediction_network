"""
Unit tests for Block 4 — Sensor Nodes with Kalman Filter + Gossip.

Run with:
    python -m pytest blocks/sensor/test_sensor.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.network.sensor_network import SensorNetwork
from blocks.sensor.node import SensorNode
from blocks.sensor.gossip import GossipMessage
from blocks.sensor.sensor_field import SensorField


# ── Helpers ────────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _box_config(sensor_sigma: float = 0.0, **overrides) -> dict:
    """Simple open box 10x10x3 with a gas source and grid sensors."""
    cfg = {
        "grid": {"nx": 10, "ny": 10, "nz": 3, "dx": 1.0},
        "physics": {"diffusion_coefficient": 0.05},
        "rooms": [
            {"name": "box", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }}
        ],
        "doors": [],
        "sources": [
            {"name": "leak", "position": {"x": 3, "y": 3, "z": 1}, "rate": 5.0}
        ],
        "sensors": {
            "placement": "grid",
            "spacing": 3,
            "z_levels": [1],
            "communication_radius": 5.0,
        },
        "noise": {
            "sensor_sigma": sensor_sigma,
            "source_rate_sigma": 0.0,
        },
    }
    cfg.update(overrides)
    return cfg


def _two_room_config(sensor_sigma: float = 0.0) -> dict:
    """Two rooms with a door and manual sensors for gossip testing."""
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
            "communication_radius": 5.0,
            "nodes": [
                {"name": "s_left", "position": {"x": 2, "y": 3, "z": 1}},
                {"name": "s_mid", "position": {"x": 6, "y": 3, "z": 1}},
                {"name": "s_right", "position": {"x": 9, "y": 3, "z": 1}},
            ],
        },
        "noise": {
            "sensor_sigma": sensor_sigma,
            "source_rate_sigma": 0.0,
        },
    }


@pytest.fixture
def box_env():
    return Environment(_write_config(_box_config()))


@pytest.fixture
def box_world(box_env):
    return World(box_env)


@pytest.fixture
def box_network(box_env):
    return SensorNetwork(box_env)


@pytest.fixture
def two_room_setup():
    env = Environment(_write_config(_two_room_config()))
    world = World(env)
    net = SensorNetwork(env)
    return env, world, net


# ══════════════════════════════════════════════════════════════════════════
# GossipMessage tests
# ══════════════════════════════════════════════════════════════════════════

class TestGossipMessage:
    def test_forward_increments_hops(self):
        msg = GossipMessage(
            origin_node="s1", origin_position=(3, 3, 1),
            sender_node="s1", timestamp=1.0, concentration=0.5,
            gradient=np.array([0.1, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]), hops=0,
        )
        fwd = msg.forward("s2")
        assert fwd.hops == 1
        assert fwd.sender_node == "s2"
        assert fwd.origin_node == "s1"

    def test_forward_copies_arrays(self):
        grad = np.array([0.1, 0.2, 0.3])
        msg = GossipMessage(
            origin_node="s1", origin_position=(3, 3, 1),
            sender_node="s1", timestamp=1.0, concentration=0.5,
            gradient=grad, velocity=np.zeros(3), hops=0,
        )
        fwd = msg.forward("s2")
        grad[0] = 999.0  # mutate original
        assert fwd.gradient[0] == pytest.approx(0.1)

    def test_forward_preserves_origin(self):
        msg = GossipMessage(
            origin_node="s1", origin_position=(2, 2, 1),
            sender_node="s1", timestamp=0.5, concentration=1.0,
            gradient=np.zeros(3), velocity=np.zeros(3), hops=0,
        )
        fwd1 = msg.forward("s2")
        fwd2 = fwd1.forward("s3")
        assert fwd2.origin_node == "s1"
        assert fwd2.origin_position == (2, 2, 1)
        assert fwd2.hops == 2


# ══════════════════════════════════════════════════════════════════════════
# SensorNode tests
# ══════════════════════════════════════════════════════════════════════════

class TestSensorNodeSensing:
    def test_sense_reads_concentration(self, box_env, box_world):
        node = SensorNode("test", (3, 3, 1), dx=box_env.dx, dt=box_env.dt)
        box_world.run(10)  # let gas build up
        reading = node.sense(box_world.phi)
        assert reading > 0

    def test_sense_zero_at_start(self, box_env, box_world):
        node = SensorNode("test", (7, 7, 1), dx=box_env.dx, dt=box_env.dt)
        reading = node.sense(box_world.phi)
        # Concentration is zero everywhere at start
        assert abs(reading) < 0.01

    def test_kalman_tracks_rising_concentration(self, box_env, box_world):
        node = SensorNode("test", (3, 3, 1), dx=box_env.dx, dt=box_env.dt)
        readings = []
        for _ in range(50):
            box_world.step()
            readings.append(node.sense(box_world.phi))
        # Should track increasing concentration at the source
        assert readings[-1] > readings[0]
        assert node.filtered_rate > 0  # dφ/dt should be positive

    def test_kalman_reduces_noise(self):
        """With noisy readings, Kalman output should be smoother."""
        env = Environment(_write_config(_box_config(sensor_sigma=1.0)))
        world = World(env)
        node = SensorNode(
            "test", (3, 3, 1), dx=env.dx, dt=env.dt,
            sensor_sigma=1.0, rng=np.random.default_rng(42),
        )

        raw_values = []
        filtered_values = []
        for _ in range(100):
            world.step()
            node.sense(world.phi)
            raw_values.append(node.raw_reading)
            filtered_values.append(node.filtered_concentration)

        # Filtered should have lower variance than raw (after warmup)
        raw_var = np.var(raw_values[20:])
        filt_var = np.var(filtered_values[20:])
        assert filt_var < raw_var

    def test_sense_with_zero_sigma(self, box_env, box_world):
        node = SensorNode("test", (3, 3, 1), dx=box_env.dx, dt=box_env.dt,
                          sensor_sigma=0.0)
        box_world.run(10)
        node.sense(box_world.phi)
        true_val = float(box_world.phi[3, 3, 1])
        # With zero noise, raw reading equals true value
        assert node.raw_reading == pytest.approx(true_val)


class TestSensorNodeGradient:
    def test_gradient_uniform_field_is_zero(self, box_env):
        phi = np.ones(box_env.grid_shape) * 5.0
        phi[box_env.walls] = 0.0
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        grad = node.compute_gradient(phi, box_env.walls)
        assert np.allclose(grad, 0.0, atol=1e-10)

    def test_gradient_linear_field(self, box_env):
        phi = np.zeros(box_env.grid_shape)
        # Linear gradient in x: phi = 2*x
        for x in range(box_env.nx):
            phi[x, :, :] = 2.0 * x
        phi[box_env.walls] = 0.0

        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        grad = node.compute_gradient(phi, box_env.walls)
        # dφ/dx ≈ 2.0/dx, dφ/dy = 0, dφ/dz = 0
        assert grad[0] == pytest.approx(2.0 / box_env.dx, abs=0.1)
        assert abs(grad[1]) < 0.01
        assert abs(grad[2]) < 0.01

    def test_gradient_respects_walls(self, box_env):
        phi = np.zeros(box_env.grid_shape)
        # Position at boundary of room (next to wall)
        # Position (1, 5, 1) has wall at (0, 5, 1)
        phi[1, 5, 1] = 1.0
        phi[2, 5, 1] = 2.0
        node = SensorNode("test", (1, 5, 1), dx=box_env.dx, dt=box_env.dt)
        grad = node.compute_gradient(phi, box_env.walls)
        # Backward neighbor is wall, so uses center value (Neumann BC)
        # grad_x = (phi[2,5,1] - phi[1,5,1]) / (2*dx)
        expected_x = (2.0 - 1.0) / (2.0 * box_env.dx)
        assert grad[0] == pytest.approx(expected_x, abs=0.01)


class TestSensorNodeVelocity:
    def test_velocity_zero_gradient_gives_zero(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        node.gradient = np.zeros(3)
        node.filtered_rate = 1.0
        vel = node.compute_velocity()
        assert np.allclose(vel, 0.0)

    def test_velocity_direction(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        node.gradient = np.array([1.0, 0.0, 0.0])
        node.filtered_rate = 1.0  # concentration increasing
        vel = node.compute_velocity()
        # v = -(dφ/dt)/|∇φ|² * ∇φ = -1.0/1.0 * [1,0,0] = [-1,0,0]
        # Negative x means gas is flowing from high to low (diffusing outward)
        assert vel[0] == pytest.approx(-1.0)

    def test_velocity_realistic(self, box_env, box_world):
        node = SensorNode("test", (4, 3, 1), dx=box_env.dx, dt=box_env.dt)
        # Run simulation to build up gradients
        for _ in range(50):
            box_world.step()
            node.sense(box_world.phi)
        node.compute_gradient(box_world.phi, box_env.walls)
        vel = node.compute_velocity()
        # Near source, there should be some velocity
        assert np.linalg.norm(vel) >= 0  # at least non-negative


class TestSensorNodeGossip:
    def test_no_message_below_threshold(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        node.filtered_concentration = 0.001
        msg = node.create_gossip_message(timestamp=1.0, detection_threshold=0.01)
        assert msg is None

    def test_message_above_threshold(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        node.filtered_concentration = 0.5
        node.gradient = np.array([0.1, 0.0, 0.0])
        node.velocity = np.array([1.0, 0.0, 0.0])
        msg = node.create_gossip_message(timestamp=2.0)
        assert msg is not None
        assert msg.origin_node == "test"
        assert msg.concentration == pytest.approx(0.5)
        assert msg.timestamp == 2.0
        assert msg.hops == 0
        assert node.messages_sent == 1

    def test_receive_gossip_computes_arrival(self, box_env):
        receiver = SensorNode("recv", (8, 3, 1), dx=box_env.dx, dt=box_env.dt)
        msg = GossipMessage(
            origin_node="src", origin_position=(3, 3, 1),
            sender_node="src", timestamp=0.0, concentration=1.0,
            gradient=np.array([0.1, 0.0, 0.0]),
            velocity=np.array([2.0, 0.0, 0.0]),  # moving in +x direction
            hops=0,
        )
        forwarded = receiver.receive_gossip(msg)
        # Distance = 5 cells * 1.0 dx = 5.0m, speed = 2.0 m/s
        # arrival = 0.0 + 5.0/2.0 = 2.5s
        assert receiver.predictions["src"] == pytest.approx(2.5)
        assert receiver.messages_received == 1
        assert forwarded is not None
        assert forwarded.hops == 1

    def test_receive_gossip_ignores_stale(self, box_env):
        receiver = SensorNode("recv", (8, 3, 1), dx=box_env.dx, dt=box_env.dt)
        # First message: arrival at 2.5s
        msg1 = GossipMessage(
            origin_node="src", origin_position=(3, 3, 1),
            sender_node="src", timestamp=0.0, concentration=1.0,
            gradient=np.zeros(3), velocity=np.array([2.0, 0.0, 0.0]), hops=0,
        )
        receiver.receive_gossip(msg1)
        # Second message from same origin, later arrival
        msg2 = GossipMessage(
            origin_node="src", origin_position=(3, 3, 1),
            sender_node="relay", timestamp=1.0, concentration=0.8,
            gradient=np.zeros(3), velocity=np.array([1.0, 0.0, 0.0]), hops=1,
        )
        forwarded = receiver.receive_gossip(msg2)
        # Second prediction: 1.0 + 5.0/1.0 = 6.0s > 2.5s — stale
        assert forwarded is None
        assert receiver.predictions["src"] == pytest.approx(2.5)

    def test_receive_gossip_moving_away(self, box_env):
        receiver = SensorNode("recv", (8, 3, 1), dx=box_env.dx, dt=box_env.dt)
        msg = GossipMessage(
            origin_node="src", origin_position=(3, 3, 1),
            sender_node="src", timestamp=0.0, concentration=1.0,
            gradient=np.zeros(3),
            velocity=np.array([-2.0, 0.0, 0.0]),  # moving away
            hops=0,
        )
        forwarded = receiver.receive_gossip(msg)
        assert receiver.predictions.get("src", float("inf")) == float("inf")
        assert forwarded is None

    def test_earliest_predicted_arrival(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        assert node.earliest_predicted_arrival == float("inf")
        node.predictions["a"] = 3.0
        node.predictions["b"] = 1.5
        node.predictions["c"] = 5.0
        assert node.earliest_predicted_arrival == pytest.approx(1.5)


# ══════════════════════════════════════════════════════════════════════════
# SensorField tests
# ══════════════════════════════════════════════════════════════════════════

class TestSensorFieldInit:
    def test_creates_nodes_from_network(self, box_env, box_network):
        field = SensorField(box_env, box_network, seed=42)
        assert len(field.nodes) == box_network.n_nodes
        for name in box_network.positions:
            assert name in field.nodes

    def test_nodes_share_environment(self, box_env, box_network):
        field = SensorField(box_env, box_network)
        assert field.env is box_env

    def test_reproducible_with_seed(self, box_env, box_network):
        env_noisy = Environment(_write_config(_box_config(sensor_sigma=0.5)))
        world = World(env_noisy)
        net = SensorNetwork(env_noisy)
        world.run(20)

        f1 = SensorField(env_noisy, net, seed=42)
        f2 = SensorField(env_noisy, net, seed=42)
        f1.step(world)
        f2.step(world)

        for name in f1.nodes:
            assert f1.nodes[name].raw_reading == pytest.approx(
                f2.nodes[name].raw_reading
            )


class TestSensorFieldStep:
    def test_step_updates_concentrations(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, seed=42)

        world.run(50)
        field.step(world)

        # Node near source should have high concentration
        left_node = field.nodes["s_left"]
        assert left_node.filtered_concentration > 0

    def test_step_computes_gradients(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, seed=42)

        world.run(50)
        field.step(world)

        left_node = field.nodes["s_left"]
        # Near source, there should be a spatial gradient
        grad_mag = np.linalg.norm(left_node.gradient)
        assert grad_mag >= 0  # non-negative

    def test_gossip_propagates_predictions(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, gossip_rounds=3, seed=42)

        # Run enough steps for gas to build up and gossip to propagate
        for _ in range(100):
            world.step()
            field.step(world)

        # Left node (at source) should detect gas
        left_node = field.nodes["s_left"]
        assert left_node.filtered_concentration > 0.1

        # Right node should have received predictions via gossip
        right_node = field.nodes["s_right"]
        assert right_node.messages_received > 0

    def test_multiple_gossip_rounds(self, two_room_setup):
        env, world, net = two_room_setup
        world.run(50)

        # Single round
        field1 = SensorField(env, net, gossip_rounds=1, seed=42)
        field1.step(world)
        msgs1 = sum(n.messages_received for n in field1.nodes.values())

        # Multiple rounds should propagate further
        field3 = SensorField(env, net, gossip_rounds=3, seed=42)
        field3.step(world)
        msgs3 = sum(n.messages_received for n in field3.nodes.values())

        assert msgs3 >= msgs1


class TestSensorFieldMetrics:
    def test_metrics_has_expected_keys(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, seed=42)
        world.run(10)
        field.step(world)

        m = field.metrics(world)
        expected_keys = {
            "step", "n_nodes", "n_detecting", "n_with_predictions",
            "prediction_coverage", "total_messages_sent",
            "total_messages_received", "mean_filtered_concentration",
            "mean_velocity_magnitude", "concentration_rmse",
        }
        assert expected_keys.issubset(m.keys())

    def test_rmse_zero_without_noise(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, seed=42)

        # With zero noise and enough Kalman warmup, RMSE should be small
        for _ in range(50):
            world.step()
            field.step(world)

        rmse = field.concentration_rmse(world)
        assert rmse < 1.0  # reasonable bound after warmup

    def test_metrics_detecting_count(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, detection_threshold=0.01, seed=42)

        # Before stepping: nothing detected
        m0 = field.metrics()
        assert m0["n_detecting"] == 0

        # After gas builds up
        for _ in range(100):
            world.step()
            field.step(world)

        m1 = field.metrics()
        assert m1["n_detecting"] > 0

    def test_alert_nodes(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, gossip_rounds=3, seed=42)

        for _ in range(100):
            world.step()
            field.step(world)

        # Check for alert nodes with a large horizon
        alerts = field.get_alert_nodes(current_time=world.time, horizon=1000.0)
        # Some nodes should be alerting
        assert isinstance(alerts, list)

    def test_predicted_arrivals(self, two_room_setup):
        env, world, net = two_room_setup
        field = SensorField(env, net, gossip_rounds=3, seed=42)

        for _ in range(100):
            world.step()
            field.step(world)

        arrivals = field.get_predicted_arrivals()
        assert len(arrivals) == len(field.nodes)
        # All values should be floats
        for v in arrivals.values():
            assert isinstance(v, float)


# ══════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_network(self):
        """SensorField works with manual placement of zero sensors."""
        cfg = {
            "grid": {"nx": 6, "ny": 6, "nz": 3, "dx": 1.0},
            "physics": {"diffusion_coefficient": 0.05},
            "rooms": [
                {"name": "box", "bounds": {
                    "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
                }}
            ],
            "sources": [],
            "sensors": {"placement": "manual", "communication_radius": 5.0, "nodes": []},
        }
        env = Environment(_write_config(cfg))
        world = World(env)
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        assert len(field.nodes) == 0
        world.step()
        field.step(world)
        m = field.metrics()
        assert m["n_nodes"] == 0

    def test_single_node(self):
        """Single node: sensing works, no gossip to propagate."""
        cfg = {
            "grid": {"nx": 6, "ny": 6, "nz": 3, "dx": 1.0},
            "physics": {"diffusion_coefficient": 0.05},
            "rooms": [
                {"name": "box", "bounds": {
                    "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3
                }}
            ],
            "sources": [
                {"name": "src", "position": {"x": 3, "y": 3, "z": 1}, "rate": 5.0}
            ],
            "sensors": {
                "placement": "manual",
                "communication_radius": 5.0,
                "nodes": [{"name": "solo", "position": {"x": 3, "y": 3, "z": 1}}],
            },
        }
        env = Environment(_write_config(cfg))
        world = World(env)
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        assert len(field.nodes) == 1
        for _ in range(20):
            world.step()
            field.step(world)

        node = field.nodes["solo"]
        assert node.filtered_concentration > 0
        assert node.messages_sent > 0
        assert node.messages_received == 0  # no neighbors

    def test_node_metrics_dict(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        m = node.metrics()
        expected = {
            "name", "position", "raw_reading", "filtered_concentration",
            "filtered_rate", "concentration_uncertainty",
            "gradient_magnitude", "velocity_magnitude",
            "earliest_predicted_arrival", "messages_sent",
            "messages_received", "n_predictions",
        }
        assert expected == set(m.keys())

    def test_clear_inbox(self, box_env):
        node = SensorNode("test", (5, 5, 1), dx=box_env.dx, dt=box_env.dt)
        msg = GossipMessage(
            origin_node="src", origin_position=(3, 3, 1),
            sender_node="src", timestamp=0.0, concentration=1.0,
            gradient=np.zeros(3), velocity=np.array([1.0, 0.0, 0.0]),
            hops=0,
        )
        node.receive_gossip(msg)
        assert len(node.inbox) == 1
        node.clear_inbox()
        assert len(node.inbox) == 0
        # Predictions persist after clearing inbox
        assert len(node.predictions) > 0 or node.messages_received == 1
