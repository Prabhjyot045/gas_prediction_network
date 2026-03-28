"""
Integration tests — all HVAC blocks working together.

Verifies:
- Shared Environment object stays in sync across all blocks
- Temperature field integrity between World and Renderer
- Sensor network uses same environment
- Interface reads environment → feeds SensorField → actuates vents
- SensorField does pure inference (no World coupling in step())
- Simulation loop ties all blocks through the Interface
- Benchmark runs edge vs centralized comparison
- MetricsCollector accumulates from all blocks
- Decoupling between blocks

Run with:
    python -m pytest blocks/test_integration.py -v
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
from blocks.sensor.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.interface.interface import EnvironmentInterface
from blocks.simulation.simulation import Simulation
from blocks.benchmark.benchmark import Benchmark
from blocks.metrics.collector import MetricsCollector


# ── Helpers ───────────────────────────────────────────────────────────────

pv.OFF_SCREEN = True


def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _two_room_config() -> dict:
    """Two rooms connected by a hallway, with a heat source in room_A."""
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
            {"name": "heat_A", "zone": "room_A", "rate": 0.5,
             "schedule": {"start": 0, "end": None}},
        ],
        "cooling_plant": {"Q_total": 5.0, "supply_temperature": 12.0},
        "sensors": {"placement": "grid", "spacing": 3, "z_levels": [1],
                     "communication_radius": 5.0},
        "noise": {"sensor_sigma": 0.0},
        "network": {"polling_interval": 5.0, "jitter_sigma": 0.5, "compute_delay": 1.0},
    }


@pytest.fixture
def setup():
    """Create linked Environment, World, and Renderer sharing the same env."""
    cfg = _write_config(_two_room_config())
    env = Environment(cfg)
    world = World(env)
    renderer = Renderer(env)
    return env, world, renderer


# ══════════════════════════════════════════════════════════════════════════
# 1. Shared Environment — single source of truth
# ══════════════════════════════════════════════════════════════════════════

class TestSharedEnvironment:
    def test_same_env_object(self, setup):
        """World and Renderer must reference the exact same Environment."""
        env, world, renderer = setup
        assert world.env is env
        assert renderer.env is env

    def test_wall_array_identity(self, setup):
        """Both blocks see the same wall array (not a copy)."""
        env, world, renderer = setup
        assert world.env.walls is renderer.env.walls

    def test_damper_config_shared(self, setup):
        """Damper config is accessible from the shared env."""
        env, world, renderer = setup
        assert "vav_A" in env.dampers
        assert "vav_B" in env.dampers
        assert env.dampers["vav_A"].zone == "room_A"

    def test_room_setpoints_accessible(self, setup):
        """Room setpoints are part of the shared environment."""
        env, world, renderer = setup
        assert env.rooms["room_A"].setpoint == 22.0
        assert env.rooms["room_B"].setpoint == 22.0


# ══════════════════════════════════════════════════════════════════════════
# 2. Temperature field integrity
# ══════════════════════════════════════════════════════════════════════════

class TestTemperatureIntegrity:
    def test_temperature_volume_matches_world(self, setup):
        """Renderer volume data should match World.T exactly."""
        env, world, renderer = setup
        world.run(50)

        vol = renderer._build_temperature_volume(world.T)
        vol_data = vol.cell_data["temperature"].reshape(
            env.grid_shape, order="F"
        )

        np.testing.assert_array_equal(vol_data, world.T)

    def test_temperature_initialized_to_ambient(self, setup):
        """World temperature field starts at ambient temperature."""
        env, world, renderer = setup
        assert np.allclose(world.T, env.ambient_temperature)

    def test_heat_source_raises_temperature(self, setup):
        """Running the world with a heat source should raise room_A temperature."""
        env, world, renderer = setup
        room_A = env.rooms["room_A"]
        initial_temp = world.T[room_A.slices].mean()

        world.run(50)

        final_temp = world.T[room_A.slices].mean()
        assert final_temp > initial_temp


# ══════════════════════════════════════════════════════════════════════════
# 3. Snapshot rendering
# ══════════════════════════════════════════════════════════════════════════

class TestSnapshotIntegration:
    def test_snapshot_after_stepping(self, setup):
        """Snapshot should succeed after World has been stepped."""
        env, world, renderer = setup
        world.run(50)
        pl = renderer.snapshot(world, title="Integration Test")
        assert isinstance(pl, pv.Plotter)
        pl.close()


# ══════════════════════════════════════════════════════════════════════════
# 4. Network integration (Block 3 ↔ Block 1)
# ══════════════════════════════════════════════════════════════════════════

class TestNetworkIntegration:
    def test_network_shares_env(self, setup):
        """SensorNetwork should use the same Environment as World."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        assert net.env is env

    def test_all_sensors_in_open_cells(self, setup):
        """Every sensor position must be in a non-wall cell."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        for name, pos in net.positions.items():
            assert not env.walls[pos], f"Sensor {name} at {pos} is in a wall"

    def test_network_metrics_well_formed(self, setup):
        """network.metrics() should return a complete dict."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        m = net.metrics()
        assert isinstance(m["n_nodes"], int)
        assert isinstance(m["is_connected"], bool)
        assert isinstance(m["coverage"], float)


# ══════════════════════════════════════════════════════════════════════════
# 5. Metrics integration (MetricsCollector ↔ all blocks)
# ══════════════════════════════════════════════════════════════════════════

class TestMetricsIntegration:
    def test_world_metrics_dict(self, setup):
        """world.metrics() should return a well-formed dict."""
        env, world, renderer = setup
        world.run(10)
        m = world.metrics()
        assert "step" in m
        assert "mean_temperature" in m
        assert "max_overshoot" in m
        assert m["step"] == 10

    def test_collector_accumulates_world_metrics(self, setup):
        """MetricsCollector should accumulate world metrics over time."""
        env, world, renderer = setup
        mc = MetricsCollector("integration_test")

        for _ in range(5):
            world.step()
            m = world.metrics()
            mc.record(m, step=world.step_count)
            mc.record_scalar("mean_temperature", m["mean_temperature"], world.step_count)

        assert len(mc.records) == 5
        steps, values = mc.scalar_series("mean_temperature")
        assert len(steps) == 5

    def test_collector_accumulates_network_metrics(self, setup):
        """MetricsCollector should store network topology metrics."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        mc = MetricsCollector("net_test")
        mc.record(net.metrics(), step=0)
        assert mc.latest()["n_nodes"] > 0

    def test_combined_metrics_snapshot(self, setup):
        """Both world and network metrics in one collector."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        mc = MetricsCollector("combined")

        mc.set_metadata(environment="two_room", comm_radius=net.comm_radius)
        mc.record({"type": "network", **net.metrics()}, step=0)

        world.run(10)
        mc.record({"type": "world", **world.metrics()}, step=10)

        snap = mc.snapshot()
        assert snap["n_records"] == 2

    def test_collector_save_and_reload(self, setup, tmp_path):
        """Save to JSON and verify it's loadable."""
        env, world, renderer = setup
        mc = MetricsCollector("save_test")
        world.run(5)
        mc.record(world.metrics(), step=5)

        path = mc.save_json(tmp_path / "test_output.json")
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["n_records"] == 1
        assert data["records"][0]["step"] == 5


# ══════════════════════════════════════════════════════════════════════════
# 6. SensorField integration — pure inference, no World coupling in step()
# ══════════════════════════════════════════════════════════════════════════

class TestSensorFieldIntegration:
    def test_sensor_field_shares_env(self, setup):
        """SensorField should reference the same Environment as World and Network."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        assert field.env is env
        assert field.network is net
        assert field.env is world.env

    def test_sensor_field_processes_readings(self, setup):
        """SensorField nodes should process scalar readings (not World.T directly)."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        world.run(50)
        # Interface reads World.T → scalar readings
        readings = {
            name: float(world.T[node.position])
            for name, node in field.nodes.items()
        }
        field.step(readings, timestamp=world.time)

        # Nodes near the heat source should have values above ambient
        any_heated = any(
            n.filtered_value > env.ambient_temperature
            for n in field.nodes.values()
        )
        assert any_heated

    def test_sensor_field_does_not_step_world(self, setup):
        """SensorField.step() must not advance the World simulation."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        world.run(10)
        step_before = world.step_count
        readings = {name: float(world.T[node.position]) for name, node in field.nodes.items()}
        field.step(readings, timestamp=world.time)
        assert world.step_count == step_before

    def test_urgency_increases_with_heat(self, setup):
        """Nodes receiving rising values should develop non-zero urgency."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        for step_i in range(50):
            world.step()
            readings = {
                name: float(world.T[node.position])
                for name, node in field.nodes.items()
            }
            field.step(readings, timestamp=world.time)

        any_urgent = any(
            n.urgency > 0 for n in field.nodes.values()
        )
        assert any_urgent

    def test_gossip_uses_network_topology(self, setup):
        """Gossip messages should flow only along Network edges."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, gossip_rounds=3, seed=42)

        for _ in range(50):
            world.step()
            readings = {
                name: float(world.T[node.position])
                for name, node in field.nodes.items()
            }
            field.step(readings, timestamp=world.time)

        for name, node in field.nodes.items():
            if net.graph.degree(name) == 0:
                assert node.messages_received == 0

    def test_sensor_field_metrics_in_collector(self, setup):
        """SensorField metrics should integrate with MetricsCollector."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        mc = MetricsCollector("sensor_integration")

        for _ in range(20):
            world.step()
            readings = {
                name: float(world.T[node.position])
                for name, node in field.nodes.items()
            }
            field.step(readings, timestamp=world.time)

        m = field.metrics()
        mc.record(m, step=world.step_count)

        snap = mc.snapshot()
        assert snap["n_records"] == 1
        assert snap["records"][0]["n_nodes"] > 0


# ══════════════════════════════════════════════════════════════════════════
# 7. Interface integration (reads environment, feeds sensors, actuates vents)
# ══════════════════════════════════════════════════════════════════════════

class TestInterfaceIntegration:
    def test_interface_shares_env(self, setup):
        """EnvironmentInterface should reference the same Environment."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", seed=42)
        assert iface.env is env
        assert iface.sensor_field is field

    def test_interface_reads_and_feeds(self, setup):
        """Interface.step() should read World.T and feed SensorField."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", seed=42)

        for _ in range(20):
            world.step()
            iface.step(world)

        # After stepping, sensors should have been fed values
        any_heated = any(
            n.filtered_value > env.ambient_temperature
            for n in field.nodes.values()
        )
        assert any_heated

    def test_interface_sets_damper_openings(self, setup):
        """Interface.step() should return valid damper openings."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", seed=42)

        for _ in range(20):
            world.step()
            openings = iface.step(world)

        assert "vav_A" in openings
        assert "vav_B" in openings
        for v in openings.values():
            assert 0.0 <= v <= 1.0

    def test_interface_does_not_step_world(self, setup):
        """Interface.step() must not advance the World."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", seed=42)

        world.run(10)
        step_before = world.step_count
        iface.step(world)
        assert world.step_count == step_before

    def test_heated_room_gets_more_cooling(self, setup):
        """Room with active heat source should get higher vent opening."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, gossip_rounds=2, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", proximity_radius=5.0, seed=42)

        room_A = env.rooms["room_A"]
        for _ in range(30):
            world.T[room_A.slices] += 0.3
            world.step()
            iface.step(world)

        openings = iface.step(world)
        assert openings["vav_A"] >= openings["vav_B"]


# ══════════════════════════════════════════════════════════════════════════
# 8. Simulation integration (full loop)
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationIntegration:
    def test_simulation_all_blocks_share_env(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        env = sim.env
        assert sim.world.env is env
        assert sim.network.env is env
        assert sim.sensor_field.env is env
        assert sim.interface.env is env

    def test_simulation_step_order(self):
        """Each step: world -> interface (read → infer → actuate)."""
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)

        sim.step()
        assert sim.world.step_count == 1
        assert sim.sensor_field._step_count == 1

    def test_simulation_metrics_collection(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(20, record_every=10)
        assert len(sim.collector.records) == 2
        rec = sim.collector.records[-1]
        assert "max_overshoot" in rec
        assert "cumulative_comfort_violation" in rec
        assert "cumulative_energy" in rec
        assert "mean_age_of_information" in rec

    def test_simulation_cumulative_comfort_violation(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(50)
        assert sim.cumulative_comfort_violation >= 0

    def test_simulation_cumulative_energy(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(50)
        assert sim.cumulative_energy > 0

    def test_simulation_summary(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(10)
        s = sim.summary()
        assert "Policy" in s
        assert "overshoot" in s


# ══════════════════════════════════════════════════════════════════════════
# 9. Benchmark integration (comparison)
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkIntegration:
    def test_benchmark_runs_both_policies(self):
        cfg = _write_config(_two_room_config())
        bm = Benchmark(cfg, n_steps=50, record_every=10, seed=42)
        comparison = bm.run()
        assert "edge" in comparison
        assert "centralized" in comparison
        assert comparison["edge"]["cumulative_energy"] >= 0
        assert comparison["centralized"]["cumulative_energy"] >= 0
        assert "comparison" in comparison

    def test_benchmark_results_independent(self):
        """Each policy run should be independent (separate World instances)."""
        cfg = _write_config(_two_room_config())
        bm = Benchmark(cfg, n_steps=50, record_every=10, seed=42)
        edge_sim = bm.run_edge()
        cent_sim = bm.run_centralized()
        assert edge_sim.world is not cent_sim.world
        assert edge_sim.env is not cent_sim.env

    def test_benchmark_comparison_metrics(self):
        cfg = _write_config(_two_room_config())
        bm = Benchmark(cfg, n_steps=50, record_every=10, seed=42)
        result = bm.run()
        comp = result["comparison"]
        assert "comfort_improvement_pct" in comp
        assert "energy_savings_pct" in comp
        assert "edge_aoi" in comp
        assert "centralized_aoi" in comp


# ══════════════════════════════════════════════════════════════════════════
# 10. Decoupling verification
# ══════════════════════════════════════════════════════════════════════════

class TestDecoupling:
    def test_renderer_does_not_step_world(self, setup):
        """Renderer methods should never advance the World."""
        env, world, renderer = setup
        assert world.step_count == 0

        pl = renderer.snapshot(world)
        assert world.step_count == 0
        pl.close()

    def test_blocks_work_independently(self):
        """Each block works with only Environment as the shared interface."""
        cfg = _write_config(_two_room_config())

        # Block 1 only
        env1 = Environment(cfg)
        world = World(env1)
        world.run(50)
        assert world.T.mean() >= env1.ambient_temperature

        # Block 2 only (separate env instance)
        env2 = Environment(cfg)
        renderer = Renderer(env2)
        assert renderer._wall_mesh.n_points > 0

        # They CAN work together when given the same env
        renderer_shared = Renderer(env1)
        pl = renderer_shared.snapshot(world)
        assert isinstance(pl, pv.Plotter)
        pl.close()

    def test_sensor_field_decoupled_from_world(self):
        """SensorField.step() takes readings dict, NOT a World object."""
        cfg = _write_config(_two_room_config())
        env = Environment(cfg)
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        # Feed synthetic readings — no World needed
        readings = {name: 25.0 for name in field.nodes}
        field.step(readings, timestamp=0.0)

        for node in field.nodes.values():
            assert node.filtered_value == pytest.approx(25.0)


# ══════════════════════════════════════════════════════════════════════════
# 11. Full integration — all blocks together
# ══════════════════════════════════════════════════════════════════════════

class TestFullIntegration:
    def test_all_blocks_together(self):
        """Full integration: World + Renderer + Network + SensorField + Interface."""
        cfg = _write_config(_two_room_config())
        env = Environment(cfg)
        world = World(env)
        renderer = Renderer(env)
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        iface = EnvironmentInterface(env, field, policy="edge", seed=42)
        mc = MetricsCollector("full_integration")

        # All share the same env
        assert world.env is renderer.env is net.env is field.env is iface.env

        # Run simulation loop via interface
        for step in range(50):
            world.step()
            iface.step(world)

        # Collect metrics from all blocks
        mc.record({"block": "world", **world.metrics()}, step=50)
        mc.record({"block": "network", **net.metrics()}, step=50)
        mc.record({"block": "sensor", **field.metrics()}, step=50)
        mc.record({"block": "interface", **iface.metrics()}, step=50)

        snap = mc.snapshot()
        assert snap["n_records"] == 4

        # Renderer can still snapshot the current state
        pl = renderer.snapshot(world, title="Full Integration")
        assert isinstance(pl, pv.Plotter)
        pl.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
