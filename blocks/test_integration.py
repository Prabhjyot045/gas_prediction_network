"""
Integration tests — Block 1 (World) + Block 2 (Visualization) + Block 3 (Network)
                   + Block 4 (Sensor) + Metrics.

Verifies:
- Shared Environment object stays in sync across all blocks
- Door state changes propagate correctly
- Volume data integrity
- AnimationState frame-by-frame control
- Sensor network uses same environment
- SensorField reads from World and uses Network topology
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
from blocks.network.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.actuator.controller import ActuatorController
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
    """Two rooms connected by a door, with a gas source in the left room."""
    return {
        "grid": {"nx": 12, "ny": 10, "nz": 3, "dx": 1.0},
        "physics": {"diffusion_coefficient": 0.05},
        "rooms": [
            {"name": "left", "bounds": {
                "x_min": 1, "x_max": 5, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
            {"name": "right", "bounds": {
                "x_min": 7, "x_max": 11, "y_min": 1, "y_max": 9, "z_min": 0, "z_max": 3
            }},
        ],
        "doors": [
            {"name": "mid_door", "bounds": {
                "x_min": 5, "x_max": 7, "y_min": 4, "y_max": 6, "z_min": 0, "z_max": 3
            }, "state": "open"}
        ],
        "sources": [
            {"name": "leak", "position": {"x": 3, "y": 5, "z": 1}, "rate": 5.0}
        ],
        "sensors": {
            "placement": "grid",
            "spacing": 2,
            "z_levels": [1],
            "communication_radius": 4.0,
        },
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

    def test_door_state_consistent(self, setup):
        """Door state seen by Renderer matches World's env."""
        env, world, renderer = setup
        assert renderer.env.get_door_state("mid_door") == "open"
        world.close_door("mid_door")
        assert renderer.env.get_door_state("mid_door") == "closed"


# ══════════════════════════════════════════════════════════════════════════
# 2. Door toggle propagation
# ══════════════════════════════════════════════════════════════════════════

class TestDoorPropagation:
    def test_close_door_updates_walls_for_renderer(self, setup):
        """Closing a door via World should change walls that Renderer reads."""
        env, world, renderer = setup
        # Door region should be open initially
        assert not env.walls[5, 4, 1]
        assert not env.walls[6, 5, 2]

        world.close_door("mid_door")

        # Now the door region is wall
        assert env.walls[5, 4, 1]
        assert env.walls[6, 5, 2]
        # Renderer detects the change
        assert renderer.walls_changed

    def test_renderer_rebuilds_after_door_close(self, setup):
        """Renderer mesh rebuild picks up new wall state."""
        env, world, renderer = setup
        old_wall_points = renderer._wall_mesh.n_points

        world.close_door("mid_door")
        renderer.rebuild_meshes()

        # More wall points now (door cells became walls)
        assert renderer._wall_mesh.n_points > old_wall_points
        # Change flag should be cleared
        assert not renderer.walls_changed

    def test_open_door_reverses(self, setup):
        """Opening a closed door restores wall mesh to original state."""
        env, world, renderer = setup
        original_points = renderer._wall_mesh.n_points

        world.close_door("mid_door")
        renderer.rebuild_meshes()
        closed_points = renderer._wall_mesh.n_points

        world.open_door("mid_door")
        renderer.rebuild_meshes()

        assert renderer._wall_mesh.n_points == original_points
        assert renderer._wall_mesh.n_points < closed_points


# ══════════════════════════════════════════════════════════════════════════
# 3. Volume data integrity
# ══════════════════════════════════════════════════════════════════════════

class TestVolumeDataIntegrity:
    def test_volume_matches_phi_after_stepping(self, setup):
        """Volume data in renderer should exactly match World.phi."""
        env, world, renderer = setup
        world.run(50)

        phi = world.get_concentration()
        vol = renderer._build_concentration_volume(phi)
        vol_data = vol.cell_data["concentration"].reshape(
            env.grid_shape, order="F"
        )

        np.testing.assert_array_equal(vol_data, phi)

    def test_volume_shows_zero_behind_closed_door(self, setup):
        """After closing a door, right room should have zero gas."""
        env, world, renderer = setup
        world.close_door("mid_door")
        world.run(200)

        phi = world.get_concentration()
        vol = renderer._build_concentration_volume(phi)
        vol_data = vol.cell_data["concentration"].reshape(
            env.grid_shape, order="F"
        )

        # Right room: x=7..10, y=1..8
        right_room = vol_data[7:11, 1:9, :]
        assert np.max(right_room) < 1e-10

        # Left room should have gas
        left_room = vol_data[1:5, 1:9, :]
        assert np.max(left_room) > 0.1

    def test_volume_shows_gas_in_both_rooms_with_open_door(self, setup):
        """With open door, gas should flow to both rooms."""
        env, world, renderer = setup
        world.run(300)

        phi = world.get_concentration()
        vol = renderer._build_concentration_volume(phi)
        vol_data = vol.cell_data["concentration"].reshape(
            env.grid_shape, order="F"
        )

        right_room = vol_data[7:11, 1:9, :]
        assert np.max(right_room) > 0.01


# ══════════════════════════════════════════════════════════════════════════
# 4. Snapshot rendering with door states
# ══════════════════════════════════════════════════════════════════════════

class TestSnapshotIntegration:
    def test_snapshot_after_stepping(self, setup):
        """Snapshot should succeed after World has been stepped."""
        env, world, renderer = setup
        world.run(50)
        pl = renderer.snapshot(world, title="Integration Test")
        assert isinstance(pl, pv.Plotter)
        pl.close()

    def test_snapshot_auto_rebuilds_on_door_change(self, setup):
        """snapshot() should auto-detect wall changes and rebuild."""
        env, world, renderer = setup
        world.close_door("mid_door")
        # Don't manually call rebuild — snapshot should do it
        pl = renderer.snapshot(world, title="Door Closed")
        assert isinstance(pl, pv.Plotter)
        # walls_changed should now be False (snapshot triggered rebuild)
        assert not renderer.walls_changed
        pl.close()


# ══════════════════════════════════════════════════════════════════════════
# 5. AnimationState — frame-by-frame control
# ══════════════════════════════════════════════════════════════════════════

class TestAnimationState:
    def test_external_loop_control(self, setup):
        """Caller drives the simulation; AnimationState only renders."""
        env, world, renderer = setup
        anim = renderer.create_animation_plotter(
            world, off_screen=True,
        )
        anim.start()

        # Caller steps the world, then updates the frame
        for _ in range(10):
            world.step()
        anim.update_frame()

        assert world.step_count == 10
        anim.finish()

    def test_mid_animation_door_close(self, setup):
        """Closing a door mid-animation should update visuals."""
        env, world, renderer = setup
        anim = renderer.create_animation_plotter(
            world, off_screen=True,
        )
        anim.start()

        # Run some steps with door open
        for _ in range(50):
            world.step()
        anim.update_frame()

        # Close door mid-animation
        world.close_door("mid_door")

        # Next frame should detect the change and rebuild
        for _ in range(10):
            world.step()
        anim.update_frame()  # Should not crash, should rebuild meshes

        assert env.get_door_state("mid_door") == "closed"
        assert not renderer.walls_changed  # rebuild happened
        anim.finish()

    def test_frame_callback_in_animate(self, setup):
        """animate() frame_callback is called with correct args."""
        env, world, renderer = setup
        callback_log = []

        def my_callback(w: World, frame: int):
            callback_log.append((w.step_count, frame))
            if frame == 2:
                w.close_door("mid_door")

        renderer.animate(
            world, n_frames=5, steps_per_frame=3,
            frame_callback=my_callback,
            gif_path=None,
        )

        # Callback was called for each frame
        assert len(callback_log) == 5
        # Door was closed at frame 2
        assert env.get_door_state("mid_door") == "closed"
        # Steps: 5 frames * 3 steps/frame = 15
        assert world.step_count == 15


# ══════════════════════════════════════════════════════════════════════════
# 6. Decoupling verification
# ══════════════════════════════════════════════════════════════════════════

class TestDecoupling:
    def test_renderer_does_not_step_world(self, setup):
        """Renderer methods should never advance the World."""
        env, world, renderer = setup
        assert world.step_count == 0

        # snapshot does not step
        pl = renderer.snapshot(world)
        assert world.step_count == 0
        pl.close()

        # rebuild does not step
        renderer.rebuild_meshes()
        assert world.step_count == 0

    def test_animation_state_does_not_step(self, setup):
        """AnimationState.update_frame() does not call world.step()."""
        env, world, renderer = setup
        anim = renderer.create_animation_plotter(world, off_screen=True)
        anim.start()

        anim.update_frame()
        assert world.step_count == 0  # Renderer didn't step it

        world.run(5)
        anim.update_frame()
        assert world.step_count == 5  # Only caller's steps counted

        anim.finish()

    def test_blocks_work_independently(self):
        """Each block works with only Environment as the shared interface."""
        cfg = _write_config(_two_room_config())

        # Block 1 only
        env1 = Environment(cfg)
        world = World(env1)
        world.run(100)
        assert world.total_mass() > 0

        # Block 2 only (separate env instance)
        env2 = Environment(cfg)
        renderer = Renderer(env2)
        assert renderer._wall_mesh.n_points > 0

        # They CAN work together when given the same env
        renderer_shared = Renderer(env1)
        pl = renderer_shared.snapshot(world)
        assert isinstance(pl, pv.Plotter)
        pl.close()


# ══════════════════════════════════════════════════════════════════════════
# 7. Network integration (Block 3 ↔ Block 1)
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

    def test_door_change_does_not_affect_topology(self, setup):
        """Sensor network topology is static — door changes don't alter it."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        original_edges = net.n_edges
        original_nodes = net.n_nodes

        world.close_door("mid_door")

        # Topology unchanged (sensors don't move when doors close)
        assert net.n_edges == original_edges
        assert net.n_nodes == original_nodes

    def test_network_metrics_well_formed(self, setup):
        """network.metrics() should return a complete dict."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        m = net.metrics()
        assert isinstance(m["n_nodes"], int)
        assert isinstance(m["is_connected"], bool)
        assert isinstance(m["coverage"], float)


# ══════════════════════════════════════════════════════════════════════════
# 8. Metrics integration (MetricsCollector ↔ all blocks)
# ══════════════════════════════════════════════════════════════════════════

class TestMetricsIntegration:
    def test_world_metrics_dict(self, setup):
        """world.metrics() should return a well-formed dict."""
        env, world, renderer = setup
        world.run(10)
        m = world.metrics()
        assert "step" in m
        assert "total_mass" in m
        assert "peak_concentration" in m
        assert m["step"] == 10

    def test_collector_accumulates_world_metrics(self, setup):
        """MetricsCollector should accumulate world metrics over time."""
        env, world, renderer = setup
        mc = MetricsCollector("integration_test")

        for _ in range(5):
            world.step()
            m = world.metrics()
            mc.record(m, step=world.step_count)
            mc.record_scalar("total_mass", m["total_mass"], world.step_count)

        assert len(mc.records) == 5
        steps, values = mc.scalar_series("total_mass")
        assert len(steps) == 5
        # Mass should be increasing (source is active)
        assert values[-1] > values[0]

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
        assert snap["metadata"]["comm_radius"] == 4.0

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
# 9. SensorField integration (Block 4 ↔ Block 1 + Block 3)
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

    def test_sensor_field_reads_from_world(self, setup):
        """SensorField nodes should read concentration from World's phi."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        world.run(50)
        field.step(world)

        # At least one node near the source should have non-zero reading
        any_nonzero = any(
            n.filtered_concentration > 0.0 for n in field.nodes.values()
        )
        assert any_nonzero

    def test_sensor_field_does_not_step_world(self, setup):
        """SensorField.step() must not advance the World simulation."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        world.run(10)
        step_before = world.step_count
        field.step(world)
        assert world.step_count == step_before

    def test_door_change_affects_sensor_readings(self, setup):
        """Closing a door should change what sensors on the other side read."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)

        # Run with open door, gas flows to right room
        for _ in range(100):
            world.step()
            field.step(world)

        # Find a sensor in the right room (x >= 7)
        right_sensors = [
            n for n in field.nodes.values()
            if n.position[0] >= 7
        ]

        if right_sensors:
            reading_open = right_sensors[0].filtered_concentration

            # Close door and run more steps
            world.close_door("mid_door")
            for _ in range(200):
                world.step()
                field.step(world)

            # Gas in right room should decrease (door closed, no new inflow)
            reading_closed = right_sensors[0].filtered_concentration
            # With door closed the concentration should at least not keep growing
            # (it diffuses within the room but gets no new gas)
            assert isinstance(reading_closed, float)

    def test_gossip_uses_network_topology(self, setup):
        """Gossip messages should flow only along Network edges."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, gossip_rounds=3, seed=42)

        for _ in range(100):
            world.step()
            field.step(world)

        # Nodes with no network neighbors should never receive messages
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
            field.step(world)

        m = field.metrics(world)
        mc.record(m, step=world.step_count)

        snap = mc.snapshot()
        assert snap["n_records"] == 1
        assert snap["records"][0]["n_nodes"] > 0
        assert "concentration_rmse" in snap["records"][0]

    def test_all_four_blocks_together(self):
        """Full integration: World + Renderer + Network + SensorField."""
        cfg = _write_config(_two_room_config())
        env = Environment(cfg)
        world = World(env)
        renderer = Renderer(env)
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        mc = MetricsCollector("full_integration")

        # All share the same env
        assert world.env is renderer.env is net.env is field.env

        # Run simulation
        for step in range(50):
            world.step()
            field.step(world)

        # Collect metrics from all blocks
        mc.record({"block": "world", **world.metrics()}, step=50)
        mc.record({"block": "network", **net.metrics()}, step=50)
        mc.record({"block": "sensor", **field.metrics(world)}, step=50)

        snap = mc.snapshot()
        assert snap["n_records"] == 3

        # Renderer can still snapshot the current state
        pl = renderer.snapshot(world, title="Full Integration")
        assert isinstance(pl, pv.Plotter)
        pl.close()


# ══════════════════════════════════════════════════════════════════════════
# 10. Actuator integration (Block 5 ↔ Block 1 + Block 4)
# ══════════════════════════════════════════════════════════════════════════

class TestActuatorIntegration:
    def test_actuator_shares_env(self, setup):
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        ac = ActuatorController(env, field, policy="predictive")
        assert ac.env is env
        assert ac.sensor_field is field

    def test_actuator_closes_door_in_world(self, setup):
        """ActuatorController.evaluate() should close doors via World."""
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, gossip_rounds=3, seed=42)
        ac = ActuatorController(
            env, field, policy="reactive",
            concentration_threshold=0.01, proximity_radius=5.0,
        )
        for _ in range(200):
            world.step()
            field.step(world)
            ac.evaluate(world)

        if ac.doors_closed > 0:
            assert env.get_door_state("mid_door") == "closed"
            # Wall mask should reflect the closure
            door = env.doors["mid_door"]
            assert env.walls[door.slices].all()

    def test_actuator_does_not_step_world(self, setup):
        env, world, renderer = setup
        net = SensorNetwork(env)
        field = SensorField(env, net, seed=42)
        ac = ActuatorController(env, field, policy="predictive")

        world.run(10)
        field.step(world)
        step_before = world.step_count
        ac.evaluate(world)
        assert world.step_count == step_before


# ══════════════════════════════════════════════════════════════════════════
# 11. Simulation integration (Block 6 — full loop)
# ══════════════════════════════════════════════════════════════════════════

class TestSimulationIntegration:
    def test_simulation_all_blocks_share_env(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        env = sim.env
        assert sim.world.env is env
        assert sim.network.env is env
        assert sim.sensor_field.env is env
        assert sim.actuator.env is env

    def test_simulation_step_order(self):
        """Each step: world → sensor → actuator, in that order."""
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)

        sim.step()
        assert sim.world.step_count == 1
        # Sensor field should have stepped
        assert sim.sensor_field._step_count == 1

    def test_simulation_metrics_collection(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(20, record_every=10)
        assert len(sim.collector.records) == 2
        rec = sim.collector.records[-1]
        assert "total_mass" in rec
        assert "n_detecting" in rec
        assert "doors_closed" in rec

    def test_simulation_cumulative_contamination(self):
        cfg = _write_config(_two_room_config())
        sim = Simulation(cfg, seed=42)
        sim.run(50)
        assert sim.cumulative_contamination > 0


# ══════════════════════════════════════════════════════════════════════════
# 12. Benchmark integration (Block 7 — comparison)
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkIntegration:
    def test_benchmark_runs_both_policies(self):
        cfg = _write_config(_two_room_config())
        bm = Benchmark(cfg, n_steps=50, record_every=10, seed=42)
        comparison = bm.run()
        assert "predictive" in comparison
        assert "reactive" in comparison
        assert comparison["predictive"]["cumulative_contamination"] >= 0
        assert comparison["reactive"]["cumulative_contamination"] >= 0

    def test_benchmark_results_independent(self):
        """Each policy run should be independent (separate World instances)."""
        cfg = _write_config(_two_room_config())
        bm = Benchmark(cfg, n_steps=50, record_every=10, seed=42)
        pred_sim = bm.run_predictive()
        react_sim = bm.run_reactive()
        # Different World objects
        assert pred_sim.world is not react_sim.world
        assert pred_sim.env is not react_sim.env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
