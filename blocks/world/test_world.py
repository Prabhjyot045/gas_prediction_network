"""Unit tests for the HVAC world: stability, environment, and thermal physics."""

import json
import tempfile

import numpy as np
import pytest

from blocks.world.stability import (
    compute_max_dt_diffusion,
    compute_max_dt_advection,
    compute_stable_dt,
    validate_dt,
    fourier_number,
)
from blocks.world.environment import Environment
from blocks.world.world import World


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config(**overrides) -> dict:
    """Minimal HVAC config for testing."""
    cfg = {
        "grid": {"nx": 12, "ny": 12, "nz": 3, "dx": 1.0},
        "physics": {
            "thermal_diffusivity": 0.02,
            "dt": None,
            "safety_factor": 0.4,
            "ambient_temperature": 20.0,
        },
        "rooms": [
            {
                "name": "room_A",
                "bounds": {"x_min": 1, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3},
                "setpoint": 22.0,
            },
            {
                "name": "room_B",
                "bounds": {"x_min": 7, "x_max": 11, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3},
                "setpoint": 22.0,
            },
        ],
        "hallways": [
            {
                "name": "corridor",
                "bounds": {"x_min": 5, "x_max": 7, "y_min": 1, "y_max": 5, "z_min": 0, "z_max": 3},
            },
        ],
        "vav_dampers": [
            {
                "name": "vav_A",
                "zone": "room_A",
                "position": {"x": 3, "y": 3, "z": 1},
                "max_flow": 1.0,
                "initial_opening": 0.5,
            },
        ],
        "heat_sources": [
            {
                "name": "heat_A",
                "zone": "room_A",
                "rate": 0.5,
                "schedule": {"start": 0, "end": None},
            },
        ],
        "cooling_plant": {"Q_total": 5.0, "supply_temperature": 12.0},
        "sensors": {"placement": "grid", "spacing": 3, "z_levels": [1], "communication_radius": 5.0},
        "noise": {"sensor_sigma": 0.0},
        "network": {"polling_interval": 5.0, "jitter_sigma": 0.5, "compute_delay": 1.0},
    }
    cfg.update(overrides)
    return cfg


def _write_config(cfg: dict) -> str:
    """Write config to temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


def _make_env(**overrides) -> Environment:
    return Environment(_write_config(_make_config(**overrides)))


def _make_world(**overrides) -> tuple[Environment, World]:
    env = _make_env(**overrides)
    return env, World(env)


# ── Stability Tests ──────────────────────────────────────────────────────

class TestStability:
    def test_max_dt_diffusion(self):
        assert compute_max_dt_diffusion(1.0, 0.02) == pytest.approx(1.0 / (6 * 0.02))

    def test_max_dt_advection_zero(self):
        assert compute_max_dt_advection(1.0, 0.0) == float("inf")

    def test_max_dt_advection_positive(self):
        assert compute_max_dt_advection(1.0, 2.0) == pytest.approx(0.5)

    def test_stable_dt_uses_stricter(self):
        dt = compute_stable_dt(1.0, 0.02, v_max=2.0, safety_factor=1.0)
        assert dt == pytest.approx(0.5)

    def test_stable_dt_safety_factor(self):
        dt = compute_stable_dt(1.0, 0.02, safety_factor=0.4)
        assert dt == pytest.approx(0.4 * 1.0 / (6 * 0.02))

    def test_validate_dt_passes(self):
        dt = compute_stable_dt(1.0, 0.02, safety_factor=0.4)
        validate_dt(1.0, 0.02, dt)

    def test_validate_dt_raises_diffusion(self):
        with pytest.raises(ValueError, match="diffusion limit"):
            validate_dt(1.0, 0.02, 100.0)

    def test_invalid_dx(self):
        with pytest.raises(ValueError):
            compute_max_dt_diffusion(0, 0.02)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            compute_max_dt_diffusion(1.0, -0.02)

    def test_fourier_number(self):
        fo = fourier_number(0.1, 1.0, 0.02)
        assert fo == pytest.approx(0.002)


# ── Environment Tests ────────────────────────────────────────────────────

class TestEnvironment:
    def test_grid_shape(self):
        env = _make_env()
        assert env.grid_shape == (12, 12, 3)

    def test_walls_carved(self):
        env = _make_env()
        assert not env.walls[3, 3, 1]
        assert env.walls[0, 0, 0]

    def test_rooms_parsed(self):
        env = _make_env()
        assert "room_A" in env.rooms
        assert "room_B" in env.rooms
        assert env.rooms["room_A"].setpoint == 22.0

    def test_hallway_carved(self):
        env = _make_env()
        assert not env.walls[6, 3, 1]

    def test_zone_mask(self):
        env = _make_env()
        assert env.zone_mask[3, 3, 1] == "room_A"
        assert env.zone_mask[9, 3, 1] == "room_B"
        assert env.zone_mask[6, 3, 1] == ""

    def test_auto_dt(self):
        env = _make_env()
        expected = 0.4 * 1.0**2 / (6 * 0.02)
        assert env.dt == pytest.approx(expected)

    def test_dampers_parsed(self):
        env = _make_env()
        assert "vav_A" in env.dampers
        assert env.dampers["vav_A"].opening == 0.5
        assert env.dampers["vav_A"].max_flow == 1.0

    def test_heat_sources_parsed(self):
        env = _make_env()
        assert len(env.heat_sources) == 1
        assert env.heat_sources[0].zone == "room_A"

    def test_cooling_plant(self):
        env = _make_env()
        assert env.Q_total == 5.0
        assert env.supply_temperature == 12.0

    def test_damper_control(self):
        env = _make_env()
        env.set_damper_opening("vav_A", 0.8)
        assert env.get_damper_opening("vav_A") == 0.8

    def test_damper_clamps(self):
        env = _make_env()
        env.set_damper_opening("vav_A", 1.5)
        assert env.get_damper_opening("vav_A") == 1.0
        env.set_damper_opening("vav_A", -0.5)
        assert env.get_damper_opening("vav_A") == 0.0

    def test_damper_in_wall_raises(self):
        cfg = _make_config()
        cfg["vav_dampers"][0]["position"] = {"x": 0, "y": 0, "z": 0}
        with pytest.raises(ValueError, match="inside a wall"):
            Environment(_write_config(cfg))

    def test_damper_unknown_zone_raises(self):
        cfg = _make_config()
        cfg["vav_dampers"][0]["zone"] = "nonexistent"
        with pytest.raises(ValueError, match="unknown zone"):
            Environment(_write_config(cfg))

    def test_heat_source_unknown_zone_raises(self):
        cfg = _make_config()
        cfg["heat_sources"][0]["zone"] = "nonexistent"
        with pytest.raises(ValueError, match="unknown zone"):
            Environment(_write_config(cfg))


# ── World Tests ──────────────────────────────────────────────────────────

class TestWorld:
    def test_initial_temperature(self):
        _, world = _make_world()
        assert world.T[3, 3, 1] == pytest.approx(20.0)

    def test_heat_injection_raises_temperature(self):
        _, world = _make_world()
        initial_T = world.zone_mean_temperature("room_A")
        world.step()
        after_T = world.zone_mean_temperature("room_A")
        assert after_T > initial_T

    def test_room_b_no_heat_source(self):
        _, world = _make_world()
        initial_T = world.zone_mean_temperature("room_B")
        world.step()
        after_T = world.zone_mean_temperature("room_B")
        assert abs(after_T - initial_T) < 0.1

    def test_cooling_reduces_temperature(self):
        env, world = _make_world()
        world.T[3, 3, 1] = 30.0
        env.set_damper_opening("vav_A", 1.0)
        T_before = world.T[3, 3, 1]
        world.step()
        T_after = world.T[3, 3, 1]
        assert T_after < T_before

    def test_walls_at_ambient(self):
        env, world = _make_world()
        world.step()
        wall_T = world.T[env.walls]
        assert np.all(wall_T == env.ambient_temperature)

    def test_diffusion_spreads_heat(self):
        _, world = _make_world()
        world.T[3, 3, 1] = 50.0
        neighbors_before = [world.T[4, 3, 1], world.T[2, 3, 1]]
        world.step()
        neighbors_after = [world.T[4, 3, 1], world.T[2, 3, 1]]
        assert all(a > b for a, b in zip(neighbors_after, neighbors_before))

    def test_zone_overshoot(self):
        env, world = _make_world()
        room = env.rooms["room_A"]
        world.T[room.slices] = 25.0
        overshoot = world.zone_overshoot("room_A")
        assert overshoot == pytest.approx(3.0, abs=0.1)

    def test_zone_overshoot_within_setpoint(self):
        _, world = _make_world()
        overshoot = world.zone_overshoot("room_A")
        assert overshoot == 0.0

    def test_metrics_dict(self):
        _, world = _make_world()
        world.step()
        m = world.metrics()
        assert "step" in m
        assert "zone_temperatures" in m
        assert "max_overshoot" in m
        assert "cooling_energy_dt" in m

    def test_set_damper(self):
        _, world = _make_world()
        world.set_damper("vav_A", 0.9)
        assert world.env.get_damper_opening("vav_A") == 0.9

    def test_q_total_budget_enforced(self):
        env, world = _make_world()
        env.dampers["vav_A"].max_flow = 100.0
        env.set_damper_opening("vav_A", 1.0)
        world.T[3, 3, 1] = 30.0
        world.step()
        assert world.T[3, 3, 1] < 30.0

    def test_comfort_violation(self):
        env, world = _make_world()
        room = env.rooms["room_A"]
        world.T[room.slices] = 25.0
        cv = world.comfort_violation()
        assert cv > 0

    def test_run_multiple_steps(self):
        _, world = _make_world()
        world.run(10)
        assert world.step_count == 10
        assert world.time > 0
