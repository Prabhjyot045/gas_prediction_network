"""
Simulation — full integration of all VDPA blocks.

Ties together World, SensorNetwork, SensorField, ActuatorController, and
MetricsCollector into a single simulation loop. Supports both VDPA
(predictive) and centralized (reactive) actuation policies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.network.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.actuator.controller import ActuatorController
from blocks.metrics.collector import MetricsCollector


class Simulation:
    """Full VDPA simulation driving all blocks in a synchronous loop."""

    def __init__(
        self,
        env_config: str | Path,
        *,
        # Actuator
        actuator_policy: str = "predictive",
        actuator_horizon: float = 5.0,
        concentration_threshold: float = 0.5,
        proximity_radius: float = 3.0,
        # Sensor field
        gossip_rounds: int = 1,
        detection_threshold: float = 0.01,
        max_hops: int = 10,
        process_noise_var: float = 0.01,
        seed: int | None = 42,
        # Network
        comm_radius: float | None = None,
        # Metrics
        name: str = "simulation",
    ):
        self.env = Environment(env_config)
        self.world = World(self.env)
        self.network = SensorNetwork(self.env, comm_radius=comm_radius)
        self.sensor_field = SensorField(
            self.env, self.network,
            gossip_rounds=gossip_rounds,
            detection_threshold=detection_threshold,
            max_hops=max_hops,
            process_noise_var=process_noise_var,
            seed=seed,
        )
        self.actuator = ActuatorController(
            self.env, self.sensor_field,
            policy=actuator_policy,
            horizon=actuator_horizon,
            concentration_threshold=concentration_threshold,
            proximity_radius=proximity_radius,
        )
        self.collector = MetricsCollector(name)
        self.collector.set_metadata(
            environment=str(env_config),
            policy=actuator_policy,
            horizon=actuator_horizon,
            concentration_threshold=concentration_threshold,
            gossip_rounds=gossip_rounds,
            seed=seed,
        )

        # Track cumulative contamination integral
        self._contamination_integral: float = 0.0

    # ── Run ────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one time step.

        Order: world physics → sensor sensing → actuator evaluation.
        """
        self.world.step()
        self.sensor_field.step(self.world)
        self.actuator.evaluate(self.world)
        self._contamination_integral += self.world.contamination_integral()

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        step_callback: Callable[[Simulation, int], None] | None = None,
    ) -> MetricsCollector:
        """Run the simulation for n_steps.

        Args:
            n_steps: Number of simulation steps.
            record_every: Record metrics every N steps.
            step_callback: Optional callback(sim, step) called each step.

        Returns:
            The MetricsCollector with all recorded data.
        """
        for i in range(n_steps):
            self.step()

            if step_callback is not None:
                step_callback(self, i)

            if (i + 1) % record_every == 0 or i == n_steps - 1:
                self._record_metrics()

        return self.collector

    def _record_metrics(self) -> None:
        """Record a snapshot of metrics from all blocks."""
        step = self.world.step_count
        world_m = self.world.metrics()
        sensor_m = self.sensor_field.metrics(self.world)
        actuator_m = self.actuator.metrics()

        record = {
            **world_m,
            "concentration_rmse": sensor_m.get("concentration_rmse"),
            "n_detecting": sensor_m["n_detecting"],
            "prediction_coverage": sensor_m["prediction_coverage"],
            "total_gossip_messages": sensor_m["total_messages_sent"],
            "doors_closed": actuator_m["doors_closed"],
            "cumulative_contamination": round(self._contamination_integral, 6),
        }

        self.collector.record(record, step=step)

        # Also record key scalars for easy time-series plotting
        self.collector.record_scalar("total_mass", world_m["total_mass"], step)
        self.collector.record_scalar("contaminated_volume", world_m["contaminated_volume"], step)
        self.collector.record_scalar("cumulative_contamination", self._contamination_integral, step)
        self.collector.record_scalar("peak_concentration", world_m["peak_concentration"], step)
        if sensor_m.get("concentration_rmse") is not None:
            self.collector.record_scalar("concentration_rmse", sensor_m["concentration_rmse"], step)
        self.collector.record_scalar("prediction_coverage", sensor_m["prediction_coverage"], step)
        self.collector.record_scalar("n_detecting", sensor_m["n_detecting"], step)

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def cumulative_contamination(self) -> float:
        """Time-integrated contamination volume (primary benchmark metric)."""
        return self._contamination_integral

    def summary(self) -> str:
        """Human-readable summary of current simulation state."""
        w = self.world
        a = self.actuator
        return (
            f"Simulation: {w.step_count} steps, t={w.time:.4f}s\n"
            f"  Policy: {a.policy}\n"
            f"  Total mass: {w.total_mass():.4f}\n"
            f"  Contaminated volume: {w.contaminated_volume()}\n"
            f"  Cumulative contamination: {self._contamination_integral:.4f}\n"
            f"  Doors closed: {a.doors_closed}/{len(self.env.doors)}\n"
            f"  Response time: {a.response_time}"
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> Simulation:
        """Create a Simulation from a JSON config file.

        Config schema:
        {
            "environment": "path/to/env.json",
            "actuator": {"policy": "predictive", "horizon": 5.0, ...},
            "sensor_field": {"gossip_rounds": 1, ...},
            "simulation": {"seed": 42, "name": "my_run"}
        }
        """
        with open(config_path) as f:
            cfg = json.load(f)

        env_path = cfg["environment"]
        act = cfg.get("actuator", {})
        sf = cfg.get("sensor_field", {})
        sim = cfg.get("simulation", {})

        return cls(
            env_config=env_path,
            actuator_policy=act.get("policy", "predictive"),
            actuator_horizon=act.get("horizon", 5.0),
            concentration_threshold=act.get("concentration_threshold", 0.5),
            proximity_radius=act.get("proximity_radius", 3.0),
            gossip_rounds=sf.get("gossip_rounds", 1),
            detection_threshold=sf.get("detection_threshold", 0.01),
            max_hops=sf.get("max_hops", 10),
            seed=sim.get("seed", 42),
            name=sim.get("name", "simulation"),
        )
