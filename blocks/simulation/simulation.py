"""
Simulation — full integration of all Aether-Edge blocks.

Architecture:
  World (physics) → Interface (read) → SensorField (infer) → Interface (actuate) → World

The Interface block is the only component that touches both the physical
environment (World) and the inference network (SensorField). The sensor
network never sees the World directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.sensor.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.interface.interface import EnvironmentInterface
from blocks.metrics.collector import MetricsCollector


class Simulation:
    """Full HVAC simulation driving all blocks in a synchronous loop."""

    def __init__(
        self,
        env_config: str | Path,
        *,
        # Actuator
        actuator_policy: str = "edge",
        proximity_radius: float = 5.0,
        # Sensor field
        gossip_rounds: int = 2,
        talk_threshold: float = 0.01,
        max_hops: int = 10,
        buffer_seconds: float = 30.0,
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
            talk_threshold=talk_threshold,
            max_hops=max_hops,
            buffer_seconds=buffer_seconds,
            seed=seed,
        )
        self.interface = EnvironmentInterface(
            self.env, self.sensor_field,
            policy=actuator_policy,
            proximity_radius=proximity_radius,
            polling_interval=self.env.polling_interval,
            jitter_sigma=self.env.jitter_sigma,
            compute_delay=self.env.compute_delay,
            seed=seed,
        )
        self.collector = MetricsCollector(name)
        self.collector.set_metadata(
            environment=str(env_config),
            policy=actuator_policy,
            gossip_rounds=gossip_rounds,
            seed=seed,
        )

        # Accumulate comfort violation over time
        self._comfort_violation_integral: float = 0.0
        self._energy_integral: float = 0.0

    # ── Run ────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one time step.

        Order: world physics -> interface (read → infer → actuate).
        """
        self.world.step()
        self.interface.step(self.world)
        self._comfort_violation_integral += self.world.comfort_violation()
        self._energy_integral += self.world.total_cooling_energy()

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        step_callback: Callable[[Simulation, int], None] | None = None,
    ) -> MetricsCollector:
        """Run the simulation for n_steps.

        Returns the MetricsCollector with all recorded data.
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
        sensor_m = self.sensor_field.metrics()
        interface_m = self.interface.metrics()

        record = {
            **world_m,
            "temperature_rmse": round(self.interface.temperature_rmse(self.world), 6),
            "n_heating": sensor_m["n_heating"],
            "urgency_coverage": sensor_m["urgency_coverage"],
            "total_gossip_messages": sensor_m["total_messages_sent"],
            "mean_age_of_information": interface_m["mean_age_of_information"],
            "cumulative_comfort_violation": round(self._comfort_violation_integral, 6),
            "cumulative_energy": round(self._energy_integral, 6),
        }

        self.collector.record(record, step=step)

        # Scalar time-series
        self.collector.record_scalar("max_overshoot", world_m["max_overshoot"], step)
        self.collector.record_scalar("total_overshoot", world_m["total_overshoot"], step)
        self.collector.record_scalar("mean_temperature", world_m["mean_temperature"], step)
        self.collector.record_scalar("cumulative_comfort_violation", self._comfort_violation_integral, step)
        self.collector.record_scalar("cumulative_energy", self._energy_integral, step)
        self.collector.record_scalar("temperature_rmse", record["temperature_rmse"], step)

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def cumulative_comfort_violation(self) -> float:
        return self._comfort_violation_integral

    @property
    def cumulative_energy(self) -> float:
        return self._energy_integral

    def summary(self) -> str:
        w = self.world
        iface = self.interface
        return (
            f"Simulation: {w.step_count} steps, t={w.time:.4f}s\n"
            f"  Policy: {iface.policy}\n"
            f"  Max overshoot: {w.max_overshoot():.4f}C\n"
            f"  Cumulative comfort violation: {self._comfort_violation_integral:.4f}\n"
            f"  Cumulative energy: {self._energy_integral:.4f}\n"
            f"  Mean AoI: {iface.mean_age_of_information:.4f}s\n"
            f"  Gossip messages: {iface.total_messages}"
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> Simulation:
        """Create a Simulation from a JSON config file."""
        with open(config_path) as f:
            cfg = json.load(f)

        env_path = cfg["environment"]
        act = cfg.get("actuator", {})
        sf = cfg.get("sensor_field", {})
        sim = cfg.get("simulation", {})

        return cls(
            env_config=env_path,
            actuator_policy=act.get("policy", "edge"),
            proximity_radius=act.get("proximity_radius", 5.0),
            gossip_rounds=sf.get("gossip_rounds", 2),
            talk_threshold=sf.get("talk_threshold", 0.01),
            max_hops=sf.get("max_hops", 10),
            buffer_seconds=sf.get("buffer_seconds", 30.0),
            seed=sim.get("seed", 42),
            name=sim.get("name", "simulation"),
        )
