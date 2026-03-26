"""
Benchmark — Aether-Edge (decentralized) vs Centralized comparison.

Runs the same HVAC environment with both actuation policies and compares:
- Overshoot Error (max degrees above setpoint)
- Settling Time (steps until zones within tolerance)
- Comfort Violation (time-integrated overshoot)
- Energy Usage (total cooling energy)
- Packet Overhead (gossip messages sent)
- Age of Information (staleness of sensor data)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blocks.simulation.simulation import Simulation
from blocks.metrics.collector import MetricsCollector


class Benchmark:
    """Runs edge vs centralized comparison on the same HVAC environment."""

    def __init__(
        self,
        env_config: str | Path,
        n_steps: int = 500,
        record_every: int = 10,
        *,
        # Shared
        gossip_rounds: int = 2,
        proximity_radius: float = 5.0,
        talk_threshold: float = 0.01,
        buffer_seconds: float = 30.0,
        seed: int = 42,
        output_dir: str | Path | None = None,
    ):
        self.env_config = str(env_config)
        self.n_steps = n_steps
        self.record_every = record_every

        self.gossip_rounds = gossip_rounds
        self.proximity_radius = proximity_radius
        self.talk_threshold = talk_threshold
        self.buffer_seconds = buffer_seconds
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None

        self.edge_result: MetricsCollector | None = None
        self.centralized_result: MetricsCollector | None = None

    # ── Run ────────────────────────────────────────────────────────────

    def run_edge(self) -> Simulation:
        """Run simulation with Aether-Edge decentralized actuation."""
        sim = Simulation(
            self.env_config,
            actuator_policy="edge",
            gossip_rounds=self.gossip_rounds,
            proximity_radius=self.proximity_radius,
            talk_threshold=self.talk_threshold,
            buffer_seconds=self.buffer_seconds,
            seed=self.seed,
            name="aether_edge",
        )
        sim.run(self.n_steps, record_every=self.record_every)
        self.edge_result = sim.collector
        return sim

    def run_centralized(self) -> Simulation:
        """Run simulation with centralized reactive actuation."""
        sim = Simulation(
            self.env_config,
            actuator_policy="centralized",
            proximity_radius=self.proximity_radius,
            talk_threshold=self.talk_threshold,
            buffer_seconds=self.buffer_seconds,
            seed=self.seed,
            name="centralized",
        )
        sim.run(self.n_steps, record_every=self.record_every)
        self.centralized_result = sim.collector
        return sim

    def run(self) -> dict[str, Any]:
        """Run both policies and return comparison results."""
        edge_sim = self.run_edge()
        cent_sim = self.run_centralized()

        comparison = self.compare(edge_sim, cent_sim)

        if self.output_dir:
            self._save_results(comparison)

        return comparison

    # ── Compare ────────────────────────────────────────────────────────

    def compare(
        self, edge_sim: Simulation, cent_sim: Simulation
    ) -> dict[str, Any]:
        """Compare results from both simulation runs."""
        edge_cv = edge_sim.cumulative_comfort_violation
        cent_cv = cent_sim.cumulative_comfort_violation

        comfort_improvement = (
            (cent_cv - edge_cv) / cent_cv * 100
            if cent_cv > 0 else 0.0
        )

        edge_energy = edge_sim.cumulative_energy
        cent_energy = cent_sim.cumulative_energy

        energy_savings = (
            (cent_energy - edge_energy) / cent_energy * 100
            if cent_energy > 0 else 0.0
        )

        return {
            "environment": self.env_config,
            "n_steps": self.n_steps,
            "edge": {
                "cumulative_comfort_violation": round(edge_cv, 4),
                "cumulative_energy": round(edge_energy, 4),
                "max_overshoot": round(edge_sim.world.max_overshoot(), 4),
                "mean_aoi": round(edge_sim.actuator.mean_age_of_information, 4),
                "total_messages": edge_sim.actuator.total_messages,
                "zone_temperatures": {
                    z: round(edge_sim.world.zone_mean_temperature(z), 4)
                    for z in edge_sim.env.rooms
                },
            },
            "centralized": {
                "cumulative_comfort_violation": round(cent_cv, 4),
                "cumulative_energy": round(cent_energy, 4),
                "max_overshoot": round(cent_sim.world.max_overshoot(), 4),
                "mean_aoi": round(cent_sim.actuator.mean_age_of_information, 4),
                "total_messages": cent_sim.actuator.total_messages,
                "zone_temperatures": {
                    z: round(cent_sim.world.zone_mean_temperature(z), 4)
                    for z in cent_sim.env.rooms
                },
            },
            "comparison": {
                "comfort_improvement_pct": round(comfort_improvement, 2),
                "energy_savings_pct": round(energy_savings, 2),
                "edge_aoi": round(edge_sim.actuator.mean_age_of_information, 4),
                "centralized_aoi": round(cent_sim.actuator.mean_age_of_information, 4),
                "edge_messages": edge_sim.actuator.total_messages,
                "centralized_messages": cent_sim.actuator.total_messages,
            },
        }

    # ── I/O ────────────────────────────────────────────────────────────

    def _save_results(self, comparison: dict) -> None:
        """Save comparison and per-policy results to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        if self.edge_result:
            self.edge_result.save_json(self.output_dir / "edge_metrics.json")
            self.edge_result.save_csv(self.output_dir / "edge_scalars.csv")
        if self.centralized_result:
            self.centralized_result.save_json(self.output_dir / "centralized_metrics.json")
            self.centralized_result.save_csv(self.output_dir / "centralized_scalars.csv")

    @classmethod
    def from_config(cls, config_path: str | Path) -> Benchmark:
        """Create a Benchmark from a JSON config file."""
        with open(config_path) as f:
            cfg = json.load(f)

        return cls(
            env_config=cfg["environment"],
            n_steps=cfg.get("n_steps", 500),
            record_every=cfg.get("record_every", 10),
            gossip_rounds=cfg.get("gossip_rounds", 2),
            proximity_radius=cfg.get("proximity_radius", 5.0),
            talk_threshold=cfg.get("talk_threshold", 0.01),
            buffer_seconds=cfg.get("buffer_seconds", 30.0),
            seed=cfg.get("seed", 42),
            output_dir=cfg.get("output_dir"),
        )
