"""
Benchmark — VDPA (predictive) vs centralized (reactive) comparison.

Runs the same environment with both actuation policies and compares:
- Time-integrated contamination volume (primary metric)
- Response time (detection → actuation)
- Peak concentration in protected rooms
- Contaminated volume over time
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blocks.simulation.simulation import Simulation
from blocks.metrics.collector import MetricsCollector


class Benchmark:
    """Runs VDPA vs centralized reactive comparison on the same environment."""

    def __init__(
        self,
        env_config: str | Path,
        n_steps: int = 500,
        record_every: int = 10,
        *,
        # VDPA predictive settings
        predictive_horizon: float = 5.0,
        gossip_rounds: int = 1,
        proximity_radius: float = 3.0,
        # Reactive settings
        reactive_threshold: float = 0.5,
        # Shared
        detection_threshold: float = 0.01,
        seed: int = 42,
        output_dir: str | Path | None = None,
    ):
        self.env_config = str(env_config)
        self.n_steps = n_steps
        self.record_every = record_every

        self.predictive_horizon = predictive_horizon
        self.gossip_rounds = gossip_rounds
        self.proximity_radius = proximity_radius
        self.reactive_threshold = reactive_threshold
        self.detection_threshold = detection_threshold
        self.seed = seed
        self.output_dir = Path(output_dir) if output_dir else None

        self.predictive_result: MetricsCollector | None = None
        self.reactive_result: MetricsCollector | None = None

    # ── Run ────────────────────────────────────────────────────────────

    def run_predictive(self) -> Simulation:
        """Run simulation with VDPA predictive actuation."""
        sim = Simulation(
            self.env_config,
            actuator_policy="predictive",
            actuator_horizon=self.predictive_horizon,
            gossip_rounds=self.gossip_rounds,
            proximity_radius=self.proximity_radius,
            detection_threshold=self.detection_threshold,
            seed=self.seed,
            name="vdpa_predictive",
        )
        sim.run(self.n_steps, record_every=self.record_every)
        self.predictive_result = sim.collector
        return sim

    def run_reactive(self) -> Simulation:
        """Run simulation with centralized reactive actuation."""
        sim = Simulation(
            self.env_config,
            actuator_policy="reactive",
            concentration_threshold=self.reactive_threshold,
            proximity_radius=self.proximity_radius,
            detection_threshold=self.detection_threshold,
            seed=self.seed,
            name="centralized_reactive",
        )
        sim.run(self.n_steps, record_every=self.record_every)
        self.reactive_result = sim.collector
        return sim

    def run(self) -> dict[str, Any]:
        """Run both policies and return comparison results."""
        pred_sim = self.run_predictive()
        react_sim = self.run_reactive()

        comparison = self.compare(pred_sim, react_sim)

        if self.output_dir:
            self._save_results(comparison)

        return comparison

    # ── Compare ────────────────────────────────────────────────────────

    def compare(
        self, pred_sim: Simulation, react_sim: Simulation
    ) -> dict[str, Any]:
        """Compare results from both simulation runs."""
        pred_a = pred_sim.actuator
        react_a = react_sim.actuator

        pred_contam = pred_sim.cumulative_contamination
        react_contam = react_sim.cumulative_contamination

        improvement = (
            (react_contam - pred_contam) / react_contam * 100
            if react_contam > 0 else 0.0
        )

        return {
            "environment": self.env_config,
            "n_steps": self.n_steps,
            "predictive": {
                "cumulative_contamination": round(pred_contam, 4),
                "response_time": pred_a.response_time,
                "first_detection_time": pred_a.first_detection_time,
                "first_actuation_time": pred_a.first_actuation_time,
                "doors_closed": pred_a.doors_closed,
                "total_mass_final": round(pred_sim.world.total_mass(), 4),
                "peak_concentration": round(float(pred_sim.world.phi.max()), 4),
                "contaminated_volume": pred_sim.world.contaminated_volume(),
            },
            "reactive": {
                "cumulative_contamination": round(react_contam, 4),
                "response_time": react_a.response_time,
                "first_detection_time": react_a.first_detection_time,
                "first_actuation_time": react_a.first_actuation_time,
                "doors_closed": react_a.doors_closed,
                "total_mass_final": round(react_sim.world.total_mass(), 4),
                "peak_concentration": round(float(react_sim.world.phi.max()), 4),
                "contaminated_volume": react_sim.world.contaminated_volume(),
            },
            "comparison": {
                "contamination_reduction_pct": round(improvement, 2),
                "predictive_faster_by": (
                    round(react_a.response_time - pred_a.response_time, 6)
                    if pred_a.response_time is not None and react_a.response_time is not None
                    else None
                ),
            },
        }

    # ── I/O ────────────────────────────────────────────────────────────

    def _save_results(self, comparison: dict) -> None:
        """Save comparison and per-policy results to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Comparison summary
        with open(self.output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        # Per-policy detailed results
        if self.predictive_result:
            self.predictive_result.save_json(
                self.output_dir / "predictive_metrics.json"
            )
            self.predictive_result.save_csv(
                self.output_dir / "predictive_scalars.csv"
            )
        if self.reactive_result:
            self.reactive_result.save_json(
                self.output_dir / "reactive_metrics.json"
            )
            self.reactive_result.save_csv(
                self.output_dir / "reactive_scalars.csv"
            )

    @classmethod
    def from_config(cls, config_path: str | Path) -> Benchmark:
        """Create a Benchmark from a JSON config file.

        Config schema:
        {
            "environment": "path/to/env.json",
            "n_steps": 500,
            "record_every": 10,
            "predictive": {"horizon": 5.0, "gossip_rounds": 1},
            "reactive": {"threshold": 0.5},
            "seed": 42,
            "output_dir": "results/benchmark"
        }
        """
        with open(config_path) as f:
            cfg = json.load(f)

        pred = cfg.get("predictive", {})
        react = cfg.get("reactive", {})

        return cls(
            env_config=cfg["environment"],
            n_steps=cfg.get("n_steps", 500),
            record_every=cfg.get("record_every", 10),
            predictive_horizon=pred.get("horizon", 5.0),
            gossip_rounds=pred.get("gossip_rounds", 1),
            proximity_radius=cfg.get("proximity_radius", 3.0),
            reactive_threshold=react.get("threshold", 0.5),
            detection_threshold=cfg.get("detection_threshold", 0.01),
            seed=cfg.get("seed", 42),
            output_dir=cfg.get("output_dir"),
        )
