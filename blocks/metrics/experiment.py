"""
ExperimentRunner — drives parameter sweeps from JSON config files.

Generates the Cartesian product of parameter values, patches the base
environment config, and runs a user-provided function for each combination.
Results are collected into a JSON file for downstream analysis.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Callable

from .collector import MetricsCollector


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation (e.g. 'sensors.spacing')."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


class ExperimentRunner:
    """Runs parameter sweeps from a JSON experiment config.

    Experiment config format:
    {
        "name": "my_sweep",
        "base_environment": "configs/environments/default_maze.json",
        "parameters": {
            "sensors.spacing": [2, 3, 5],
            "sensors.communication_radius": [3.0, 5.0]
        },
        "simulation": {
            "n_steps": 500,
            "metrics_every": 50
        },
        "output_dir": "results/my_sweep"
    }
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config = json.load(f)

        self.name: str = self.config.get("name", "experiment")
        self.output_dir = Path(self.config.get("output_dir", f"results/{self.name}"))
        self.sim_config: dict = self.config.get("simulation", {})

        # Load base environment config
        base_path = Path(self.config["base_environment"])
        with open(base_path) as f:
            self._base_env_config: dict = json.load(f)

        # Generate parameter grid
        self._param_grid = self._build_grid()

    def _build_grid(self) -> list[dict[str, Any]]:
        """Generate Cartesian product of parameter values."""
        params = self.config.get("parameters", {})
        if not params:
            return [{}]

        keys = list(params.keys())
        values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
        combos = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combos]

    @property
    def n_runs(self) -> int:
        return len(self._param_grid)

    def make_env_config(self, params: dict[str, Any]) -> dict:
        """Patch the base environment config with the given parameter values."""
        config = copy.deepcopy(self._base_env_config)
        for key, value in params.items():
            _set_nested(config, key, value)
        return config

    def run(
        self,
        run_fn: Callable[[dict, dict, MetricsCollector], None],
    ) -> list[dict[str, Any]]:
        """Execute run_fn for each parameter combination.

        Args:
            run_fn: Called with (env_config_dict, sim_config, collector).
                    The function should build Environment/World/etc from
                    the config dict, run the simulation, and push metrics
                    into the collector.

        Returns:
            List of result dicts (one per run), also saved to output_dir.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for i, params in enumerate(self._param_grid):
            print(f"  Run {i+1}/{self.n_runs}: {params}")

            env_config = self.make_env_config(params)
            collector = MetricsCollector(name=f"{self.name}_run_{i}")
            collector.set_metadata(
                run_index=i,
                parameters=params,
                simulation=self.sim_config,
            )

            run_fn(env_config, self.sim_config, collector)

            result = {
                "run_index": i,
                "parameters": params,
                "final_metrics": collector.latest(),
                "n_records": len(collector.records),
            }
            all_results.append(result)

            # Save individual run
            collector.save_json(self.output_dir / f"run_{i:03d}.json")

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "name": self.name,
                "n_runs": self.n_runs,
                "parameter_grid": self._param_grid,
                "results": all_results,
            }, f, indent=2, default=str)

        print(f"Results saved to {self.output_dir}/")
        return all_results
