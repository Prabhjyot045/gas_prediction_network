"""
Unit tests for Metrics infrastructure.

Run with:
    python -m pytest blocks/metrics/test_metrics.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from blocks.metrics.collector import MetricsCollector
from blocks.metrics.experiment import ExperimentRunner, _set_nested


# ══════════════════════════════════════════════════════════════════════════
# MetricsCollector tests
# ══════════════════════════════════════════════════════════════════════════

class TestMetricsCollector:
    def test_empty_collector(self):
        mc = MetricsCollector("test")
        assert mc.name == "test"
        assert len(mc.records) == 0
        assert mc.latest() == {}

    def test_record_and_latest(self):
        mc = MetricsCollector()
        mc.record({"mass": 10.0, "contam": 5}, step=1)
        mc.record({"mass": 20.0, "contam": 12}, step=2)
        assert len(mc.records) == 2
        assert mc.latest()["mass"] == 20.0

    def test_record_scalar(self):
        mc = MetricsCollector()
        mc.record_scalar("mass", 10.0, step=0)
        mc.record_scalar("mass", 20.0, step=1)
        mc.record_scalar("mass", 30.0, step=2)
        steps, values = mc.scalar_series("mass")
        assert steps == [0, 1, 2]
        assert values == [10.0, 20.0, 30.0]

    def test_scalar_names(self):
        mc = MetricsCollector()
        mc.record_scalar("mass", 1.0, step=0)
        mc.record_scalar("contam", 2.0, step=0)
        assert sorted(mc.scalar_names()) == ["contam", "mass"]

    def test_metadata(self):
        mc = MetricsCollector()
        mc.set_metadata(config="test.json", spacing=3)
        snap = mc.snapshot()
        assert snap["metadata"]["config"] == "test.json"

    def test_snapshot_structure(self):
        mc = MetricsCollector("exp1")
        mc.record({"a": 1}, step=0)
        mc.record_scalar("b", 2.0, step=0)
        snap = mc.snapshot()
        assert snap["name"] == "exp1"
        assert "records" in snap
        assert "scalars" in snap
        assert "elapsed_seconds" in snap

    def test_save_json(self, tmp_path):
        mc = MetricsCollector("test")
        mc.record({"x": 1}, step=0)
        path = mc.save_json(tmp_path / "out.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "test"
        assert len(data["records"]) == 1

    def test_save_csv(self, tmp_path):
        mc = MetricsCollector()
        mc.record_scalar("mass", 10.0, step=0)
        mc.record_scalar("mass", 20.0, step=1)
        mc.record_scalar("contam", 5.0, step=0)
        mc.record_scalar("contam", 8.0, step=1)
        path = mc.save_csv(tmp_path / "out.csv")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "mass" in lines[0]


# ══════════════════════════════════════════════════════════════════════════
# ExperimentRunner tests
# ══════════════════════════════════════════════════════════════════════════

class TestExperimentRunner:
    def _write_experiment_config(self, tmp_path, base_env_path) -> Path:
        config = {
            "name": "test_sweep",
            "base_environment": str(base_env_path),
            "parameters": {
                "sensors.spacing": [2, 4],
                "sensors.communication_radius": [3.0, 5.0],
            },
            "simulation": {"n_steps": 10},
            "output_dir": str(tmp_path / "results"),
        }
        path = tmp_path / "experiment.json"
        path.write_text(json.dumps(config))
        return path

    def _write_base_env(self, tmp_path) -> Path:
        env_config = {
            "grid": {"nx": 10, "ny": 10, "nz": 3, "dx": 1.0},
            "physics": {"diffusion_coefficient": 0.05},
            "rooms": [{"name": "box", "bounds": {
                "x_min": 1, "x_max": 9, "y_min": 1, "y_max": 9,
                "z_min": 0, "z_max": 3
            }}],
            "sensors": {"placement": "grid", "spacing": 3,
                        "z_levels": [1], "communication_radius": 5.0},
        }
        path = tmp_path / "env.json"
        path.write_text(json.dumps(env_config))
        return path

    def test_grid_size(self, tmp_path):
        env_path = self._write_base_env(tmp_path)
        exp_path = self._write_experiment_config(tmp_path, env_path)
        runner = ExperimentRunner(exp_path)
        # 2 spacings x 2 radii = 4 runs
        assert runner.n_runs == 4

    def test_make_env_config_patches(self, tmp_path):
        env_path = self._write_base_env(tmp_path)
        exp_path = self._write_experiment_config(tmp_path, env_path)
        runner = ExperimentRunner(exp_path)
        patched = runner.make_env_config({"sensors.spacing": 7})
        assert patched["sensors"]["spacing"] == 7

    def test_run_executes_all(self, tmp_path):
        env_path = self._write_base_env(tmp_path)
        exp_path = self._write_experiment_config(tmp_path, env_path)
        runner = ExperimentRunner(exp_path)

        call_count = 0
        def dummy_run(env_config, sim_config, collector):
            nonlocal call_count
            call_count += 1
            collector.record({"done": True}, step=0)

        results = runner.run(dummy_run)
        assert call_count == 4
        assert len(results) == 4
        # Check output files
        assert (tmp_path / "results" / "summary.json").exists()
        assert (tmp_path / "results" / "run_000.json").exists()


class TestSetNested:
    def test_single_level(self):
        d = {"a": 1}
        _set_nested(d, "a", 2)
        assert d["a"] == 2

    def test_nested(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.b", 99)
        assert d["a"]["b"] == 99

    def test_creates_intermediate(self):
        d = {}
        _set_nested(d, "a.b.c", 42)
        assert d["a"]["b"]["c"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
