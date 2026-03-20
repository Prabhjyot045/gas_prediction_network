"""
Unit tests for Block 7 — Benchmark.

Run with:
    python -m pytest blocks/benchmark/test_benchmark.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from blocks.benchmark.benchmark import Benchmark


# ── Helpers ────────────────────────────────────────────────────────────────

def _write_config(config: dict) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config, f)
    f.close()
    return Path(f.name)


def _two_room_config() -> dict:
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
            "communication_radius": 6.0,
            "nodes": [
                {"name": "s_left", "position": {"x": 2, "y": 3, "z": 1}},
                {"name": "s_door", "position": {"x": 4, "y": 3, "z": 1}},
                {"name": "s_right", "position": {"x": 9, "y": 3, "z": 1}},
            ],
        },
        "noise": {"sensor_sigma": 0.0},
    }


@pytest.fixture
def env_path():
    return _write_config(_two_room_config())


# ══════════════════════════════════════════════════════════════════════════
# Benchmark execution
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkRun:
    def test_run_predictive(self, env_path):
        bm = Benchmark(env_path, n_steps=50, record_every=10, seed=42)
        sim = bm.run_predictive()
        assert sim.world.step_count == 50
        assert bm.predictive_result is not None

    def test_run_reactive(self, env_path):
        bm = Benchmark(env_path, n_steps=50, record_every=10, seed=42)
        sim = bm.run_reactive()
        assert sim.world.step_count == 50
        assert bm.reactive_result is not None

    def test_run_both(self, env_path):
        bm = Benchmark(env_path, n_steps=50, record_every=10, seed=42)
        comparison = bm.run()
        assert "predictive" in comparison
        assert "reactive" in comparison
        assert "comparison" in comparison


# ══════════════════════════════════════════════════════════════════════════
# Comparison results
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkComparison:
    def test_comparison_keys(self, env_path):
        bm = Benchmark(env_path, n_steps=100, record_every=20, seed=42)
        comparison = bm.run()

        pred = comparison["predictive"]
        expected_pred = {
            "cumulative_contamination", "response_time",
            "first_detection_time", "first_actuation_time",
            "doors_closed", "total_mass_final",
            "peak_concentration", "contaminated_volume",
        }
        assert expected_pred == set(pred.keys())

        comp = comparison["comparison"]
        assert "contamination_reduction_pct" in comp

    def test_contamination_is_positive(self, env_path):
        bm = Benchmark(env_path, n_steps=100, record_every=100, seed=42)
        comparison = bm.run()
        assert comparison["predictive"]["cumulative_contamination"] >= 0
        assert comparison["reactive"]["cumulative_contamination"] >= 0

    def test_both_detect_gas(self, env_path):
        bm = Benchmark(env_path, n_steps=100, record_every=100, seed=42)
        comparison = bm.run()
        # Gas source is at sensor position, so both should detect
        assert comparison["predictive"]["first_detection_time"] is not None
        assert comparison["reactive"]["first_detection_time"] is not None


# ══════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkIO:
    def test_save_results(self, env_path, tmp_path):
        bm = Benchmark(
            env_path, n_steps=50, record_every=10,
            seed=42, output_dir=tmp_path / "bench_out",
        )
        bm.run()

        assert (tmp_path / "bench_out" / "comparison.json").exists()
        assert (tmp_path / "bench_out" / "predictive_metrics.json").exists()
        assert (tmp_path / "bench_out" / "reactive_metrics.json").exists()

        data = json.loads((tmp_path / "bench_out" / "comparison.json").read_text())
        assert "predictive" in data

    def test_from_config(self, env_path, tmp_path):
        bench_cfg = {
            "environment": str(env_path),
            "n_steps": 30,
            "record_every": 10,
            "predictive": {"horizon": 3.0, "gossip_rounds": 2},
            "reactive": {"threshold": 0.3},
            "seed": 99,
        }
        cfg_path = tmp_path / "bench_config.json"
        cfg_path.write_text(json.dumps(bench_cfg))

        bm = Benchmark.from_config(cfg_path)
        assert bm.n_steps == 30
        assert bm.predictive_horizon == 3.0
        assert bm.reactive_threshold == 0.3
        assert bm.seed == 99
