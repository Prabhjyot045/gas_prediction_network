"""
Unit tests for Block 7 — Benchmark (Edge vs Centralized).

Run with:
    python -m pytest blocks/benchmark/test_benchmark.py -v
"""

from __future__ import annotations

import json
import tempfile

import pytest

from blocks.benchmark.benchmark import Benchmark


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_config() -> dict:
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


def _write_config(cfg: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


def _make_benchmark(n_steps=50) -> Benchmark:
    path = _write_config(_make_config())
    return Benchmark(path, n_steps=n_steps, record_every=10, seed=42)


# ══════════════════════════════════════════════════════════════════════════
# Run tests
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkRun:
    def test_run_completes(self):
        b = _make_benchmark()
        result = b.run()
        assert "edge" in result
        assert "centralized" in result
        assert "comparison" in result

    def test_both_policies_produce_metrics(self):
        b = _make_benchmark()
        result = b.run()
        assert result["edge"]["cumulative_energy"] > 0
        assert result["centralized"]["cumulative_energy"] > 0

    def test_independent_worlds(self):
        b = _make_benchmark()
        edge_sim = b.run_edge()
        cent_sim = b.run_centralized()
        assert edge_sim.world is not cent_sim.world


# ══════════════════════════════════════════════════════════════════════════
# Comparison tests
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkComparison:
    def test_comparison_has_metrics(self):
        b = _make_benchmark()
        result = b.run()
        comp = result["comparison"]
        assert "comfort_improvement_pct" in comp
        assert "energy_savings_pct" in comp
        assert "edge_aoi" in comp
        assert "centralized_aoi" in comp

    def test_edge_has_zero_aoi(self):
        b = _make_benchmark()
        result = b.run()
        assert result["edge"]["mean_aoi"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# I/O tests
# ══════════════════════════════════════════════════════════════════════════

class TestBenchmarkIO:
    def test_save_results(self, tmp_path):
        b = _make_benchmark()
        b.output_dir = tmp_path / "results"
        b.run()
        assert (tmp_path / "results" / "comparison.json").exists()
        assert (tmp_path / "results" / "edge_metrics.json").exists()
        assert (tmp_path / "results" / "centralized_metrics.json").exists()

    def test_from_config(self):
        bench_cfg = {
            "environment": _write_config(_make_config()),
            "n_steps": 20,
            "record_every": 10,
            "seed": 42,
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(bench_cfg, f)
        f.close()
        b = Benchmark.from_config(f.name)
        result = b.run()
        assert "comparison" in result
