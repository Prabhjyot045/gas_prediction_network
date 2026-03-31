#!/usr/bin/env python
"""
Aether-Edge Benchmark Runner

Runs both Edge (decentralized) and Centralized (reactive) simulations,
generates comparison results and animated heatmap GIFs for each policy.

Outputs saved to results/benchmark/:
  - comparison.json           Full metrics comparison
  - edge_metrics.json         Per-step edge metrics
  - edge_scalars.csv          Edge scalar time-series
  - centralized_metrics.json  Per-step centralized metrics
  - centralized_scalars.csv   Centralized scalar time-series
  - edge_heatmap.gif          Animated heatmap of edge simulation
  - centralized_heatmap.gif   Animated heatmap of centralized simulation
  - comparison_chart.png      Side-by-side metrics comparison chart

Usage:
    python run_benchmark.py
    python run_benchmark.py --config configs/environments/university_floor.json
    python run_benchmark.py --steps 600 --frames 200
    python run_benchmark.py --no-gif   # skip GIF generation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from blocks.simulation.simulation import Simulation
from blocks.benchmark.benchmark import Benchmark
from blocks.visualization.heatmap_anim import render_heatmap_gif


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge Benchmark Runner")
    p.add_argument(
        "--config", type=str,
        default="configs/environments/university_floor.json",
        help="Environment config file",
    )
    p.add_argument("--steps", type=int, default=400, help="Simulation steps")
    p.add_argument("--frames", type=int, default=150, help="GIF animation frames")
    p.add_argument("--steps-per-frame", type=int, default=3, help="Sim steps per GIF frame")
    p.add_argument("--pre-steps", type=int, default=0, help="Warm-up steps before recording GIF")
    p.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    p.add_argument("--z-slice", type=int, default=1, help="Z-layer to visualize")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results (default: results/<config_name>_<timestamp>)",
    )
    p.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    return p.parse_args()


# ── Comparison chart ─────────────────────────────────────────────────


def generate_comparison_chart(
    comparison: dict,
    edge_collector,
    cent_collector,
    output_path: Path,
) -> None:
    """Generate a 2x3 comparison chart and save as PNG."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Aether-Edge vs Centralized (Baseline) \u2014 Benchmark Results",
        fontsize=14, fontweight="bold",
    )

    e = comparison["edge"]
    c = comparison["centralized"]

    # Row 1: Bar charts for key scalars
    metrics = ["Comfort\nViolation", "Energy", "Messages"]
    edge_vals = [e["cumulative_comfort_violation"], e["cumulative_energy"], e["total_messages"]]
    cent_vals = [c["cumulative_comfort_violation"], c["cumulative_energy"], c["total_messages"]]

    for i, (metric, ev, cv) in enumerate(zip(metrics, edge_vals, cent_vals)):
        ax = axes[0, i]
        bars = ax.bar(["Edge", "Centralized"], [ev, cv], color=["#2196F3", "#F44336"], width=0.5)
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel("Value")
        for bar, val in zip(bars, [ev, cv]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Row 2: Time-series comparisons
    series_configs = [
        ("cumulative_comfort_violation", "Comfort Violation Over Time", "Comfort Violation"),
        ("cumulative_energy", "Energy Usage Over Time", "Energy"),
        ("max_overshoot", "Peak Temperature Overshoot", "Max Overshoot (\u00b0C)"),
    ]

    for col, (scalar_name, title, ylabel) in enumerate(series_configs):
        ax = axes[1, col]
        s1, v1 = edge_collector.scalar_series(scalar_name)
        s2, v2 = cent_collector.scalar_series(scalar_name)
        if s1 and s2:
            ax.plot(s1, v1, "b-", linewidth=2, label="Edge")
            ax.plot(s2, v2, "r--", linewidth=2, label="Centralized")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend()

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        config_stem = Path(args.config).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"{config_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AETHER-EDGE BENCHMARK")
    print("=" * 60)
    print(f"Config:    {args.config}")
    print(f"Steps:     {args.steps}")
    print(f"Output:    {output_dir}/")
    print()

    # ── Step 1: Run benchmark (JSON/CSV metrics) ────────────────────
    print("[1/3] Running benchmark simulations...")
    bm = Benchmark(
        env_config=args.config,
        n_steps=args.steps,
        record_every=5,
        seed=args.seed,
        output_dir=output_dir,
    )
    comparison = bm.run()

    e = comparison["edge"]
    c = comparison["centralized"]
    comp = comparison["comparison"]

    print(f"  Edge:        comfort={e['cumulative_comfort_violation']:.2f}  "
          f"energy={e['cumulative_energy']:.1f}  AoI={e['mean_aoi']:.1f}s  "
          f"msgs={e['total_messages']}")
    print(f"  Centralized: comfort={c['cumulative_comfort_violation']:.2f}  "
          f"energy={c['cumulative_energy']:.1f}  AoI={c['mean_aoi']:.1f}s  "
          f"msgs={c['total_messages']}")
    print(f"  Improvement: comfort={comp['comfort_improvement_pct']:.1f}%  "
          f"energy={comp['energy_savings_pct']:.1f}%")
    print()

    # ── Step 2: Generate comparison chart ───────────────────────────
    print("[2/3] Generating comparison chart...")
    generate_comparison_chart(
        comparison, bm.edge_result, bm.centralized_result,
        output_dir / "comparison_chart.png",
    )
    print()

    # ── Step 3: Generate heatmap GIFs ───────────────────────────────
    if args.no_gif:
        print("[3/3] Skipping GIF generation (--no-gif)")
    else:
        print("[3/3] Generating heatmap GIFs...")
        gif_kwargs = dict(
            n_frames=args.frames,
            steps_per_frame=args.steps_per_frame,
            z_slice=args.z_slice,
            fps=args.fps,
        )

        for policy, label, gif_name in [
            ("edge", "Aether-Edge (Decentralized)", "edge_heatmap.gif"),
            ("centralized", "Centralized (Reactive)", "centralized_heatmap.gif"),
        ]:
            print(f"  {label}...")
            sim = Simulation(args.config, actuator_policy=policy, seed=args.seed)
            for _ in range(args.pre_steps):
                sim.step()
            render_heatmap_gif(
                sim,
                output_path=output_dir / gif_name,
                policy_label=label,
                **gif_kwargs,
            )

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name:<30} {size_str:>10}")

    print()
    print(json.dumps(comparison, indent=2, default=str))


if __name__ == "__main__":
    main()
