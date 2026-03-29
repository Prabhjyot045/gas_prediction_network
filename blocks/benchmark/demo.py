"""
Standalone demo for Block 7 — Aether-Edge vs Centralized Benchmark.

Usage:
    python -m blocks.benchmark.demo --config configs/environments/university_floor.json
    python -m blocks.benchmark.demo --config configs/environments/university_floor.json --steps 500 --save results/benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blocks.benchmark.benchmark import Benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge Block 7 — Benchmark Demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--buffer", type=float, default=30.0)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None, help="Save results to directory")
    p.add_argument("--z-slice", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bm = Benchmark(
        env_config=args.config,
        n_steps=args.steps,
        record_every=5,
        gossip_rounds=2,
        buffer_seconds=args.buffer,
        talk_threshold=args.threshold,
        seed=args.seed,
        output_dir=args.save,
    )

    print("Running Aether-Edge (decentralized) simulation...")
    edge_sim = bm.run_edge()
    print(f"  Done: {edge_sim.summary()}")

    print("\nRunning Centralized (reactive) simulation...")
    cent_sim = bm.run_centralized()
    print(f"  Done: {cent_sim.summary()}")

    comparison = bm.compare(edge_sim, cent_sim)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(comparison, indent=2, default=str))

    if args.save:
        bm._save_results(comparison)
        print(f"\nResults saved to {args.save}/")

    # ── Visualization ──────────────────────────────────────────────────
    z = args.z_slice
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Aether-Edge vs Centralized (Baseline) — Benchmark", fontsize=14)

    # Row 1: Final temperature fields
    for col, (sim, label) in enumerate([
        (edge_sim, "Edge (Predictive)"), (cent_sim, "Centralized (Reactive)")
    ]):
        ax = axes[0, col]
        temp_slice = sim.world.T[:, :, z].T
        wall_slice = sim.env.walls[:, :, z].T
        masked = np.ma.array(temp_slice, mask=wall_slice)
        im = ax.imshow(masked, cmap="hot", origin="lower",
                       extent=(-0.5, sim.env.nx - 0.5, -0.5, sim.env.ny - 0.5), vmax=30.0, vmin=20.0)
        plt.colorbar(im, ax=ax, label="Temperature (C)")

        for damper_name, damper in sim.env.dampers.items():
            cx, cy = damper.position[0], damper.position[1]
            color = "lime" if damper.opening > 0.1 else "red"
            ax.plot(cx, cy, "s", color=color, markersize=10, markeredgecolor="white")

        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")

    # Top-right: comparison summary
    ax = axes[0, 2]
    ax.axis("off")
    e = comparison["edge"]
    c = comparison["centralized"]
    comp = comparison["comparison"]
    text = (
        f"{'Metric':<28} {'Edge':>12} {'Centralized':>12}\n"
        f"{'─' * 54}\n"
        f"{'Cumul. Comfort Violation':<28} {e['cumulative_comfort_violation']:>12.2f} {c['cumulative_comfort_violation']:>12.2f}\n"
        f"{'Cumul. Energy':<28} {e['cumulative_energy']:>12.2f} {c['cumulative_energy']:>12.2f}\n"
        f"{'Max Overshoot (C)':<28} {e['max_overshoot']:>12.2f} {c['max_overshoot']:>12.2f}\n"
        f"{'Mean Age of Info (s)':<28} {e['mean_aoi']:>12.2f} {c['mean_aoi']:>12.2f}\n"
        f"{'Total Messages':<28} {e['total_messages']:>12} {c['total_messages']:>12}\n"
        f"{'─' * 54}\n"
        f"Comfort Improvement: {comp['comfort_improvement_pct']:.1f}%\n"
        f"Energy Savings:      {comp['energy_savings_pct']:.1f}%"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    # Row 2: Time series comparison
    # Cumulative comfort violation
    ax = axes[1, 0]
    s1, v1 = edge_sim.collector.scalar_series("cumulative_comfort_violation")
    s2, v2 = cent_sim.collector.scalar_series("cumulative_comfort_violation")
    ax.plot(s1, v1, "b-", linewidth=2, label="Edge")
    ax.plot(s2, v2, "r--", linewidth=2, label="Centralized")
    ax.set_xlabel("Step")
    ax.set_ylabel("Comfort Violation")
    ax.set_title("Comfort Violation Over Time", fontsize=10)
    ax.legend()

    # Cumulative energy
    ax = axes[1, 1]
    s1, e1 = edge_sim.collector.scalar_series("cumulative_energy")
    s2, e2 = cent_sim.collector.scalar_series("cumulative_energy")
    ax.plot(s1, e1, "b-", linewidth=2, label="Edge")
    ax.plot(s2, e2, "r--", linewidth=2, label="Centralized")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Usage Over Time", fontsize=10)
    ax.legend()

    # Max overshoot
    ax = axes[1, 2]
    s1, o1 = edge_sim.collector.scalar_series("max_overshoot")
    s2, o2 = cent_sim.collector.scalar_series("max_overshoot")
    ax.plot(s1, o1, "b-", linewidth=2, label="Edge")
    ax.plot(s2, o2, "r--", linewidth=2, label="Centralized")
    ax.set_xlabel("Step")
    ax.set_ylabel("Max Overshoot (C)")
    ax.set_title("Peak Temperature Overshoot", fontsize=10)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
