"""
Standalone demo for Block 7 — VDPA vs Centralized Benchmark.

Usage:
    python -m blocks.benchmark.demo --config configs/environments/default_maze.json
    python -m blocks.benchmark.demo --config configs/environments/default_maze.json --steps 500 --save results/benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.benchmark.benchmark import Benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 7 — Benchmark Demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None, help="Save results to directory")
    p.add_argument("--z-slice", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bm = Benchmark(
        env_config=args.config,
        n_steps=args.steps,
        record_every=5,
        predictive_horizon=args.horizon,
        gossip_rounds=2,
        reactive_threshold=args.threshold,
        seed=args.seed,
        output_dir=args.save,
    )

    print("Running VDPA (predictive) simulation...")
    pred_sim = bm.run_predictive()
    print(f"  Done: {pred_sim.summary()}")

    print("\nRunning centralized (reactive) simulation...")
    react_sim = bm.run_reactive()
    print(f"  Done: {react_sim.summary()}")

    comparison = bm.compare(pred_sim, react_sim)

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
    fig.suptitle("VDPA vs Centralized Reactive — Benchmark", fontsize=14)

    # Row 1: Final concentration fields
    for col, (sim, label) in enumerate([
        (pred_sim, "VDPA (Predictive)"), (react_sim, "Centralized (Reactive)")
    ]):
        ax = axes[0, col]
        phi_slice = sim.world.phi[:, :, z].T
        wall_slice = sim.env.walls[:, :, z].T
        masked = np.ma.array(phi_slice, mask=wall_slice)
        im = ax.imshow(masked, cmap="hot", origin="lower",
                       extent=(-0.5, sim.env.nx - 0.5, -0.5, sim.env.ny - 0.5))
        plt.colorbar(im, ax=ax, label="Concentration")

        for door_name, door in sim.env.doors.items():
            s = door.slices
            cx, cy = (s[0].start + s[0].stop) / 2, (s[1].start + s[1].stop) / 2
            color = "red" if door.state == "closed" else "lime"
            ax.plot(cx, cy, "s", color=color, markersize=10, markeredgecolor="white")

        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")

    # Top-right: comparison summary
    ax = axes[0, 2]
    ax.axis("off")
    p = comparison["predictive"]
    r = comparison["reactive"]
    c = comparison["comparison"]
    text = (
        f"{'Metric':<28} {'Predictive':>12} {'Reactive':>12}\n"
        f"{'─' * 54}\n"
        f"{'Cumul. Contamination':<28} {p['cumulative_contamination']:>12.2f} {r['cumulative_contamination']:>12.2f}\n"
        f"{'Response Time (s)':<28} {str(p['response_time'] or 'N/A'):>12} {str(r['response_time'] or 'N/A'):>12}\n"
        f"{'First Detection (s)':<28} {str(p['first_detection_time'] or 'N/A'):>12} {str(r['first_detection_time'] or 'N/A'):>12}\n"
        f"{'First Actuation (s)':<28} {str(p['first_actuation_time'] or 'N/A'):>12} {str(r['first_actuation_time'] or 'N/A'):>12}\n"
        f"{'Doors Closed':<28} {p['doors_closed']:>12} {r['doors_closed']:>12}\n"
        f"{'Final Total Mass':<28} {p['total_mass_final']:>12.2f} {r['total_mass_final']:>12.2f}\n"
        f"{'Contaminated Volume':<28} {p['contaminated_volume']:>12} {r['contaminated_volume']:>12}\n"
        f"{'─' * 54}\n"
        f"Contamination Reduction: {c['contamination_reduction_pct']:.1f}%"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    # Row 2: Time series comparison
    # Cumulative contamination
    ax = axes[1, 0]
    s1, c1 = pred_sim.collector.scalar_series("cumulative_contamination")
    s2, c2 = react_sim.collector.scalar_series("cumulative_contamination")
    ax.plot(s1, c1, "b-", linewidth=2, label="Predictive")
    ax.plot(s2, c2, "r--", linewidth=2, label="Reactive")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Contamination")
    ax.set_title("Contamination Over Time", fontsize=10)
    ax.legend()

    # Total mass
    ax = axes[1, 1]
    s1, m1 = pred_sim.collector.scalar_series("total_mass")
    s2, m2 = react_sim.collector.scalar_series("total_mass")
    ax.plot(s1, m1, "b-", linewidth=2, label="Predictive")
    ax.plot(s2, m2, "r--", linewidth=2, label="Reactive")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Mass")
    ax.set_title("Total Mass Over Time", fontsize=10)
    ax.legend()

    # Contaminated volume
    ax = axes[1, 2]
    s1, v1 = pred_sim.collector.scalar_series("contaminated_volume")
    s2, v2 = react_sim.collector.scalar_series("contaminated_volume")
    ax.plot(s1, v1, "b-", linewidth=2, label="Predictive")
    ax.plot(s2, v2, "r--", linewidth=2, label="Reactive")
    ax.set_xlabel("Step")
    ax.set_ylabel("Contaminated Cells")
    ax.set_title("Contaminated Volume Over Time", fontsize=10)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
