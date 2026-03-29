"""
Standalone demo for the Full HVAC Simulation.

Runs a single simulation with either the edge or centralized policy,
prints a live summary, and saves a 2x2 metrics plot.

Usage:
    python -m blocks.simulation.demo --config configs/environments/university_floor.json
    python -m blocks.simulation.demo --config configs/environments/university_floor.json --policy centralized
    python -m blocks.simulation.demo --config configs/environments/university_floor.json --steps 500 --save results/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from blocks.simulation.simulation import Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge — Full Simulation Demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--policy", choices=["edge", "centralized"], default="edge")
    p.add_argument("--gossip-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None, help="Save results to directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim = Simulation(
        args.config,
        actuator_policy=args.policy,
        gossip_rounds=args.gossip_rounds,
        seed=args.seed,
        name=f"demo_{args.policy}",
    )

    print(sim.env.summary())
    print(f"\nPolicy: {args.policy} | Steps: {args.steps}")
    print(f"Sensors: {sim.network.n_nodes} | Edges: {sim.network.n_edges}")

    sim.run(args.steps, record_every=5)

    print(f"\n{sim.summary()}")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        sim.collector.save_json(out / "metrics.json")
        sim.collector.save_csv(out / "scalars.csv")
        print(f"\nResults saved to {out}/")

    # ── Metrics plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Aether-Edge Simulation — {args.policy}", fontsize=13)

    # Cumulative comfort violation
    ax = axes[0, 0]
    steps, vals = sim.collector.scalar_series("cumulative_comfort_violation")
    ax.plot(steps, vals, "b-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Comfort Violation (°C·s)")
    ax.set_title("Cumulative Comfort Violation", fontsize=10)

    # Cumulative energy
    ax = axes[0, 1]
    steps, vals = sim.collector.scalar_series("cumulative_energy")
    ax.plot(steps, vals, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.set_title("Cumulative Energy Usage", fontsize=10)

    # Max overshoot
    ax = axes[1, 0]
    steps, vals = sim.collector.scalar_series("max_overshoot")
    ax.plot(steps, vals, "orange", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Max Overshoot (°C)")
    ax.set_title("Peak Temperature Overshoot", fontsize=10)

    # Mean temperature
    ax = axes[1, 1]
    steps, vals = sim.collector.scalar_series("mean_temperature")
    ax.plot(steps, vals, "g-", linewidth=2)
    ax.axhline(
        y=sim.env.ambient_temperature, color="gray", linestyle="--",
        linewidth=1, label=f"Ambient ({sim.env.ambient_temperature}°C)",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Mean Temperature", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if args.save:
        fig.savefig(Path(args.save) / "metrics_plot.png", dpi=150, bbox_inches="tight")
        print(f"Metrics plot saved to {args.save}/metrics_plot.png")

    plt.show()


if __name__ == "__main__":
    main()
