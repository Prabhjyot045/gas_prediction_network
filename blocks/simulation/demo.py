"""
Standalone demo for Block 6 — Full Simulation.

Usage:
    python -m blocks.simulation.demo --config configs/environments/default_maze.json
    python -m blocks.simulation.demo --config configs/environments/default_maze.json --policy reactive
    python -m blocks.simulation.demo --config configs/environments/default_maze.json --steps 500 --save results/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.simulation.simulation import Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 6 — Full Simulation Demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--policy", choices=["predictive", "reactive"], default="predictive")
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gossip-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None, help="Save results to directory")
    p.add_argument("--z-slice", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim = Simulation(
        args.config,
        actuator_policy=args.policy,
        actuator_horizon=args.horizon,
        concentration_threshold=args.threshold,
        gossip_rounds=args.gossip_rounds,
        seed=args.seed,
        name=f"demo_{args.policy}",
    )

    print(sim.env.summary())
    print(f"\nPolicy: {args.policy} | Steps: {args.steps}")
    print(f"Sensors: {sim.network.n_nodes} | Edges: {sim.network.n_edges}")

    sim.run(args.steps, record_every=5)

    print(f"\n{sim.summary()}")

    # Save if requested
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        sim.collector.save_json(out / "metrics.json")
        sim.collector.save_csv(out / "scalars.csv")
        print(f"\nResults saved to {out}/")

    # Visualization
    z = args.z_slice
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"VDPA Full Simulation — {args.policy}", fontsize=13)

    # Top-left: final concentration
    ax = axes[0, 0]
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

    ax.set_title(f"Final Concentration (z={z})", fontsize=10)
    ax.set_aspect("equal")

    # Top-right: cumulative contamination
    ax = axes[0, 1]
    steps, contam = sim.collector.scalar_series("cumulative_contamination")
    ax.plot(steps, contam, "r-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Contamination")
    ax.set_title("Time-Integrated Contamination", fontsize=10)

    # Bottom-left: total mass + contaminated volume
    ax = axes[1, 0]
    steps, mass = sim.collector.scalar_series("total_mass")
    ax.plot(steps, mass, "b-", label="Total Mass")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Mass", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    ax2 = ax.twinx()
    steps, vol = sim.collector.scalar_series("contaminated_volume")
    ax2.plot(steps, vol, "r--", label="Contaminated Volume")
    ax2.set_ylabel("Contaminated Cells", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.set_title("Mass & Contaminated Volume", fontsize=10)

    # Bottom-right: RMSE + prediction coverage
    ax = axes[1, 1]
    steps_r, rmse = sim.collector.scalar_series("concentration_rmse")
    if rmse:
        ax.plot(steps_r, rmse, "b-", label="Kalman RMSE")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")

    ax2 = ax.twinx()
    steps_p, cov = sim.collector.scalar_series("prediction_coverage")
    ax2.plot(steps_p, cov, "g--", label="Prediction Coverage")
    ax2.set_ylabel("Coverage", color="green")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="green")
    ax.set_title("Kalman RMSE & Gossip Coverage", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
