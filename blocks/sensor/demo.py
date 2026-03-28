"""
Standalone demo for Block 4 — Sensor Nodes with Kalman Filter + Gossip.

Usage:
    python -m blocks.sensor.demo --config configs/environments/default_maze.json
    python -m blocks.sensor.demo --config configs/environments/default_maze.json --steps 300
    python -m blocks.sensor.demo --config configs/environments/default_maze.json --gossip-rounds 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.sensor.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 4 — Sensor + Gossip Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to environment JSON config")
    p.add_argument("--steps", type=int, default=200,
                    help="Number of simulation steps (default: 200)")
    p.add_argument("--gossip-rounds", type=int, default=1,
                    help="Gossip propagation rounds per step (default: 1)")
    p.add_argument("--z-slice", type=int, default=2,
                    help="Z-level for visualization (default: 2)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Setup
    env = Environment(args.config)
    world = World(env)
    network = SensorNetwork(env)
    field = SensorField(
        env, network,
        gossip_rounds=args.gossip_rounds,
        seed=args.seed,
    )

    print(env.summary())
    print(f"\nSensor field: {len(field.nodes)} nodes, "
          f"gossip_rounds={args.gossip_rounds}")

    # Run simulation, collect time series
    steps_log = []
    rmse_log = []
    detecting_log = []
    prediction_coverage_log = []
    messages_log = []

    record_every = max(1, args.steps // 50)

    for step in range(args.steps):
        world.step()
        field.step(world)

        if step % record_every == 0 or step == args.steps - 1:
            m = field.metrics(world)
            steps_log.append(step)
            rmse_log.append(m.get("concentration_rmse", 0.0))
            detecting_log.append(m["n_detecting"])
            prediction_coverage_log.append(m["prediction_coverage"])
            messages_log.append(m["total_messages_sent"])

    # Final metrics
    final_m = field.metrics(world)
    print(f"\n--- Final metrics (step {args.steps}) ---")
    for k, v in final_m.items():
        print(f"  {k}: {v}")

    # ── Visualization ──────────────────────────────────────────────────
    z = args.z_slice
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("VDPA Block 4 — Kalman Filter + Gossip Protocol", fontsize=13)

    # Top-left: Concentration field + sensor positions colored by filtered reading
    ax = axes[0, 0]
    phi_slice = world.phi[:, :, z].T
    wall_slice = env.walls[:, :, z].T
    masked = np.ma.array(phi_slice, mask=wall_slice)
    im = ax.imshow(masked, cmap="hot", origin="lower",
                   extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5))
    plt.colorbar(im, ax=ax, label="Concentration")

    # Overlay sensor positions
    positions = network.node_positions_array()
    z_mask = positions[:, 2] == z
    if np.any(z_mask):
        names_at_z = [n for n, p in network.positions.items() if p[2] == z]
        readings = [field.nodes[n].filtered_concentration for n in names_at_z]
        ax.scatter(positions[z_mask, 0], positions[z_mask, 1],
                   c=readings, cmap="hot", s=40, edgecolors="cyan",
                   linewidths=0.8, zorder=5,
                   vmin=im.get_clim()[0], vmax=im.get_clim()[1])

    ax.set_title(f"Concentration + Kalman Estimates (z={z})", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    # Top-right: Predicted arrival times at each sensor
    ax = axes[0, 1]
    wall_slice = env.walls[:, :, z].T
    ax.imshow(wall_slice.T if wall_slice.ndim == 2 else wall_slice,
              cmap="Greys", origin="lower", alpha=0.3,
              extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5))

    if np.any(z_mask):
        names_at_z = [n for n, p in network.positions.items() if p[2] == z]
        arrivals = []
        for n in names_at_z:
            a = field.nodes[n].earliest_predicted_arrival
            arrivals.append(a if a < float("inf") else -1)

        arrivals_arr = np.array(arrivals)
        has_pred = arrivals_arr > 0
        no_pred = ~has_pred

        if np.any(no_pred):
            ax.scatter(positions[z_mask][no_pred, 0], positions[z_mask][no_pred, 1],
                       c="gray", s=30, marker="o", alpha=0.5, label="No prediction")

        if np.any(has_pred):
            sc = ax.scatter(
                positions[z_mask][has_pred, 0], positions[z_mask][has_pred, 1],
                c=arrivals_arr[has_pred], cmap="RdYlGn", s=50,
                edgecolors="black", linewidths=0.5, zorder=5,
            )
            plt.colorbar(sc, ax=ax, label="Predicted Arrival Time (s)")

    ax.set_title("Gossip-Predicted Arrival Times", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom-left: RMSE + detecting nodes over time
    ax = axes[1, 0]
    ax.plot(steps_log, rmse_log, "b-", label="Kalman RMSE")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(steps_log, detecting_log, "r--", label="Detecting Nodes")
    ax2.set_ylabel("Detecting Nodes", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Kalman Filter Performance", fontsize=10)

    # Bottom-right: Prediction coverage + message count
    ax = axes[1, 1]
    ax.plot(steps_log, prediction_coverage_log, "g-", label="Prediction Coverage")
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage (fraction)", color="green")
    ax.tick_params(axis="y", labelcolor="green")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(steps_log, messages_log, "m--", label="Total Messages")
    ax2.set_ylabel("Messages Sent", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Gossip Propagation", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
