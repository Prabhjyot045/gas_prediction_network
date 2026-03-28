"""
Standalone demo for Block 3 — Sensor Mesh Topology.

Usage:
    python -m blocks.network.demo --config configs/environments/default_maze.json
    python -m blocks.network.demo --config configs/environments/default_maze.json --comm-radius 4.0
    python -m blocks.network.demo --config configs/environments/default_maze.json --z-slice 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.world.environment import Environment
from blocks.sensor.sensor_network import SensorNetwork


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 3 — Sensor Mesh Topology Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to environment JSON config")
    p.add_argument("--comm-radius", type=float, default=None,
                    help="Override communication radius")
    p.add_argument("--z-slice", type=int, default=2,
                    help="Z-level to visualize (default: 2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    net = SensorNetwork(env, comm_radius=args.comm_radius)

    # Print metrics
    print(env.summary())
    print()
    m = net.metrics()
    print("Sensor Network Metrics:")
    for key, value in m.items():
        if key != "degree_distribution":
            print(f"  {key}: {value}")
    print(f"  degree_distribution: {m['degree_distribution']}")

    z = args.z_slice
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("VDPA Block 3 — Sensor Mesh Topology", fontsize=13)

    # ── Left: Topology overlay on floor plan ──────────────────────────────
    ax = axes[0]

    # Draw walls
    wall_slice = env.walls[:, :, z]
    ax.imshow(wall_slice.T, cmap="Greys", origin="lower",
              alpha=0.4, aspect="equal", extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5))

    # Draw edges
    for u, v, data in net.graph.edges(data=True):
        p1 = net.positions[u]
        p2 = net.positions[v]
        if p1[2] == z or p2[2] == z:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "w-", linewidth=0.6, alpha=0.5)

    # Draw nodes
    node_pos = net.node_positions_array()
    z_mask = node_pos[:, 2] == z
    if np.any(z_mask):
        # Color by degree
        names_at_z = [n for n, p in net.positions.items() if p[2] == z]
        degrees = [net.graph.degree(n) for n in names_at_z]
        sc = ax.scatter(
            node_pos[z_mask, 0], node_pos[z_mask, 1],
            c=degrees, cmap="YlOrRd", s=60, edgecolors="white",
            linewidths=0.5, zorder=5, vmin=0,
        )
        plt.colorbar(sc, ax=ax, label="Node Degree")

    # Mark sources
    for src in env.sources:
        if src.position[2] == z:
            ax.plot(src.position[0], src.position[1], "c^",
                    markersize=12, label=src.name, zorder=6)

    ax.set_xlim(-0.5, env.nx - 0.5)
    ax.set_ylim(-0.5, env.ny - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Sensor Network at z={z}\n"
                 f"{net.n_nodes} nodes, {net.n_edges} edges, "
                 f"radius={net.comm_radius}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    # ── Right: Degree distribution histogram ──────────────────────────────
    ax2 = axes[1]
    dd = net.degree_distribution()
    if dd:
        ax2.bar(dd.keys(), dd.values(), color="coral", edgecolor="black")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Degree Distribution\n"
                       f"avg={net.average_degree():.1f}, "
                       f"clustering={net.clustering_coefficient():.3f}",
                       fontsize=10)
        ax2.set_xticks(sorted(dd.keys()))

    # Add metrics text box
    info = (
        f"Connected: {m['is_connected']}\n"
        f"Components: {m['connected_components']}\n"
        f"Diameter: {m['diameter']}\n"
        f"Avg Path: {m['average_path_length']}\n"
        f"Coverage: {m['coverage']:.1%}"
    )
    ax2.text(0.95, 0.95, info, transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
