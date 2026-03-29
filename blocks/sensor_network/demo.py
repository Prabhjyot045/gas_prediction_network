"""
Plot sensor placement and communication graph.

Usage:
python -m blocks.sensor_network.demo --config configs/environments/university_floor.json
"""

from __future__ import annotations

import argparse

from blocks.world.environment import Environment
from blocks.sensor_network.sensor_network import SensorNetwork
from blocks.sensor_network.plot_sensor_network import (
    plot_network_slice,
    plot_network_projection,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    # visualization options
    parser.add_argument("--slice-z", type=int, default=None)
    parser.add_argument("--edges", action="store_true")

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Load environment
    env = Environment(args.config)

    print(f"[INFO] Loaded environment: {args.config}")
    print(f"[INFO] Grid: ({env.nx}, {env.ny}, {env.nz})")

    # Build sensor network
    net = SensorNetwork(env)

    print("[INFO] Sensor network metrics:")
    for k, v in net.metrics().items():
        print(f"  {k}: {v}")

    # Plot
    if args.slice_z is not None:
        plot_network_slice(
            net,
            z=args.slice_z,
            show_edges=args.edges,
        )
    else:
        plot_network_projection(
            net,
            show_edges=args.edges,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()