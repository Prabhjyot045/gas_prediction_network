"""
Standalone demo for Block 1 — World (3D Thermal Diffusion + Cooling).

Usage:
    python -m blocks.world.demo --config configs/environments/hvac_office.json
    python -m blocks.world.demo --config configs/environments/hvac_office.json --steps 500 --z-slice 1
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from blocks.world.environment import Environment
from blocks.world.world import World


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge Block 1 — Thermal Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to HVAC environment JSON config")
    p.add_argument("--steps", type=int, default=300,
                    help="Number of simulation steps (default: 300)")
    p.add_argument("--z-slice", type=int, default=1,
                    help="Z-level to visualize (default: 1)")
    p.add_argument("--plot-every", type=int, default=5,
                    help="Render every N steps (default: 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    world = World(env)
    print(env.summary())
    print(f"Running {args.steps} steps, visualizing z={args.z_slice}")

    z = args.z_slice
    if z < 0 or z >= env.nz:
        print(f"Error: z-slice {z} out of range [0, {env.nz})")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Aether-Edge Block 1 — Thermal Physics Engine", fontsize=13)

    # Left: temperature heatmap
    ax_heat = axes[0]
    T_min, T_max = env.supply_temperature, env.ambient_temperature + 15
    im = ax_heat.imshow(
        world.T[:, :, z].T,
        cmap="coolwarm", origin="lower",
        vmin=T_min, vmax=T_max, interpolation="bilinear", aspect="equal",
    )
    plt.colorbar(im, ax=ax_heat, label="Temperature (C)")

    wall_slice = env.walls[:, :, z]
    wall_vis = np.ma.masked_where(~wall_slice, np.ones_like(wall_slice, dtype=float))
    ax_heat.imshow(
        wall_vis.T, cmap="Greys", origin="lower",
        alpha=0.7, vmin=0, vmax=1, aspect="equal",
    )

    # Mark damper positions
    for d in env.dampers.values():
        if d.position[2] == z:
            ax_heat.plot(d.position[0], d.position[1], "cv", markersize=10, label=d.name)
    ax_heat.legend(loc="upper right", fontsize=7)
    ax_heat.set_xlabel("X")
    ax_heat.set_ylabel("Y")

    # Middle: zone temperatures over time
    ax_zones = axes[1]
    zone_histories = {name: [] for name in env.rooms}
    step_history = []

    # Right: overshoot over time
    ax_over = axes[2]
    overshoot_history = []

    def update(frame: int) -> list:
        for _ in range(args.plot_every):
            world.step()

        im.set_data(world.T[:, :, z].T)

        step_history.append(world.step_count)
        for name in env.rooms:
            zone_histories[name].append(world.zone_mean_temperature(name))
        overshoot_history.append(world.max_overshoot())

        ax_zones.cla()
        for name, hist in zone_histories.items():
            sp = env.rooms[name].setpoint
            ax_zones.plot(step_history, hist, linewidth=1.5, label=f"{name} (sp={sp})")
        ax_zones.axhline(env.ambient_temperature, color="gray", linestyle="--", alpha=0.5, label="ambient")
        ax_zones.set_xlabel("Step")
        ax_zones.set_ylabel("Temperature (C)")
        ax_zones.set_title("Zone Temperatures")
        ax_zones.legend(fontsize=7)

        ax_over.cla()
        ax_over.fill_between(step_history, overshoot_history, alpha=0.3, color="red")
        ax_over.plot(step_history, overshoot_history, "r-", linewidth=1.5)
        ax_over.set_xlabel("Step")
        ax_over.set_ylabel("Max Overshoot (C)")
        ax_over.set_title("Comfort Violation")

        ax_heat.set_title(f"Temperature at z={z}\nStep {world.step_count}", fontsize=10)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=args.steps, interval=50, blit=False, repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
