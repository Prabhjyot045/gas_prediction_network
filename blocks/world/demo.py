"""
Standalone demo for Block 1 — World (3D FTCS Diffusion).

Usage:
    python -m blocks.world.demo --config configs/environments/default_maze.json
    python -m blocks.world.demo --config configs/environments/default_maze.json --steps 500 --z-slice 2
    python -m blocks.world.demo --config configs/environments/default_maze.json --close-door door_B_to_hallway --close-at 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.world.environment import Environment
from blocks.world.world import World


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 1 — Diffusion Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to environment JSON config")
    p.add_argument("--steps", type=int, default=300,
                    help="Number of simulation steps (default: 300)")
    p.add_argument("--z-slice", type=int, default=2,
                    help="Z-level to visualize (default: 2)")
    p.add_argument("--plot-every", type=int, default=5,
                    help="Render every N steps (default: 5)")
    p.add_argument("--close-door", type=str, default=None,
                    help="Name of door to close mid-simulation")
    p.add_argument("--close-at", type=int, default=100,
                    help="Step at which to close the door (default: 100)")
    p.add_argument("--vmax", type=float, default=5.0,
                    help="Max colormap value (default: 5.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    world = World(env)
    print(env.summary())
    print(f"Running {args.steps} steps, visualizing z={args.z_slice}")
    print(f"Alpha (D*dt/dx²) = {world._alpha:.6f}")

    z = args.z_slice
    if z < 0 or z >= env.nz:
        print(f"Error: z-slice {z} out of range [0, {env.nz})")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("VDPA Block 1 — 3D Diffusion Engine", fontsize=13)

    # Left: concentration heatmap
    ax_heat = axes[0]
    im = ax_heat.imshow(
        world.phi[:, :, z].T,
        cmap="inferno", origin="lower",
        vmin=0, vmax=args.vmax, interpolation="bilinear", aspect="equal",
    )
    plt.colorbar(im, ax=ax_heat, label="Concentration")

    # Wall overlay
    wall_slice = env.walls[:, :, z]
    wall_vis = np.ma.masked_where(~wall_slice, np.ones_like(wall_slice, dtype=float))
    ax_heat.imshow(
        wall_vis.T, cmap="Greys", origin="lower",
        alpha=0.7, vmin=0, vmax=1, aspect="equal",
    )

    # Mark sources
    for src in env.sources:
        if src.position[2] == z:
            ax_heat.plot(src.position[0], src.position[1], "g^", markersize=10, label=src.name)
    ax_heat.legend(loc="upper right", fontsize=8)
    ax_heat.set_xlabel("X")
    ax_heat.set_ylabel("Y")

    # Right: mass and contamination over time
    ax_plot = axes[1]
    mass_history = []
    contam_history = []
    step_history = []

    ax_plot.set_xlabel("Step")
    ax_plot.set_ylabel("Total Mass", color="tab:blue")
    ax_contam = ax_plot.twinx()
    ax_contam.set_ylabel("Contaminated Cells", color="tab:red")

    door_closed = False

    def update(frame: int) -> list:
        nonlocal door_closed

        for _ in range(args.plot_every):
            current_step = world.step_count

            # Close door mid-simulation if requested
            if (args.close_door and not door_closed
                    and current_step >= args.close_at):
                world.close_door(args.close_door)
                door_closed = True
                print(f"  Step {current_step}: Closed '{args.close_door}'")

            world.step()

        # Update heatmap
        im.set_data(world.phi[:, :, z].T)

        # Track metrics
        step_history.append(world.step_count)
        mass_history.append(world.total_mass())
        contam_history.append(world.contaminated_volume(threshold=0.1))

        # Update plots
        ax_plot.cla()
        ax_contam.cla()
        ax_plot.plot(step_history, mass_history, "tab:blue", linewidth=1.5)
        ax_plot.set_xlabel("Step")
        ax_plot.set_ylabel("Total Mass", color="tab:blue")
        ax_contam.plot(step_history, contam_history, "tab:red", linewidth=1.5)
        ax_contam.set_ylabel("Contaminated Cells", color="tab:red")

        status = f"Step {world.step_count}/{args.steps * args.plot_every}"
        ax_heat.set_title(f"Concentration at z={z}\n{status}", fontsize=10)
        ax_plot.set_title("Metrics", fontsize=10)

        # Update wall overlay if door changed
        if door_closed:
            wall_slice_now = env.walls[:, :, z]
            wall_vis_now = np.ma.masked_where(
                ~wall_slice_now, np.ones_like(wall_slice_now, dtype=float)
            )
            # Redraw walls (simple approach)
            ax_heat.images[-1].set_data(wall_vis_now.T)

        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=args.steps, interval=50, blit=False, repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
