"""
2D Heatmap Animation — top-down thermal view of HVAC simulation.

Generates a matplotlib animation showing:
  - Room boundaries and labels
  - Heat source locations (fire markers)
  - VAV damper locations (snowflake markers) with live opening %
  - Live temperature heatmap that updates every frame
  - Per-room temperature readout

Usage:
    python -m blocks.visualization.heatmap_anim --config configs/environments/university_floor.json
    python -m blocks.visualization.heatmap_anim --config configs/environments/university_floor.json --gif hvac_heatmap.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.simulation.simulation import Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge — 2D Heatmap Animation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--frames", type=int, default=200, help="Number of simulation frames")
    p.add_argument("--steps-per-frame", type=int, default=3, help="Sim steps per animation frame")
    p.add_argument("--pre-steps", type=int, default=0, help="Warm-up steps before recording")
    p.add_argument("--z-slice", type=int, default=1, help="Which z-layer to visualize")
    p.add_argument("--gif", type=str, default=None, help="Save to GIF file")
    p.add_argument("--fps", type=int, default=15, help="Frames per second for GIF")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Boot the full Aether-Edge simulation
    sim = Simulation(args.config, actuator_policy="edge", seed=42)
    env = sim.env
    world = sim.world
    z = args.z_slice

    print(env.summary())

    # Warm up
    if args.pre_steps > 0:
        print(f"Warming up: {args.pre_steps} steps...")
        for _ in range(args.pre_steps):
            sim.step()

    # ── Figure setup ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # Initial temperature slice
    wall_slice = env.walls[:, :, z].T
    T_slice = world.T[:, :, z].T
    masked_T = np.ma.array(T_slice, mask=wall_slice)

    # Heatmap
    im = ax.imshow(
        masked_T, cmap="RdYlBu_r", origin="lower",
        extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5),
        vmin=18.0, vmax=24.0, interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, label="Temperature (°C)", shrink=0.8, pad=0.02)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # Draw walls as dark overlay
    wall_overlay = np.ma.array(np.ones_like(wall_slice, dtype=float), mask=~wall_slice)
    ax.imshow(
        wall_overlay, cmap="Greys", origin="lower",
        extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5),
        alpha=0.9, vmin=0, vmax=1,
    )

    # Draw room outlines and labels
    for room_name, room in env.rooms.items():
        sx, sy, sz = room.slices
        rect = mpatches.FancyBboxPatch(
            (sx.start - 0.5, sy.start - 0.5),
            sx.stop - sx.start, sy.stop - sy.start,
            boxstyle="round,pad=0.3", linewidth=2,
            edgecolor="cyan", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        cx = (sx.start + sx.stop) / 2
        cy = (sy.start + sy.stop) / 2
        label = room_name.replace("_", " ").title()
        ax.text(cx, cy, label, color="white", fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    # Mark heat sources (use room center for zone-level sources)
    heat_positions = []
    for src in env.heat_sources:
        room = env.rooms[src.zone]
        sx, sy, sz = room.slices
        cx = (sx.start + sx.stop) / 2
        cy = (sy.start + sy.stop) / 2
        # Offset slightly so multiple sources in the same room don't overlap
        offset_x = np.random.default_rng(abs(hash(src.name)) % (2**31)).uniform(-2, 2)
        offset_y = np.random.default_rng(abs(hash(src.name) + 1) % (2**31)).uniform(-2, 2)
        heat_positions.append((cx + offset_x, cy + offset_y, src.name))

    for hx, hy, hname in heat_positions:
        ax.plot(hx, hy, marker="^", color="orange", markersize=10,
                markeredgecolor="red", markeredgewidth=1.5, zorder=5)

    # Mark VAV dampers
    damper_texts = {}
    for dname, damper in env.dampers.items():
        dx, dy = damper.position[0], damper.position[1]
        ax.plot(dx, dy, marker="*", color="deepskyblue", markersize=14,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
        damper_texts[dname] = ax.text(
            dx, dy - 1.5, f"{damper.opening*100:.0f}%",
            color="deepskyblue", fontsize=7, ha="center", va="top", fontweight="bold",
        )

    # Room temperature readouts (top-right corner of each room)
    room_temp_texts = {}
    for room_name, room in env.rooms.items():
        sx, sy, sz = room.slices
        tx = sx.stop - 1.5
        ty = sy.stop - 1.5
        room_temp_texts[room_name] = ax.text(
            tx, ty, "", color="yellow", fontsize=8, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7),
        )

    # Title
    title_text = ax.set_title("", color="white", fontsize=13, fontweight="bold", pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="orange", edgecolor="red", label="Heat Source"),
        plt.Line2D([0], [0], marker="*", color="w", label="VAV Damper",
                   markerfacecolor="deepskyblue", markersize=12, linestyle="None"),
        mpatches.Patch(edgecolor="cyan", facecolor="none", linestyle="--", label="Room Boundary"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
                    facecolor="black", edgecolor="gray", labelcolor="white")

    ax.set_xlabel("X (meters)", color="white")
    ax.set_ylabel("Y (meters)", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#0e0e1a")
    ax.set_aspect("equal")

    # ── Animation loop ───────────────────────────────────────────────
    def update(frame_num):
        # Step the simulation multiple times per visual frame
        for _ in range(args.steps_per_frame):
            sim.step()

        # Update heatmap data
        T_slice = world.T[:, :, z].T
        masked_T = np.ma.array(T_slice, mask=wall_slice)
        im.set_data(masked_T)

        # Update damper opening labels
        for dname, damper in env.dampers.items():
            damper_texts[dname].set_text(f"{damper.opening*100:.0f}%")

        # Update room temperature readouts
        for room_name in env.rooms:
            temp = world.zone_mean_temperature(room_name)
            room_temp_texts[room_name].set_text(f"{temp:.1f}°C")

        # Update title
        title_text.set_text(
            f"Aether-Edge HVAC Simulation  |  "
            f"Step {world.step_count}  |  "
            f"Time {world.time:.0f}s  |  "
            f"Overshoot {world.max_overshoot():.2f}°C"
        )

        return [im, title_text] + list(damper_texts.values()) + list(room_temp_texts.values())

    print(f"Rendering {args.frames} frames ({args.steps_per_frame} sim steps each)...")
    anim = animation.FuncAnimation(
        fig, update, frames=args.frames, interval=1000 // args.fps, blit=False,
    )

    if args.gif:
        print(f"Saving to {args.gif}...")
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(args.gif, writer=writer, dpi=100)
        print(f"Animation saved to {args.gif}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
