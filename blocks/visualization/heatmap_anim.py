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
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

from blocks.simulation.simulation import Simulation

if TYPE_CHECKING:
    pass


def render_heatmap_gif(
    sim: Simulation,
    output_path: Path | str | None,
    *,
    n_frames: int = 200,
    steps_per_frame: int = 3,
    z_slice: int = 1,
    fps: int = 15,
    policy_label: str = "",
) -> None:
    """Render an animated heatmap of a running Simulation.

    Advances the simulation by (n_frames * steps_per_frame) total steps.
    If output_path is None, displays the animation interactively.
    If output_path is provided, saves a GIF and closes the figure.

    Args:
        sim: A fully constructed Simulation (not yet stepped, or pre-warmed).
        output_path: Path to save the GIF, or None to display interactively.
        n_frames: Number of animation frames.
        steps_per_frame: Simulation steps advanced per visual frame.
        z_slice: Z-layer index to visualize.
        fps: Frames per second for the saved GIF.
        policy_label: Display label shown in the title (e.g. "Edge" or "Centralized").
    """
    env = sim.env
    world = sim.world
    z = z_slice

    label = policy_label or sim.interface.policy.title()

    # ── Figure setup ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")

    wall_slice = env.walls[:, :, z].T
    T_slice = world.T[:, :, z].T
    masked_T = np.ma.array(T_slice, mask=wall_slice)

    im = ax.imshow(
        masked_T, cmap="RdYlBu_r", origin="lower",
        extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5),
        vmin=18.0, vmax=24.0, interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, label="Temperature (\u00b0C)", shrink=0.8, pad=0.02)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # Wall overlay
    wall_overlay = np.ma.array(np.ones_like(wall_slice, dtype=float), mask=~wall_slice)
    ax.imshow(
        wall_overlay, cmap="Greys", origin="lower",
        extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5),
        alpha=0.9, vmin=0, vmax=1,
    )

    # Room outlines and labels
    for room_name, room in env.rooms.items():
        sx, sy, _ = room.slices
        rect = mpatches.FancyBboxPatch(
            (sx.start - 0.5, sy.start - 0.5),
            sx.stop - sx.start, sy.stop - sy.start,
            boxstyle="round,pad=0.3", linewidth=2,
            edgecolor="cyan", facecolor="none", linestyle="--",
        )
        ax.add_patch(rect)
        cx = (sx.start + sx.stop) / 2
        cy = (sy.start + sy.stop) / 2
        ax.text(
            cx, cy, room_name.replace("_", " ").title(),
            color="white", fontsize=8, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    # Heat source markers — use stable hash for reproducible offsets
    for src in env.heat_sources:
        room = env.rooms[src.zone]
        sx, sy, _ = room.slices
        cx = (sx.start + sx.stop) / 2
        cy = (sy.start + sy.stop) / 2
        seed_x = int(hashlib.md5(src.name.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        seed_y = int(hashlib.md5((src.name + "_y").encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        ox = np.random.default_rng(seed_x).uniform(-2, 2)
        oy = np.random.default_rng(seed_y).uniform(-2, 2)
        ax.plot(cx + ox, cy + oy, marker="^", color="orange", markersize=10,
                markeredgecolor="red", markeredgewidth=1.5, zorder=5)

    # VAV damper markers and live opening labels
    damper_texts: dict[str, plt.Text] = {}
    for dname, damper in env.dampers.items():
        dx, dy = damper.position[0], damper.position[1]
        ax.plot(dx, dy, marker="*", color="deepskyblue", markersize=14,
                markeredgecolor="white", markeredgewidth=1, zorder=5)
        damper_texts[dname] = ax.text(
            dx, dy - 1.5, f"{damper.opening * 100:.0f}%",
            color="deepskyblue", fontsize=7, ha="center", va="top", fontweight="bold",
        )

    # Per-room temperature readouts
    room_temp_texts: dict[str, plt.Text] = {}
    for room_name, room in env.rooms.items():
        sx, sy, _ = room.slices
        room_temp_texts[room_name] = ax.text(
            sx.stop - 1.5, sy.stop - 1.5, "", color="yellow", fontsize=8,
            fontweight="bold", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.7),
        )

    title_text = ax.set_title("", color="white", fontsize=13, fontweight="bold", pad=10)

    legend_elements = [
        mpatches.Patch(facecolor="orange", edgecolor="red", label="Heat Source"),
        plt.Line2D([0], [0], marker="*", color="w", label="VAV Damper",
                   markerfacecolor="deepskyblue", markersize=12, linestyle="None"),
        mpatches.Patch(edgecolor="cyan", facecolor="none", linestyle="--", label="Room Boundary"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              facecolor="black", edgecolor="gray", labelcolor="white")

    ax.set_xlabel("X (meters)", color="white")
    ax.set_ylabel("Y (meters)", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#0e0e1a")
    ax.set_aspect("equal")

    # ── Animation loop ───────────────────────────────────────────────
    def update(_frame):
        for _ in range(steps_per_frame):
            sim.step()

        T_now = world.T[:, :, z].T
        im.set_data(np.ma.array(T_now, mask=wall_slice))

        for dname, damper in env.dampers.items():
            damper_texts[dname].set_text(f"{damper.opening * 100:.0f}%")

        for rn in env.rooms:
            room_temp_texts[rn].set_text(f"{world.zone_mean_temperature(rn):.1f}\u00b0C")

        title_text.set_text(
            f"{label}  |  Step {world.step_count}  |  "
            f"t={world.time:.0f}s  |  Overshoot {world.max_overshoot():.2f}\u00b0C"
        )
        return [im, title_text] + list(damper_texts.values()) + list(room_temp_texts.values())

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if output_path is not None:
        print(f"  Saving GIF → {output_path} ({n_frames} frames @ {fps} fps)...")
        anim.save(str(output_path), writer=animation.PillowWriter(fps=fps), dpi=100)
        plt.close(fig)
    else:
        plt.show()


# ── CLI entry point ──────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge — 2D Heatmap Animation")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--policy", choices=["edge", "centralized"], default="edge")
    p.add_argument("--frames", type=int, default=200, help="Number of animation frames")
    p.add_argument("--steps-per-frame", type=int, default=3, help="Sim steps per frame")
    p.add_argument("--pre-steps", type=int, default=0, help="Warm-up steps before recording")
    p.add_argument("--z-slice", type=int, default=1, help="Z-layer to visualize")
    p.add_argument("--gif", type=str, default=None, help="Save to GIF (display if omitted)")
    p.add_argument("--fps", type=int, default=15, help="Frames per second")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sim = Simulation(args.config, actuator_policy=args.policy, seed=args.seed)
    print(sim.env.summary())

    if args.pre_steps > 0:
        print(f"Warming up: {args.pre_steps} steps...")
        for _ in range(args.pre_steps):
            sim.step()

    print(f"Rendering {args.frames} frames ({args.steps_per_frame} sim steps each)...")
    render_heatmap_gif(
        sim,
        output_path=Path(args.gif) if args.gif else None,
        n_frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        z_slice=args.z_slice,
        fps=args.fps,
    )
    if args.gif:
        print(f"Saved to {args.gif}")


if __name__ == "__main__":
    main()
