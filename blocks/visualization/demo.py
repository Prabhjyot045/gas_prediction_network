"""
Standalone demo for Block 2 — 3D Visualization.

Usage:
    python -m blocks.visualization.demo --config configs/environments/default_maze.json
    python -m blocks.visualization.demo --config configs/environments/default_maze.json --mode animate --frames 200
    python -m blocks.visualization.demo --config configs/environments/default_maze.json --mode snapshot --pre-steps 300
    python -m blocks.visualization.demo --config configs/environments/default_maze.json --mode animate --gif output.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.visualization.renderer import Renderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 2 — 3D Visualization Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to environment JSON config")
    p.add_argument("--mode", choices=["snapshot", "animate"], default="animate",
                    help="Visualization mode (default: animate)")
    p.add_argument("--pre-steps", type=int, default=100,
                    help="Simulation steps to run before snapshot (default: 100)")
    p.add_argument("--frames", type=int, default=200,
                    help="Animation frames (default: 200)")
    p.add_argument("--steps-per-frame", type=int, default=5,
                    help="Simulation steps per animation frame (default: 5)")
    p.add_argument("--vmax", type=float, default=5.0,
                    help="Max colormap value (default: 5.0)")
    p.add_argument("--gif", type=str, default=None,
                    help="Save animation as GIF to this path")
    p.add_argument("--close-door", type=str, default=None,
                    help="Name of door to close (before sim, or at --close-at frame)")
    p.add_argument("--close-at", type=int, default=None,
                    help="Frame number at which to close the door (mid-animation)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    world = World(env)
    print(env.summary())

    # Pre-close door (before simulation starts)
    if args.close_door and args.close_at is None:
        world.close_door(args.close_door)
        print(f"Closed door: {args.close_door}")

    renderer = Renderer(
        env,
        clim=(0.0, args.vmax),
    )

    # Build mid-animation callback for door closing at a specific frame
    frame_callback = None
    if args.close_door and args.close_at is not None:
        door_name = args.close_door
        close_frame = args.close_at
        door_closed = False

        def frame_callback(w, frame):
            nonlocal door_closed
            if not door_closed and frame >= close_frame:
                w.close_door(door_name)
                door_closed = True
                print(f"  Frame {frame}: Closed '{door_name}'")

    if args.mode == "snapshot":
        print(f"Running {args.pre_steps} steps before snapshot...")
        world.run(args.pre_steps)
        title = f"VDPA Snapshot | Step {world.step_count} | Mass: {world.total_mass():.1f}"
        renderer.show(world, title=title)

    elif args.mode == "animate":
        print(f"Animating {args.frames} frames ({args.steps_per_frame} steps/frame)...")
        renderer.animate(
            world,
            n_frames=args.frames,
            steps_per_frame=args.steps_per_frame,
            gif_path=args.gif,
            frame_callback=frame_callback,
        )


if __name__ == "__main__":
    main()
