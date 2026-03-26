"""
Standalone demo for Block 2 — 3D HVAC Visualization.

Usage:
    python -m blocks.visualization.demo --config configs/environments/hvac_office.json
    python -m blocks.visualization.demo --config configs/environments/hvac_office.json --pre-steps 300
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
    p = argparse.ArgumentParser(description="Aether-Edge Block 2 — 3D Visualization Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to HVAC environment JSON config")
    p.add_argument("--pre-steps", type=int, default=200,
                    help="Simulation steps to run before snapshot (default: 200)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    world = World(env)
    print(env.summary())

    renderer = Renderer(env)

    print(f"Running {args.pre_steps} steps before snapshot...")
    world.run(args.pre_steps)
    title = f"HVAC Thermal | Step {world.step_count} | Max overshoot: {world.max_overshoot():.2f}C"
    renderer.show(world, title=title)


if __name__ == "__main__":
    main()
