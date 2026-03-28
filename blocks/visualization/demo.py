"""
Standalone demo for Block 2 — 3D HVAC Visualization.

Usage:
    python -m blocks.visualization.demo --config configs/environments/hvac_office.json
    python -m blocks.visualization.demo --config configs/environments/hvac_office.json --gif output.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.visualization.renderer import Renderer
from blocks.simulation.simulation import Simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aether-Edge Block 2 — 3D Visualization Demo")
    p.add_argument("--config", type=str, required=True,
                    help="Path to HVAC environment JSON config")
    p.add_argument("--pre-steps", type=int, default=0,
                    help="Simulation steps to run before snapshot (default: 0)")
    p.add_argument("--frames", type=int, default=100, help="Number of frames for GIF")
    p.add_argument("--gif", type=str, default=None, help="Output path for animated GIF")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load the full Aether-Edge simulation brain rather than just the dumb physics world
    sim = Simulation(args.config, actuator_policy="edge")
    world = sim.world
    print(sim.env.summary())

    # Tighten the color bounds (19C to 24C) so subtle 2°C heat shifts aggressively change colors
    renderer = Renderer(sim.env, clim=(19.0, 24.0))

    if args.pre_steps > 0:
        print(f"Running {args.pre_steps} steps before snapshot...")
        # Step the full simulation (physics + sensors + edge computing logic)
        for _ in range(args.pre_steps):
            sim.step()

    if args.gif:
        print(f"Generating animation {args.gif} with {args.frames} frames...")
        # Manually drive the animation plotter
        pl = pv.Plotter(off_screen=True, window_size=renderer.window_size)
        pl.set_background(renderer.background)
        renderer._add_actors(pl, world)
        
        pl.open_gif(args.gif)
        
        for i in range(args.frames):
            sim.step()  # Let the smart Edge sensors drive the HVAC cooling!
            
            # Update the volumetric block dynamically without rebuilding
            renderer.vol.cell_data["temperature"] = world.T.ravel(order="F")
            
            title = f"HVAC Edge System | Step {world.step_count} | Overshoot: {world.max_overshoot():.2f}C"
            pl.add_text(title, name="title", font_size=12, color="white", position="upper_edge")
            
            pl.write_frame()
            if i % 10 == 0:
                print(f"Rendered frame {i}/{args.frames}")
                
        pl.close()
        print(f"Animation saved to {args.gif}")
    else:
        title = f"HVAC Thermal | Step {world.step_count} | Max overshoot: {world.max_overshoot():.2f}C"
        renderer.show(world, title=title)


if __name__ == "__main__":
    main()
