"""
Standalone demo for Block 5 — Actuator Controller.

Usage:
    python -m blocks.actuator.demo --config configs/environments/default_maze.json
    python -m blocks.actuator.demo --config configs/environments/default_maze.json --policy reactive
    python -m blocks.actuator.demo --config configs/environments/default_maze.json --horizon 3.0
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
from blocks.network.sensor_network import SensorNetwork
from blocks.sensor.sensor_field import SensorField
from blocks.actuator.controller import ActuatorController


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VDPA Block 5 — Actuator Demo")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--policy", choices=["predictive", "reactive"], default="predictive")
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--z-slice", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    env = Environment(args.config)
    world = World(env)
    network = SensorNetwork(env)
    field = SensorField(env, network, gossip_rounds=2, seed=args.seed)
    actuator = ActuatorController(
        env, field,
        policy=args.policy,
        horizon=args.horizon,
        concentration_threshold=args.threshold,
    )

    print(env.summary())
    print(f"\nActuator policy: {args.policy}")
    print(f"Door-sensor mapping:")
    for door, sensors in actuator.door_sensors.items():
        print(f"  {door}: {len(sensors)} nearby sensors")

    # Run simulation
    mass_log = []
    contam_log = []
    steps_log = []
    door_events = []

    for step in range(args.steps):
        world.step()
        field.step(world)
        closed = actuator.evaluate(world)

        for d in closed:
            door_events.append((step, d))
            print(f"  Step {step}: CLOSED {d} (t={world.time:.4f}s)")

        if step % 5 == 0:
            steps_log.append(step)
            mass_log.append(world.total_mass())
            contam_log.append(world.contaminated_volume())

    # Print summary
    m = actuator.metrics()
    print(f"\n--- Actuator Summary ---")
    print(f"  Doors closed: {m['doors_closed']}/{len(env.doors)}")
    print(f"  First detection: {m['first_detection_time']}")
    print(f"  First actuation: {m['first_actuation_time']}")
    print(f"  Response time: {m['response_time']}")

    # Visualization
    z = args.z_slice
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Block 5 — Actuator ({args.policy})", fontsize=13)

    # Left: final concentration with door states
    ax = axes[0]
    phi_slice = world.phi[:, :, z].T
    wall_slice = env.walls[:, :, z].T
    masked = np.ma.array(phi_slice, mask=wall_slice)
    im = ax.imshow(masked, cmap="hot", origin="lower",
                   extent=(-0.5, env.nx - 0.5, -0.5, env.ny - 0.5))
    plt.colorbar(im, ax=ax, label="Concentration")

    # Mark doors
    for door_name, door in env.doors.items():
        s = door.slices
        cx = (s[0].start + s[0].stop) / 2
        cy = (s[1].start + s[1].stop) / 2
        color = "red" if door.state == "closed" else "lime"
        ax.plot(cx, cy, "s", color=color, markersize=12, markeredgecolor="white",
                label=f"{door_name} ({door.state})")

    ax.set_title(f"Final State (z={z})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_aspect("equal")

    # Middle: total mass over time
    ax = axes[1]
    ax.plot(steps_log, mass_log, "b-")
    for step, name in door_events:
        ax.axvline(step, color="red", linestyle="--", alpha=0.5, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Mass")
    ax.set_title("Total Mass Over Time", fontsize=10)

    # Right: contaminated volume
    ax = axes[2]
    ax.plot(steps_log, contam_log, "r-")
    for step, name in door_events:
        ax.axvline(step, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Contaminated Cells")
    ax.set_title("Contaminated Volume Over Time", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
