"""
Sensor placement strategies.

Generates named sensor positions from the environment config.
Three strategies: grid (uniform spacing), random (N samples), manual (explicit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from blocks.world.environment import Environment


def place_sensors(env: Environment) -> list[tuple[str, tuple[int, int, int]]]:
    """Dispatch to the correct placement strategy from env.sensor_config.

    Returns list of (name, (x, y, z)) tuples. All positions are validated
    to be non-wall cells.
    """
    cfg = env.sensor_config
    if not cfg:
        return []

    strategy = cfg.get("placement", "grid")
    if strategy == "grid":
        return grid_placement(env, cfg)
    elif strategy == "random":
        return random_placement(env, cfg)
    elif strategy == "manual":
        return manual_placement(env, cfg)
    else:
        raise ValueError(f"Unknown sensor placement strategy: '{strategy}'")


def grid_placement(
    env: Environment,
    cfg: dict,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Place sensors on a uniform grid at the given spacing and z-levels."""
    spacing = cfg.get("spacing", 3)
    z_levels = cfg.get("z_levels", [env.nz // 2])

    sensors = []
    for z in z_levels:
        if z < 0 or z >= env.nz:
            continue
        for x in range(spacing, env.nx, spacing):
            for y in range(spacing, env.ny, spacing):
                if not env.walls[x, y, z]:
                    name = f"s_{x:02d}_{y:02d}_{z:02d}"
                    sensors.append((name, (x, y, z)))
    return sensors


def random_placement(
    env: Environment,
    cfg: dict,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Place N sensors randomly in non-wall cells."""
    count = cfg.get("count", 20)
    seed = cfg.get("seed", 42)
    z_levels = cfg.get("z_levels", None)

    rng = np.random.default_rng(seed)

    # Collect all candidate cells
    candidates = []
    for x in range(env.nx):
        for y in range(env.ny):
            for z in range(env.nz):
                if not env.walls[x, y, z]:
                    if z_levels is None or z in z_levels:
                        candidates.append((x, y, z))

    if len(candidates) == 0:
        return []

    count = min(count, len(candidates))
    indices = rng.choice(len(candidates), size=count, replace=False)

    sensors = []
    for i, idx in enumerate(sorted(indices)):
        pos = candidates[idx]
        name = f"s_{pos[0]:02d}_{pos[1]:02d}_{pos[2]:02d}"
        sensors.append((name, pos))
    return sensors


def manual_placement(
    env: Environment,
    cfg: dict,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Place sensors at explicitly defined positions."""
    sensors = []
    for node_cfg in cfg.get("nodes", []):
        pos = node_cfg["position"]
        position = (pos["x"], pos["y"], pos["z"])
        name = node_cfg.get("name", f"s_{pos['x']:02d}_{pos['y']:02d}_{pos['z']:02d}")

        if env.walls[position]:
            raise ValueError(
                f"Sensor '{name}' at {position} is inside a wall."
            )
        sensors.append((name, position))
    return sensors
