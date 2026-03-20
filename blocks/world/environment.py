"""
Environment loader — parses a JSON config and builds 3D wall/door/source arrays.

The JSON defines rooms as rectangular open regions carved out of a solid grid.
Everything outside a room is wall. Doors are thin slabs that connect rooms
through walls and can be toggled at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stability import compute_stable_dt, validate_dt


@dataclass
class Source:
    """A gas emission point in the grid."""
    name: str
    position: tuple[int, int, int]
    rate: float
    start_time: float = 0.0
    end_time: float | None = None

    def is_active(self, t: float) -> bool:
        if t < self.start_time:
            return False
        if self.end_time is not None and t >= self.end_time:
            return False
        return True


@dataclass
class Door:
    """An actuatable opening between rooms."""
    name: str
    slices: tuple[slice, slice, slice]
    state: str = "open"  # "open" or "closed"


class Environment:
    """Loads a JSON environment config and builds the 3D simulation arrays."""

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config = json.load(f)

        self._parse_grid()
        self._parse_physics()
        self._build_walls()
        self._parse_sources()
        self._parse_noise()
        self._parse_sensor_config()

    # ── Grid ──────────────────────────────────────────────────────────────

    def _parse_grid(self) -> None:
        grid = self.config["grid"]
        self.nx: int = grid["nx"]
        self.ny: int = grid["ny"]
        self.nz: int = grid["nz"]
        self.dx: float = grid.get("dx", 1.0)
        self.grid_shape: tuple[int, int, int] = (self.nx, self.ny, self.nz)

    # ── Physics ───────────────────────────────────────────────────────────

    def _parse_physics(self) -> None:
        phys = self.config["physics"]
        self.diffusion_coefficient: float = phys["diffusion_coefficient"]

        user_dt = phys.get("dt")
        safety = phys.get("safety_factor", 0.4)

        if user_dt is None or user_dt == 0:
            self.dt: float = compute_stable_dt(self.dx, self.diffusion_coefficient, safety)
        else:
            validate_dt(user_dt, self.dx, self.diffusion_coefficient)
            self.dt = user_dt

    # ── Walls and Doors ───────────────────────────────────────────────────

    def _bounds_to_slices(self, bounds: dict) -> tuple[slice, slice, slice]:
        return (
            slice(bounds["x_min"], bounds["x_max"]),
            slice(bounds["y_min"], bounds["y_max"]),
            slice(bounds["z_min"], bounds["z_max"]),
        )

    def _build_walls(self) -> None:
        """Start with everything as wall, carve out rooms, then handle doors."""
        self.walls = np.ones(self.grid_shape, dtype=bool)

        # Carve rooms
        for room in self.config.get("rooms", []):
            s = self._bounds_to_slices(room["bounds"])
            self.walls[s] = False

        # Parse doors and apply initial state
        self.doors: dict[str, Door] = {}
        for door_cfg in self.config.get("doors", []):
            s = self._bounds_to_slices(door_cfg["bounds"])
            state = door_cfg.get("state", "open")
            door = Door(name=door_cfg["name"], slices=s, state=state)
            self.doors[door.name] = door

            if state == "open":
                self.walls[s] = False
            else:
                self.walls[s] = True

    # ── Sources ───────────────────────────────────────────────────────────

    def _parse_sources(self) -> None:
        self.sources: list[Source] = []
        for src_cfg in self.config.get("sources", []):
            pos = src_cfg["position"]
            position = (pos["x"], pos["y"], pos["z"])

            # Validate source is not inside a wall
            if self.walls[position]:
                raise ValueError(
                    f"Source '{src_cfg['name']}' at {position} is inside a wall. "
                    f"Place it inside a room."
                )

            self.sources.append(Source(
                name=src_cfg["name"],
                position=position,
                rate=src_cfg["rate"],
                start_time=src_cfg.get("start_time", 0.0),
                end_time=src_cfg.get("end_time"),
            ))

    # ── Noise ─────────────────────────────────────────────────────────────

    def _parse_noise(self) -> None:
        noise = self.config.get("noise", {})
        self.sensor_sigma: float = noise.get("sensor_sigma", 0.0)
        self.source_rate_sigma: float = noise.get("source_rate_sigma", 0.0)

    # ── Sensor config ───────────────────────────────────────────────────

    def _parse_sensor_config(self) -> None:
        """Parse the optional 'sensors' section from the JSON config.

        Stores the raw config for Block 3 to consume. Does NOT resolve
        positions here — that is Block 3's responsibility (placement.py).
        """
        self.sensor_config: dict = self.config.get("sensors", {})

    # ── Door control (runtime) ────────────────────────────────────────────

    def open_door(self, name: str) -> None:
        """Open a door — carve it out of the wall mask."""
        door = self.doors[name]
        door.state = "open"
        self.walls[door.slices] = False

    def close_door(self, name: str) -> None:
        """Close a door — fill it back in as wall."""
        door = self.doors[name]
        door.state = "closed"
        self.walls[door.slices] = True

    def get_door_state(self, name: str) -> str:
        return self.doors[name].state

    # ── Info ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        total = np.prod(self.grid_shape)
        open_cells = int(np.sum(~self.walls))
        return (
            f"Environment: {self.nx}x{self.ny}x{self.nz} grid, "
            f"dx={self.dx}m, D={self.diffusion_coefficient} m²/s, "
            f"dt={self.dt:.6f}s\n"
            f"  Open cells: {open_cells}/{total} "
            f"({100*open_cells/total:.1f}%)\n"
            f"  Rooms: {len(self.config.get('rooms', []))}\n"
            f"  Doors: {len(self.doors)} "
            f"({sum(1 for d in self.doors.values() if d.state == 'open')} open)\n"
            f"  Sources: {len(self.sources)}"
        )
