"""
Environment loader — parses an HVAC JSON config and builds 3D grid arrays.

The JSON defines thermal zones (rooms), hallways, VAV dampers, heat sources,
and a cooling plant. Everything outside a room/hallway is wall. VAV dampers
are continuous actuators (A ∈ [0,1]) that inject cooling into zones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stability import compute_stable_dt, validate_dt


@dataclass
class HeatSource:
    """A heat gain in a thermal zone (occupancy, equipment, solar)."""
    name: str
    zone: str
    rate: float  # degrees/s added uniformly across the zone
    start_time: float = 0.0
    end_time: float | None = None

    def is_active(self, t: float) -> bool:
        if t < self.start_time:
            return False
        if self.end_time is not None and t >= self.end_time:
            return False
        return True


@dataclass
class VAVDamper:
    """A Variable Air Volume damper that controls cooling to a zone."""
    name: str
    zone: str
    position: tuple[int, int, int]
    max_flow: float = 1.0
    opening: float = 0.5  # A ∈ [0, 1]

    @property
    def current_flow(self) -> float:
        """Actual cooling flow = opening * max_flow."""
        return self.opening * self.max_flow


@dataclass
class Room:
    """A thermal zone with a comfort setpoint."""
    name: str
    slices: tuple[slice, slice, slice]
    setpoint: float = 22.0


class Environment:
    """Loads an HVAC JSON config and builds the 3D simulation arrays."""

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config = json.load(f)

        self._parse_grid()
        self._parse_physics()
        self._build_walls()
        self._parse_heat_sources()
        self._parse_vav_dampers()
        self._parse_cooling_plant()
        self._parse_noise()
        self._parse_sensor_config()
        self._parse_network_config()

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
        self.thermal_diffusivity: float = phys["thermal_diffusivity"]
        self.ambient_temperature: float = phys.get("ambient_temperature", 20.0)

        user_dt = phys.get("dt")
        safety = phys.get("safety_factor", 0.4)

        if user_dt is None or user_dt == 0:
            self.dt: float = compute_stable_dt(
                self.dx, self.thermal_diffusivity, safety_factor=safety
            )
        else:
            validate_dt(self.dx, self.thermal_diffusivity, user_dt)
            self.dt = user_dt

    # ── Walls, Rooms, Hallways ────────────────────────────────────────────

    def _bounds_to_slices(self, bounds: dict) -> tuple[slice, slice, slice]:
        return (
            slice(bounds["x_min"], bounds["x_max"]),
            slice(bounds["y_min"], bounds["y_max"]),
            slice(bounds["z_min"], bounds["z_max"]),
        )

    def _build_walls(self) -> None:
        """Start with everything as wall, carve out rooms and hallways."""
        self.walls = np.ones(self.grid_shape, dtype=bool)

        # Parse and carve rooms (thermal zones with setpoints)
        self.rooms: dict[str, Room] = {}
        for room_cfg in self.config.get("rooms", []):
            s = self._bounds_to_slices(room_cfg["bounds"])
            self.walls[s] = False
            self.rooms[room_cfg["name"]] = Room(
                name=room_cfg["name"],
                slices=s,
                setpoint=room_cfg.get("setpoint", 22.0),
            )

        # Carve hallways (no setpoint — unconditioned)
        for hall_cfg in self.config.get("hallways", []):
            s = self._bounds_to_slices(hall_cfg["bounds"])
            self.walls[s] = False

        # Build zone mask: maps each cell to its room name (or None)
        self.zone_mask: np.ndarray = np.full(self.grid_shape, "", dtype=object)
        for room_name, room in self.rooms.items():
            self.zone_mask[room.slices] = room_name

    # ── Heat Sources ──────────────────────────────────────────────────────

    def _parse_heat_sources(self) -> None:
        self.heat_sources: list[HeatSource] = []
        for src_cfg in self.config.get("heat_sources", []):
            zone = src_cfg["zone"]
            if zone not in self.rooms:
                raise ValueError(
                    f"Heat source '{src_cfg['name']}' references unknown zone '{zone}'."
                )
            self.heat_sources.append(HeatSource(
                name=src_cfg["name"],
                zone=zone,
                rate=src_cfg["rate"],
                start_time=src_cfg.get("schedule", {}).get("start", 0.0),
                end_time=src_cfg.get("schedule", {}).get("end"),
            ))

    # ── VAV Dampers ───────────────────────────────────────────────────────

    def _parse_vav_dampers(self) -> None:
        self.dampers: dict[str, VAVDamper] = {}
        for d_cfg in self.config.get("vav_dampers", []):
            pos = d_cfg["position"]
            position = (pos["x"], pos["y"], pos["z"])

            if self.walls[position]:
                raise ValueError(
                    f"VAV damper '{d_cfg['name']}' at {position} is inside a wall."
                )

            zone = d_cfg["zone"]
            if zone not in self.rooms:
                raise ValueError(
                    f"VAV damper '{d_cfg['name']}' references unknown zone '{zone}'."
                )

            self.dampers[d_cfg["name"]] = VAVDamper(
                name=d_cfg["name"],
                zone=zone,
                position=position,
                max_flow=d_cfg.get("max_flow", 1.0),
                opening=d_cfg.get("initial_opening", 0.5),
            )

    # ── Cooling Plant ─────────────────────────────────────────────────────

    def _parse_cooling_plant(self) -> None:
        plant = self.config.get("cooling_plant", {})
        self.Q_total: float = plant.get("Q_total", 5.0)
        self.supply_temperature: float = plant.get("supply_temperature", 12.0)

    # ── Noise ─────────────────────────────────────────────────────────────

    def _parse_noise(self) -> None:
        noise = self.config.get("noise", {})
        self.sensor_sigma: float = noise.get("sensor_sigma", 0.0)

    # ── Sensor config ─────────────────────────────────────────────────────

    def _parse_sensor_config(self) -> None:
        self.sensor_config: dict = self.config.get("sensors", {})

    # ── Network config ────────────────────────────────────────────────────

    def _parse_network_config(self) -> None:
        net = self.config.get("network", {})
        self.polling_interval: float = net.get("polling_interval", 5.0)
        self.jitter_sigma: float = net.get("jitter_sigma", 0.5)
        self.compute_delay: float = net.get("compute_delay", 1.0)

    # ── Damper control (runtime) ──────────────────────────────────────────

    def set_damper_opening(self, name: str, opening: float) -> None:
        """Set a VAV damper opening. Clamps to [0, 1]."""
        damper = self.dampers[name]
        damper.opening = max(0.0, min(1.0, opening))

    def get_damper_opening(self, name: str) -> float:
        return self.dampers[name].opening

    def zone_cell_count(self, zone_name: str) -> int:
        """Number of non-wall cells in a zone."""
        room = self.rooms[zone_name]
        return int(np.sum(~self.walls[room.slices]))

    # ── Info ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        total = int(np.prod(self.grid_shape))
        open_cells = int(np.sum(~self.walls))
        return (
            f"HVAC Environment: {self.nx}x{self.ny}x{self.nz} grid, "
            f"dx={self.dx}m, alpha={self.thermal_diffusivity} m^2/s, "
            f"dt={self.dt:.6f}s\n"
            f"  Open cells: {open_cells}/{total} "
            f"({100*open_cells/total:.1f}%)\n"
            f"  Rooms: {len(self.rooms)} "
            f"(setpoints: {', '.join(f'{r.name}={r.setpoint}C' for r in self.rooms.values())})\n"
            f"  VAV dampers: {len(self.dampers)}\n"
            f"  Heat sources: {len(self.heat_sources)}\n"
            f"  Cooling plant: Q_total={self.Q_total}, T_supply={self.supply_temperature}C"
        )
