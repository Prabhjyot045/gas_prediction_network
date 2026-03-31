"""
World — 3D heat diffusion engine for HVAC simulation.

Implements the governing equation:

    dT/dt = alpha * nabla^2(T) + Q_heat - Q_cool

where T is temperature, alpha is thermal diffusivity, Q_heat is zone-level
heat injection (occupancy/equipment/solar loads), and Q_cool is VAV damper
cooling applied uniformly across each damper's assigned zone.

Numerical scheme: FTCS (Forward-Time Central-Space) on a uniform 3D grid
with Neumann (zero-flux) boundary conditions at walls and domain edges.
Time step is set automatically by the diffusion stability criterion:

    dt = safety_factor * dx^2 / (6 * alpha)     [3D FTCS limit]

Cooling pulls each zone's temperature toward the supply temperature
proportionally to damper opening, capped globally at Q_total so that the
shared plant capacity constraint is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .environment import Environment


class World:
    """3D thermal simulation driven by an HVAC Environment config."""

    def __init__(self, env: Environment):
        self.env = env

        # Temperature field initialized to ambient
        self.T = np.full(env.grid_shape, env.ambient_temperature, dtype=np.float64)
        self.time: float = 0.0
        self.step_count: int = 0

        # Pre-compute diffusion multiplier: alpha * dt / dx^2
        self._alpha = env.thermal_diffusivity * env.dt / env.dx**2

    # ── Core physics ──────────────────────────────────────────────────────

    def _shifted(self, axis: int, direction: int) -> np.ndarray:
        """Get neighbor values along `axis` in `direction` (+1 or -1).

        Applies Neumann (zero-flux) BCs:
        - At grid edges: neighbor = self (reflection)
        - At internal walls: neighbor = self (no flux into walls)
        """
        T = self.T
        walls = self.env.walls
        result = np.empty_like(T)

        if direction == 1:  # forward neighbor (i+1)
            s_dst = [slice(None)] * 3
            s_src = [slice(None)] * 3
            s_dst[axis] = slice(None, -1)
            s_src[axis] = slice(1, None)
            result[tuple(s_dst)] = T[tuple(s_src)]
            s_edge = [slice(None)] * 3
            s_edge[axis] = -1
            result[tuple(s_edge)] = T[tuple(s_edge)]
            w = np.ones_like(walls)
            w[tuple(s_dst)] = walls[tuple(s_src)]
        else:  # backward neighbor (i-1)
            s_dst = [slice(None)] * 3
            s_src = [slice(None)] * 3
            s_dst[axis] = slice(1, None)
            s_src[axis] = slice(None, -1)
            result[tuple(s_dst)] = T[tuple(s_src)]
            s_edge = [slice(None)] * 3
            s_edge[axis] = 0
            result[tuple(s_edge)] = T[tuple(s_edge)]
            w = np.ones_like(walls)
            w[tuple(s_dst)] = walls[tuple(s_src)]

        result = np.where(w, T, result)
        return result

    def _laplacian(self) -> np.ndarray:
        """Compute the discrete 3D Laplacian with Neumann BCs at walls."""
        lap = np.zeros_like(self.T)
        for axis in range(3):
            fwd = self._shifted(axis, +1)
            bwd = self._shifted(axis, -1)
            lap += fwd + bwd - 2.0 * self.T
        return lap

    def _inject_heat(self) -> None:
        """Add heat from active sources (distributed uniformly over zone cells).

        Uses ``current_rate(t)`` which respects occupancy profiles — the heat
        rate changes over time as people enter and leave rooms.
        """
        for src in self.env.heat_sources:
            rate = src.current_rate(self.time)
            if rate > 0:
                room = self.env.rooms[src.zone]
                zone_mask = ~self.env.walls[room.slices]
                n_cells = int(np.sum(zone_mask))
                if n_cells > 0:
                    self.T[room.slices][zone_mask] += rate * self.env.dt

    def _apply_cooling(self) -> None:
        """Apply VAV damper cooling distributed across each zone.

        Each damper supplies cooled air to its entire zone (not just the
        damper position). The cooling rate per cell pulls temperature
        toward supply temperature:

            T_new = T - (flow / n_cells) * (T - T_supply) * dt

        Total cooling across all dampers is capped at Q_total.
        """
        # Compute total requested flow
        total_flow = sum(d.current_flow for d in self.env.dampers.values())

        # Scale factor if total exceeds budget
        if total_flow > self.env.Q_total and total_flow > 0:
            scale = self.env.Q_total / total_flow
        else:
            scale = 1.0

        T_supply = self.env.supply_temperature

        for damper in self.env.dampers.values():
            room = self.env.rooms[damper.zone]
            zone_mask = ~self.env.walls[room.slices]
            n_cells = int(np.sum(zone_mask))
            if n_cells == 0:
                continue

            flow = damper.current_flow * scale
            # Per-cell cooling distributed across zone
            zone_T = self.T[room.slices][zone_mask]
            delta = (flow / n_cells) * (zone_T - T_supply) * self.env.dt
            # Clamp: can't cool below supply temperature
            max_delta = np.maximum(0.0, zone_T - T_supply)
            delta = np.minimum(delta, max_delta)
            self.T[room.slices][zone_mask] -= delta

    def _enforce_walls(self) -> None:
        """Set wall cells to ambient temperature (walls are thermally neutral)."""
        self.T[self.env.walls] = self.env.ambient_temperature

    # ── Public interface ──────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one time step."""
        # Diffusion: T_new = T + alpha * dt/dx^2 * laplacian(T)
        lap = self._laplacian()
        self.T += self._alpha * lap

        # Heat injection from sources
        self._inject_heat()

        # Cooling from VAV dampers
        self._apply_cooling()

        # Enforce wall condition
        self._enforce_walls()

        self.time += self.env.dt
        self.step_count += 1

    def run(self, n_steps: int) -> None:
        """Run multiple time steps."""
        for _ in range(n_steps):
            self.step()

    # ── Damper control ────────────────────────────────────────────────────

    def set_damper(self, name: str, opening: float) -> None:
        """Set a VAV damper opening (delegates to env)."""
        self.env.set_damper_opening(name, opening)

    # ── Zone queries ──────────────────────────────────────────────────────

    def zone_mean_temperature(self, zone_name: str) -> float:
        """Mean temperature in a zone (non-wall cells only)."""
        room = self.env.rooms[zone_name]
        zone_T = self.T[room.slices]
        zone_mask = ~self.env.walls[room.slices]
        if np.sum(zone_mask) == 0:
            return self.env.ambient_temperature
        return float(np.mean(zone_T[zone_mask]))

    def zone_max_temperature(self, zone_name: str) -> float:
        """Maximum temperature in a zone."""
        room = self.env.rooms[zone_name]
        zone_T = self.T[room.slices]
        zone_mask = ~self.env.walls[room.slices]
        if np.sum(zone_mask) == 0:
            return self.env.ambient_temperature
        return float(np.max(zone_T[zone_mask]))

    def zone_overshoot(self, zone_name: str) -> float:
        """Degrees above setpoint (0 if within setpoint)."""
        room = self.env.rooms[zone_name]
        mean_T = self.zone_mean_temperature(zone_name)
        return max(0.0, mean_T - room.setpoint)

    def total_overshoot(self) -> float:
        """Sum of overshoots across all zones."""
        return sum(self.zone_overshoot(z) for z in self.env.rooms)

    def max_overshoot(self) -> float:
        """Maximum overshoot across all zones."""
        if not self.env.rooms:
            return 0.0
        return max(self.zone_overshoot(z) for z in self.env.rooms)

    def comfort_violation(self) -> float:
        """Time-integrated overshoot for this step (sum_zones overshoot * dt)."""
        return self.total_overshoot() * self.env.dt

    def total_cooling_energy(self) -> float:
        """Total cooling power being applied this step (flow * dt)."""
        total_flow = sum(d.current_flow for d in self.env.dampers.values())
        scale = 1.0
        if total_flow > self.env.Q_total and total_flow > 0:
            scale = self.env.Q_total / total_flow
        return total_flow * scale * self.env.dt

    # ── Metrics ───────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Return all current metrics as a flat dictionary for reporting."""
        zone_temps = {
            z: round(self.zone_mean_temperature(z), 4)
            for z in self.env.rooms
        }
        zone_overshoots = {
            z: round(self.zone_overshoot(z), 4)
            for z in self.env.rooms
        }
        return {
            "step": self.step_count,
            "time": round(self.time, 6),
            "zone_temperatures": zone_temps,
            "zone_overshoots": zone_overshoots,
            "max_overshoot": round(self.max_overshoot(), 4),
            "total_overshoot": round(self.total_overshoot(), 4),
            "comfort_violation_dt": round(self.comfort_violation(), 6),
            "peak_temperature": round(float(np.max(self.T)), 4),
            "mean_temperature": round(float(np.mean(self.T[~self.env.walls])), 4),
            "cooling_energy_dt": round(self.total_cooling_energy(), 6),
        }
