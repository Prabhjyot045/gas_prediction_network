"""
World — 3D FTCS diffusion engine.

Implements the Forward-Time Central-Space finite difference scheme for the
scalar diffusion equation:

    ∂φ/∂t = D ∇²φ + S

where φ is gas concentration, D is the diffusion coefficient, and S is the
source injection term.
"""

from __future__ import annotations

import numpy as np

from .environment import Environment


class World:
    """3D diffusion simulation driven by an Environment config."""

    def __init__(self, env: Environment):
        self.env = env
        self.phi = np.zeros(env.grid_shape, dtype=np.float64)
        self.time: float = 0.0
        self.step_count: int = 0

        # Pre-compute the diffusion multiplier: D * dt / dx^2
        self._alpha = env.diffusion_coefficient * env.dt / env.dx**2

    # ── Core physics ──────────────────────────────────────────────────────

    def _shifted(self, axis: int, direction: int) -> np.ndarray:
        """Get neighbor values along `axis` in `direction` (+1 or -1).

        Applies Neumann (zero-flux) BCs:
        - At grid edges: neighbor = self (reflection)
        - At internal walls: neighbor = self (no flux into walls)

        This ensures walls are impermeable reflectors, not absorbers.
        """
        phi = self.phi
        walls = self.env.walls
        result = np.empty_like(phi)

        if direction == 1:  # forward neighbor (i+1)
            s_dst = [slice(None)] * 3
            s_src = [slice(None)] * 3
            s_dst[axis] = slice(None, -1)
            s_src[axis] = slice(1, None)
            result[tuple(s_dst)] = phi[tuple(s_src)]
            # Grid edge: reflect
            s_edge = [slice(None)] * 3
            s_edge[axis] = -1
            result[tuple(s_edge)] = phi[tuple(s_edge)]
            # Wall mask for the shifted neighbor
            w = np.ones_like(walls)
            w[tuple(s_dst)] = walls[tuple(s_src)]
        else:  # backward neighbor (i-1)
            s_dst = [slice(None)] * 3
            s_src = [slice(None)] * 3
            s_dst[axis] = slice(1, None)
            s_src[axis] = slice(None, -1)
            result[tuple(s_dst)] = phi[tuple(s_src)]
            # Grid edge: reflect
            s_edge = [slice(None)] * 3
            s_edge[axis] = 0
            result[tuple(s_edge)] = phi[tuple(s_edge)]
            # Wall mask for the shifted neighbor
            w = np.ones_like(walls)
            w[tuple(s_dst)] = walls[tuple(s_src)]

        # Where neighbor is a wall, use center value (zero gradient = zero flux)
        result = np.where(w, phi, result)
        return result

    def _laplacian(self) -> np.ndarray:
        """Compute the discrete 3D Laplacian with Neumann BCs at walls.

        For each of the 6 neighbors (±x, ±y, ±z):
        - If the neighbor is a non-wall cell, use its phi value
        - If the neighbor is a wall or grid edge, use the center cell's value
        This gives ∂φ/∂n = 0 at every wall face (impermeable, no absorption).
        """
        lap = np.zeros_like(self.phi)
        for axis in range(3):
            fwd = self._shifted(axis, +1)
            bwd = self._shifted(axis, -1)
            lap += fwd + bwd - 2.0 * self.phi
        return lap

    def _inject_sources(self) -> None:
        """Add gas from active sources."""
        for src in self.env.sources:
            if src.is_active(self.time):
                rate = src.rate
                if self.env.source_rate_sigma > 0:
                    rate += np.random.normal(0, self.env.source_rate_sigma)
                    rate = max(0.0, rate)
                self.phi[src.position] += rate * self.env.dt

    def _enforce_walls(self) -> None:
        """Zero out concentration inside wall cells.

        Walls are solid — they cannot hold gas. This is NOT an absorbing BC;
        the Laplacian already uses Neumann (zero-flux) BCs so gas reflects
        off walls rather than flowing into them.
        """
        self.phi[self.env.walls] = 0.0

    # ── Public interface ──────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the simulation by one time step."""
        # Diffusion update: φ_new = φ + α * ∇²φ
        # The Laplacian handles Neumann BCs at walls internally
        lap = self._laplacian()
        self.phi += self._alpha * lap

        # Inject sources
        self._inject_sources()

        # Zero wall cells (walls are solid, cannot hold gas)
        self._enforce_walls()

        # Clamp negative values (numerical noise)
        np.clip(self.phi, 0.0, None, out=self.phi)

        self.time += self.env.dt
        self.step_count += 1

    def run(self, n_steps: int) -> None:
        """Run multiple time steps."""
        for _ in range(n_steps):
            self.step()

    def close_door(self, name: str) -> None:
        """Close a door — updates wall mask AND zeros concentration in the door cells.

        This is a hard boundary: gas already in the door region is removed,
        and no further diffusion occurs through this region.
        """
        self.env.close_door(name)
        door = self.env.doors[name]
        self.phi[door.slices] = 0.0

    def open_door(self, name: str) -> None:
        """Open a door — removes wall mask at door cells.

        Concentration in the newly opened cells starts at zero (physically:
        the door frame was sealed and contains no gas).
        """
        self.env.open_door(name)

    def get_concentration(self) -> np.ndarray:
        """Return a read-only view of the concentration field."""
        result = self.phi.copy()
        result.flags.writeable = False
        return result

    def total_mass(self) -> float:
        """Total gas in the system (sum of concentration over all open cells)."""
        return float(np.sum(self.phi[~self.env.walls]))

    def contaminated_volume(self, threshold: float = 0.1) -> int:
        """Number of open cells with concentration above threshold."""
        above = (self.phi > threshold) & (~self.env.walls)
        return int(np.sum(above))

    def metrics(self, threshold: float = 0.1) -> dict:
        """Return all current metrics as a flat dictionary for reporting."""
        return {
            "step": self.step_count,
            "time": round(self.time, 6),
            "total_mass": self.total_mass(),
            "contaminated_volume": self.contaminated_volume(threshold),
            "contamination_integral_dt": self.contamination_integral(threshold),
            "peak_concentration": float(np.max(self.phi)),
        }

    def contamination_integral(self, threshold: float = 0.1) -> float:
        """Contribution to the time-integrated contamination volume for this step.

        Returns: count_of_contaminated_cells * dx^3 * dt
        Call this every step and accumulate for the benchmark metric.
        """
        vol = self.contaminated_volume(threshold)
        return vol * self.env.dx**3 * self.env.dt
