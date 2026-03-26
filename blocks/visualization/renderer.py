"""
3D Visualization Renderer using PyVista.

Renders the temperature field as a volumetric heatmap, walls as solid
blocks, and VAV damper positions as markers. Supports both static snapshots
and animated time-stepping.

Design: The Renderer is a *view* — it never steps the World itself. The
caller owns the simulation loop and calls renderer methods to update visuals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World


class Renderer:
    """Interactive 3D renderer for the HVAC simulation."""

    def __init__(
        self,
        env: Environment,
        *,
        opacity_range: tuple[float, float] = (0.0, 0.6),
        clim: tuple[float, float] | None = None,
        cmap: str = "coolwarm",
        background: str = "black",
        window_size: tuple[int, int] = (1400, 800),
    ):
        self.env = env
        self.opacity_range = opacity_range
        self.clim = clim or (env.supply_temperature, env.ambient_temperature + 15)
        self.cmap = cmap
        self.background = background
        self.window_size = window_size

        self._wall_mesh = self._build_wall_mesh()
        self._damper_markers = self._build_damper_markers()

    # ── Mesh builders ─────────────────────────────────────────────────────

    def _build_wall_mesh(self) -> pv.PolyData:
        """Create a mesh of cubes for all wall cells."""
        walls = self.env.walls
        coords = np.argwhere(walls).astype(float)
        if len(coords) == 0:
            return pv.PolyData()
        points = pv.PolyData(coords)
        cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
        return points.glyph(geom=cube, scale=False, orient=False)

    def _build_damper_markers(self) -> pv.PolyData:
        """Create markers at VAV damper positions."""
        if not self.env.dampers:
            return pv.PolyData()
        coords = np.array(
            [d.position for d in self.env.dampers.values()], dtype=float
        )
        points = pv.PolyData(coords)
        sphere = pv.Sphere(radius=0.4)
        return points.glyph(geom=sphere, scale=False, orient=False)

    def _build_temperature_volume(self, T: np.ndarray) -> pv.ImageData:
        """Create a structured grid from the temperature field."""
        nx, ny, nz = T.shape
        grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
        grid.origin = (-0.5, -0.5, -0.5)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.cell_data["temperature"] = T.ravel(order="F")
        return grid

    # ── Snapshot rendering ────────────────────────────────────────────────

    def snapshot(self, world: World, title: str = "") -> pv.Plotter:
        """Render a single frame of the simulation state. Returns the plotter."""
        pl = pv.Plotter(window_size=self.window_size)
        pl.set_background(self.background)
        self._add_actors(pl, world)
        if title:
            pl.add_title(title, font_size=12, color="white")
        return pl

    def _add_actors(self, pl: pv.Plotter, world: World) -> None:
        """Add all visual actors to the plotter."""
        vol = self._build_temperature_volume(world.T)
        opacity = np.linspace(self.opacity_range[0], self.opacity_range[1], 10)
        pl.add_volume(
            vol, scalars="temperature", cmap=self.cmap,
            clim=self.clim, opacity=opacity, shade=False,
        )

        if self._wall_mesh.n_points > 0:
            pl.add_mesh(
                self._wall_mesh, color="gray", opacity=0.15,
                style="wireframe", line_width=0.5,
            )

        if self._damper_markers.n_points > 0:
            pl.add_mesh(
                self._damper_markers, color="cyan", opacity=0.9,
                label="VAV Dampers",
            )

        pl.add_legend(bcolor="black", face=None, size=(0.2, 0.15))
        pl.add_axes()
        pl.show_grid(xtitle="X", ytitle="Y", ztitle="Z",
                     color="white", font_size=8)

    def show(self, world: World, title: str = "") -> None:
        """Render and display interactively."""
        pl = self.snapshot(world, title)
        pl.show()
