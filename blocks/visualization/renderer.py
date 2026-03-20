"""
3D Visualization Renderer using PyVista.

Renders the gas concentration field as a volumetric heatmap, walls as solid
blocks, and door states as colored surfaces. Supports both static snapshots
and animated time-stepping.

Design: The Renderer is a *view* — it never steps the World itself. The
caller owns the simulation loop and calls renderer methods to update visuals.
This keeps Block 1 (physics) and Block 2 (rendering) cleanly decoupled for
Block 6 integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World


class Renderer:
    """Interactive 3D renderer for the VDPA simulation."""

    def __init__(
        self,
        env: Environment,
        *,
        opacity_range: tuple[float, float] = (0.0, 0.6),
        clim: tuple[float, float] = (0.0, 5.0),
        cmap: str = "inferno",
        background: str = "black",
        window_size: tuple[int, int] = (1400, 800),
    ):
        self.env = env
        self.opacity_range = opacity_range
        self.clim = clim
        self.cmap = cmap
        self.background = background
        self.window_size = window_size

        # Track wall state to detect changes
        self._wall_hash: int = self._hash_walls()
        self._wall_mesh = self._build_wall_mesh()
        self._door_meshes: dict[str, pv.PolyData] = self._build_door_meshes()

    # ── State tracking ────────────────────────────────────────────────────

    def _hash_walls(self) -> int:
        """Hash of the wall array to detect changes."""
        return hash(self.env.walls.data.tobytes())

    @property
    def walls_changed(self) -> bool:
        """Check if env.walls has been mutated since last mesh build."""
        return self._hash_walls() != self._wall_hash

    def rebuild_meshes(self) -> None:
        """Rebuild wall and door meshes from current env state.

        Call this after any door open/close to keep visuals in sync.
        This is called automatically by update_frame() when walls change.
        """
        self._wall_mesh = self._build_wall_mesh()
        self._door_meshes = self._build_door_meshes()
        self._wall_hash = self._hash_walls()

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

    def _build_door_meshes(self) -> dict[str, pv.PolyData]:
        """Create a mesh for each door region."""
        meshes = {}
        for name, door in self.env.doors.items():
            sx, sy, sz = door.slices
            xs = np.arange(sx.start, sx.stop)
            ys = np.arange(sy.start, sy.stop)
            zs = np.arange(sz.start, sz.stop)
            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
            coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(float)
            if len(coords) == 0:
                continue
            points = pv.PolyData(coords)
            cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
            meshes[name] = points.glyph(geom=cube, scale=False, orient=False)
        return meshes

    def _build_concentration_volume(self, phi: np.ndarray) -> pv.ImageData:
        """Create a structured grid from the concentration field."""
        nx, ny, nz = phi.shape
        grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
        grid.origin = (-0.5, -0.5, -0.5)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.cell_data["concentration"] = phi.ravel(order="F")
        return grid

    def _build_source_markers(self) -> pv.PolyData:
        """Create sphere markers for gas sources."""
        if not self.env.sources:
            return pv.PolyData()
        coords = np.array([s.position for s in self.env.sources], dtype=float)
        points = pv.PolyData(coords)
        sphere = pv.Sphere(radius=0.4)
        return points.glyph(geom=sphere, scale=False, orient=False)

    # ── Snapshot rendering ────────────────────────────────────────────────

    def snapshot(self, world: World, title: str = "") -> pv.Plotter:
        """Render a single frame of the simulation state. Returns the plotter."""
        # Auto-rebuild if walls changed
        if self.walls_changed:
            self.rebuild_meshes()

        pl = pv.Plotter(window_size=self.window_size)
        pl.set_background(self.background)
        self._add_actors(pl, world)
        if title:
            pl.add_title(title, font_size=12, color="white")
        return pl

    def _add_actors(self, pl: pv.Plotter, world: World) -> None:
        """Add all visual actors to the plotter."""
        phi = world.get_concentration()
        vol = self._build_concentration_volume(phi)
        opacity = np.linspace(self.opacity_range[0], self.opacity_range[1], 10)
        pl.add_volume(
            vol, scalars="concentration", cmap=self.cmap,
            clim=self.clim, opacity=opacity, shade=False,
        )

        if self._wall_mesh.n_points > 0:
            pl.add_mesh(
                self._wall_mesh, color="gray", opacity=0.15,
                style="wireframe", line_width=0.5,
            )

        for name, mesh in self._door_meshes.items():
            state = self.env.get_door_state(name)
            color = "red" if state == "closed" else "lime"
            door_opacity = 0.8 if state == "closed" else 0.3
            pl.add_mesh(mesh, color=color, opacity=door_opacity,
                        label=f"{name} ({state})")

        source_mesh = self._build_source_markers()
        if source_mesh.n_points > 0:
            pl.add_mesh(source_mesh, color="cyan", opacity=0.9, label="Gas Source")

        pl.add_legend(bcolor="black", face=None, size=(0.2, 0.15))
        pl.add_axes()
        pl.show_grid(xtitle="X", ytitle="Y", ztitle="Z",
                     color="white", font_size=8)

    def show(self, world: World, title: str = "") -> None:
        """Render and display interactively."""
        pl = self.snapshot(world, title)
        pl.show()

    # ── Animation with external loop support ──────────────────────────────

    def create_animation_plotter(
        self,
        world: World,
        *,
        title_prefix: str = "VDPA",
        off_screen: bool = False,
        gif_path: str | None = None,
    ) -> "AnimationState":
        """Create a plotter ready for frame-by-frame updates.

        Returns an AnimationState that the caller uses to update each frame.
        This decouples the render loop from the simulation loop.
        """
        pl = pv.Plotter(
            window_size=self.window_size,
            off_screen=off_screen or (gif_path is not None),
        )
        pl.set_background(self.background)

        state = AnimationState(
            renderer=self,
            plotter=pl,
            world=world,
            title_prefix=title_prefix,
            gif_path=gif_path,
        )
        state._init_actors()
        return state

    def animate(
        self,
        world: World,
        n_frames: int = 200,
        steps_per_frame: int = 5,
        *,
        title_prefix: str = "VDPA",
        gif_path: str | None = None,
        frame_callback: Callable[[World, int], None] | None = None,
    ) -> None:
        """Convenience method: run simulation + render in a single loop.

        Args:
            world: The World instance to step.
            n_frames: Number of animation frames.
            steps_per_frame: Simulation steps between each render.
            title_prefix: Prefix for the window title.
            gif_path: If provided, save the animation as a GIF.
            frame_callback: Called with (world, frame_index) after stepping
                but before rendering. Use this to trigger door changes or
                other mid-simulation events.
        """
        anim = self.create_animation_plotter(
            world, title_prefix=title_prefix, gif_path=gif_path,
        )
        anim.start()

        for frame in range(n_frames):
            for _ in range(steps_per_frame):
                world.step()

            if frame_callback is not None:
                frame_callback(world, frame)

            anim.update_frame()

        anim.finish()


class AnimationState:
    """Manages the mutable state of an in-progress animation.

    This object is returned by Renderer.create_animation_plotter() and
    provides frame-by-frame control. The caller owns the simulation loop.
    """

    def __init__(
        self,
        renderer: Renderer,
        plotter: pv.Plotter,
        world: World,
        title_prefix: str,
        gif_path: str | None,
    ):
        self.renderer = renderer
        self.pl = plotter
        self.world = world
        self.title_prefix = title_prefix
        self.gif_path = gif_path

        self._vol_actor = None
        self._wall_actor = None
        self._door_actors: dict[str, pv.Actor] = {}
        self._text_actor = None
        self._opacity = np.linspace(
            renderer.opacity_range[0], renderer.opacity_range[1], 10,
        )

    def _init_actors(self) -> None:
        """Add initial actors to the plotter."""
        r = self.renderer
        phi = self.world.get_concentration()
        vol = r._build_concentration_volume(phi)

        self._vol_actor = self.pl.add_volume(
            vol, scalars="concentration", cmap=r.cmap,
            clim=r.clim, opacity=self._opacity, shade=False,
        )

        if r._wall_mesh.n_points > 0:
            self._wall_actor = self.pl.add_mesh(
                r._wall_mesh, color="gray", opacity=0.15,
                style="wireframe", line_width=0.5,
            )

        for name, mesh in r._door_meshes.items():
            state = r.env.get_door_state(name)
            color = "red" if state == "closed" else "lime"
            opacity = 0.8 if state == "closed" else 0.3
            self._door_actors[name] = self.pl.add_mesh(
                mesh, color=color, opacity=opacity,
            )

        source_mesh = r._build_source_markers()
        if source_mesh.n_points > 0:
            self.pl.add_mesh(source_mesh, color="cyan", opacity=0.9)

        self.pl.add_axes()
        self._text_actor = self.pl.add_text(
            f"{self.title_prefix} | Step 0",
            position="upper_left", font_size=10, color="white",
        )

    def start(self) -> None:
        """Show the plotter window and begin interactive updates."""
        if self.gif_path:
            self.pl.open_gif(self.gif_path)
        self.pl.show(interactive_update=True, auto_close=False)

    def update_frame(self) -> None:
        """Refresh all visual actors from current World state.

        Call this once per frame after stepping the World. Automatically
        detects wall changes (door open/close) and rebuilds meshes.
        """
        r = self.renderer

        # Detect wall changes and rebuild meshes + actors
        if r.walls_changed:
            r.rebuild_meshes()
            # Replace wall actor
            if self._wall_actor is not None:
                self.pl.remove_actor(self._wall_actor)
            if r._wall_mesh.n_points > 0:
                self._wall_actor = self.pl.add_mesh(
                    r._wall_mesh, color="gray", opacity=0.15,
                    style="wireframe", line_width=0.5,
                )
            # Replace door actors with updated colors
            for name in list(self._door_actors.keys()):
                self.pl.remove_actor(self._door_actors[name])
            self._door_actors.clear()
            for name, mesh in r._door_meshes.items():
                state = r.env.get_door_state(name)
                color = "red" if state == "closed" else "lime"
                opacity = 0.8 if state == "closed" else 0.3
                self._door_actors[name] = self.pl.add_mesh(
                    mesh, color=color, opacity=opacity,
                )

        # Update volume
        phi = self.world.get_concentration()
        new_vol = r._build_concentration_volume(phi)
        if self._vol_actor is not None:
            self.pl.remove_actor(self._vol_actor)
        self._vol_actor = self.pl.add_volume(
            new_vol, scalars="concentration", cmap=r.cmap,
            clim=r.clim, opacity=self._opacity, shade=False,
        )

        # Update text
        step = self.world.step_count
        mass = self.world.total_mass()
        contam = self.world.contaminated_volume()
        self._text_actor.SetText(
            0,
            f"{self.title_prefix} | Step {step} | "
            f"Mass: {mass:.1f} | Contaminated: {contam} cells",
        )

        self.pl.render()
        if self.gif_path:
            self.pl.write_frame()

    def finish(self) -> None:
        """Close GIF writer or show final interactive view."""
        if self.gif_path:
            self.pl.close()
        else:
            self.pl.show(interactive=True)
