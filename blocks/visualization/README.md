# Block 2: Visualization — 3D PyVista Rendering

## Overview

Renders the VDPA simulation in interactive 3D using PyVista (VTK backend). Supports static snapshots, real-time animation, and GIF export.

**Design principle:** The Renderer is a *view* — it never steps the World. The caller owns the simulation loop. This keeps Block 1 (physics) and Block 2 (rendering) decoupled for Block 6 integration.

## Visual Elements

| Element | Representation |
|---------|---------------|
| Gas concentration | Volumetric heatmap (opacity mapped to concentration) |
| Walls | Gray wireframe cubes |
| Doors (open) | Semi-transparent green cubes |
| Doors (closed) | Opaque red cubes |
| Gas sources | Cyan spheres |

## How to Run

```bash
conda activate ece659

# Animated simulation (default)
python -m blocks.visualization.demo --config configs/environments/default_maze.json

# Static snapshot after 300 steps
python -m blocks.visualization.demo --config configs/environments/default_maze.json \
    --mode snapshot --pre-steps 300

# Close a door mid-animation at frame 50
python -m blocks.visualization.demo --config configs/environments/default_maze.json \
    --close-door door_A_to_hallway --close-at 50

# Save as GIF
python -m blocks.visualization.demo --config configs/environments/default_maze.json \
    --mode animate --gif output.gif

# Run unit + integration tests
python -m pytest blocks/visualization/test_visualization.py blocks/test_integration.py -v
```

### Demo Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | Path to environment JSON |
| `--mode` | animate | `snapshot` or `animate` |
| `--pre-steps` | 100 | Steps to run before snapshot |
| `--frames` | 200 | Animation frame count |
| `--steps-per-frame` | 5 | Sim steps per rendered frame |
| `--vmax` | 5.0 | Colormap ceiling |
| `--gif` | None | Save animation to GIF path |
| `--close-door` | None | Door name to close |
| `--close-at` | None | Frame at which to close the door (mid-animation) |

## Integration Contract (Block 1 ↔ Block 2)

Both blocks share a single `Environment` instance as their interface:

```
Environment (shared)
    ├── World reads/writes: phi, walls, sources, doors
    └── Renderer reads:     walls, doors, sources, door states
```

**Key integration rules:**

1. **Single Environment instance** — `World(env)` and `Renderer(env)` must receive the same `env` object so wall mutations propagate.
2. **Renderer auto-detects wall changes** — When `world.close_door()` mutates `env.walls`, `Renderer.snapshot()` and `AnimationState.update_frame()` detect the change via `walls_changed` and automatically rebuild meshes.
3. **Renderer never steps World** — The caller owns `world.step()`. Renderer only reads `world.get_concentration()` and metrics.
4. **AnimationState for external loops** — Use `create_animation_plotter()` when Block 6's asyncio event loop drives the simulation:

```python
anim = renderer.create_animation_plotter(world, off_screen=False)
anim.start()
for _ in range(100):
    world.step()          # Caller drives physics
    anim.update_frame()   # Renderer just refreshes visuals
anim.finish()
```

5. **frame_callback for mid-sim events** — `animate()` accepts a callback for actuator triggers:

```python
def on_frame(world, frame):
    if frame == 50:
        world.close_door("door_A_to_hallway")

renderer.animate(world, n_frames=200, frame_callback=on_frame)
```

## API

```python
from blocks.world import Environment, World
from blocks.visualization import Renderer

env = Environment("configs/environments/default_maze.json")
world = World(env)
renderer = Renderer(env, clim=(0, 10), cmap="inferno")

# Static snapshot
world.run(200)
renderer.show(world, title="After 200 steps")

# Convenience animation (renderer calls world.step internally)
renderer.animate(world, n_frames=100, steps_per_frame=5)

# Frame-by-frame control (caller drives simulation)
anim = renderer.create_animation_plotter(world)
anim.start()
for _ in range(100):
    world.step()
    anim.update_frame()
anim.finish()
```

### Renderer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opacity_range` | (0.0, 0.6) | Min/max opacity for volume rendering |
| `clim` | (0.0, 5.0) | Concentration color limits |
| `cmap` | "inferno" | Matplotlib colormap name |
| `background` | "black" | Background color |
| `window_size` | (1400, 800) | Render window dimensions |
