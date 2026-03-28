# Block 2: Visualization — 3D PyVista Rendering

## Overview

Renders the HVAC simulation in interactive 3D using PyVista (VTK backend). Displays temperature fields as volumetric heatmaps with wall geometry, damper positions, and zone overlays.

**Design principle:** The Renderer is a *view* — it never steps the World. The caller owns the simulation loop.

## Visual Elements

| Element | Representation |
|---------|---------------|
| Temperature field | Volumetric heatmap (blue=cold, red=hot) |
| Walls | Gray wireframe cubes |
| VAV dampers | Colored markers showing opening level |
| Zone boundaries | Semi-transparent overlays |

## How to Run

```bash
conda activate ece659

# Animated simulation
python -m blocks.visualization.demo --config configs/environments/university_floor.json

# Static snapshot after 300 steps
python -m blocks.visualization.demo --config configs/environments/university_floor.json \
    --mode snapshot --pre-steps 300

# Run tests
python -m pytest blocks/visualization/test_visualization.py -v
```

## API

```python
from blocks.world import Environment, World
from blocks.visualization import Renderer

env = Environment("configs/environments/university_floor.json")
world = World(env)
renderer = Renderer(env)

# Static snapshot
world.run(200)
renderer.show(world, title="After 200 steps")

# Frame-by-frame control
anim = renderer.create_animation_plotter(world)
anim.start()
for _ in range(100):
    world.step()
    anim.update_frame()
anim.finish()
```

## Integration Contract (Block 1 <-> Block 2)

Both blocks share a single `Environment` instance:

```
Environment (shared)
    ├── World reads/writes: T, walls, dampers, heat sources
    └── Renderer reads:     walls, dampers, zone boundaries
```

**Key rule:** Renderer never steps World. The caller owns `world.step()`.
