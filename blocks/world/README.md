# Block 1: World — 3D FTCS Diffusion Engine

## Overview

The World block implements a 3D gas diffusion simulation using the Forward-Time Central-Space (FTCS) finite difference method on a NumPy array. The environment layout (rooms, hallways, doors, sources) is loaded from a JSON config file.

## Modules

| File | Purpose |
|------|---------|
| `stability.py` | FTCS stability constraint: `dt <= dx^2 / (6*D)` |
| `environment.py` | Loads JSON config, builds 3D wall mask, manages doors/sources |
| `world.py` | Core diffusion engine: `step()`, `close_door()`, `open_door()` |
| `demo.py` | Standalone animated visualization |
| `test_world.py` | 22 unit tests covering stability, environment, and diffusion |

## How to Run

```bash
# From project root, with conda env activated:
conda activate ece659

# Run the demo
python -m blocks.world.demo --config configs/environments/default_maze.json

# With options
python -m blocks.world.demo \
    --config configs/environments/default_maze.json \
    --steps 500 \
    --z-slice 2 \
    --close-door door_B_to_hallway \
    --close-at 200

# Run tests
python -m pytest blocks/world/test_world.py -v
```

### Demo Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | Path to environment JSON |
| `--steps` | 300 | Number of animation frames |
| `--z-slice` | 2 | Z-level to visualize |
| `--plot-every` | 5 | Simulation steps per animation frame |
| `--close-door` | None | Door name to close mid-simulation |
| `--close-at` | 100 | Step at which to close the door |
| `--vmax` | 5.0 | Colormap ceiling |

## Physics Details

### Diffusion Equation (FTCS)

```
phi[i,j,k]_new = phi[i,j,k] + alpha * laplacian[i,j,k]
```

where `alpha = D * dt / dx^2` and the Laplacian is computed via central differences in all 3 axes.

### Boundary Conditions

- **Grid edges**: Zero-flux (Neumann) — the slicing-based Laplacian naturally ignores cells outside the array
- **Walls**: Dirichlet zero — `phi[walls] = 0` enforced before and after each step
- **Doors (closed)**: Same as walls. Closing a door also zeros existing concentration in the door cells

### Stability

The FTCS scheme is conditionally stable. The maximum time step is:

```
dt_max = dx^2 / (6 * D)
```

By default, `dt = 0.4 * dt_max` (configurable via `safety_factor` in the JSON). If a user-specified `dt` exceeds `dt_max`, the environment loader raises an error.

## API

```python
from blocks.world import Environment, World

env = Environment("configs/environments/default_maze.json")
world = World(env)

# Run 100 steps
world.run(100)

# Access concentration field
phi = world.get_concentration()  # read-only numpy array

# Actuate a door
world.close_door("door_B_to_hallway")

# Metrics
mass = world.total_mass()
contaminated = world.contaminated_volume(threshold=0.1)
integral = world.contamination_integral(threshold=0.1)  # per-step contribution
```
