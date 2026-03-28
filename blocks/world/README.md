# Block 1: World — 3D HVAC Thermal Physics Engine

## Overview

The World block implements a 3D thermal simulation using the Forward-Time Central-Space (FTCS) finite difference method. The environment layout (rooms, hallways, VAV dampers, heat sources with occupancy profiles) is loaded from a JSON config file.

**Physics equation**: `dT/dt = alpha * nabla^2(T) + Q_heat - Q_cool`

## Modules

| File | Purpose |
|------|---------|
| `stability.py` | CFL stability constraint: `dt <= dx^2 / (6*alpha)` |
| `environment.py` | Loads JSON config, builds 3D wall mask, manages zones/dampers/heat sources |
| `world.py` | Core thermal engine: `step()`, `set_damper()`, zone-distributed cooling |
| `test_world.py` | 45 unit tests covering stability, environment, diffusion, cooling, occupancy |

## How to Run

```bash
conda activate ece659

# Run the demo
python -m blocks.world.demo --config configs/environments/university_floor.json

# Run tests
python -m pytest blocks/world/test_world.py -v
```

## Physics Details

### Heat Diffusion (FTCS)

```
T_new = T + alpha * laplacian(T) * dt
```

where `alpha` is the thermal diffusivity (~0.02 m^2/s for air) and the Laplacian is computed via central differences in all 3 axes.

### Cooling Model

VAV dampers inject cooling into their entire zone. Cooling is proportional to damper opening and temperature difference:

```
Q_cool = opening * max_flow * (T - T_supply)
```

Total cooling is bounded by the plant budget: `sum(Q_cool_i) <= Q_total`.

### Heat Sources

Zone-wide heat injection with optional occupancy profiles (time-varying rate via keyframe schedule).

### Boundary Conditions

- **Grid edges**: Zero-flux (Neumann)
- **Walls**: Held at ambient temperature
- **Hallways**: Open connections between rooms

### Stability

```
dt_max = dx^2 / (6 * alpha)
dt = safety_factor * dt_max  (default safety_factor = 0.4)
```

## API

```python
from blocks.world import Environment, World

env = Environment("configs/environments/university_floor.json")
world = World(env)

# Run 100 steps
world.run(100)

# Access temperature field
T = world.T  # 3D numpy array

# Actuate a damper
world.set_damper("vav_classroom_101", 0.8)

# Metrics
print(world.metrics())
print(f"Max overshoot: {world.max_overshoot():.2f}C")
print(f"Comfort violation: {world.comfort_violation():.4f}")
```
