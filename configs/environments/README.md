# Environment Configuration Schema

Environment JSON files define the 3D grid layout for VDPA simulations. Place configs in this directory and reference them with `--config`.

## Schema

### `grid` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nx` | int | 20 | Grid cells in X |
| `ny` | int | 20 | Grid cells in Y |
| `nz` | int | 5 | Grid cells in Z (height) |
| `dx` | float | 1.0 | Cell spacing in meters |

### `physics` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `diffusion_coefficient` | float | — | D in m^2/s (typical indoor: 0.01–0.1) |
| `dt` | float/null | null | Time step in seconds. If null, auto-computed |
| `safety_factor` | float | 0.4 | Fraction of max stable dt (only used when dt is null) |

### `rooms` (required)

Array of rectangular open regions. **Everything outside rooms is wall.**

```json
{
  "name": "room_A",
  "bounds": {
    "x_min": 1, "x_max": 8,
    "y_min": 1, "y_max": 8,
    "z_min": 0, "z_max": 5
  }
}
```

Bounds use Python slice convention: `x_max` is exclusive.

### `doors` (optional)

Thin slabs connecting rooms through walls. Can be toggled at runtime.

```json
{
  "name": "door_A_to_hall",
  "bounds": { "x_min": 8, "x_max": 9, "y_min": 4, "y_max": 6, "z_min": 0, "z_max": 4 },
  "state": "open"
}
```

State is `"open"` or `"closed"`. Open doors carve through walls; closed doors are walls.

### `sources` (optional)

Gas emission points.

```json
{
  "name": "gas_leak",
  "position": { "x": 4, "y": 4, "z": 2 },
  "rate": 5.0,
  "start_time": 0,
  "end_time": null
}
```

- `rate`: concentration units per second injected into the cell
- `end_time`: null means the source never stops
- Source position must be inside a room (not a wall)

### `sensors` (optional)

Configures sensor placement for Block 3 (Network).

**Grid placement** (uniform spacing):
```json
{
  "placement": "grid",
  "spacing": 3,
  "z_levels": [2],
  "communication_radius": 5.0
}
```

**Random placement**:
```json
{
  "placement": "random",
  "count": 20,
  "seed": 42,
  "z_levels": [2],
  "communication_radius": 5.0
}
```

**Manual placement**:
```json
{
  "placement": "manual",
  "communication_radius": 5.0,
  "nodes": [
    {"name": "s1", "position": {"x": 3, "y": 3, "z": 2}},
    {"name": "s2", "position": {"x": 7, "y": 7, "z": 2}}
  ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `placement` | string | "grid" | Strategy: `grid`, `random`, or `manual` |
| `spacing` | int | 3 | Grid spacing (grid mode only) |
| `count` | int | 20 | Number of sensors (random mode only) |
| `seed` | int | 42 | RNG seed for reproducibility (random mode only) |
| `z_levels` | list[int] | [nz/2] | Z-levels to place sensors on |
| `communication_radius` | float | 5.0 | Max distance for sensor communication edges |
| `nodes` | list | [] | Explicit positions (manual mode only) |

### `noise` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensor_sigma` | float | 0.0 | Gaussian noise std dev for sensor readings (used by Block 4) |
| `source_rate_sigma` | float | 0.0 | Gaussian noise on source emission rate each step |

## Example: Creating a New Environment

1. Copy `default_maze.json` as a starting point
2. Define rooms as rectangular regions that carve open space
3. Place doors at the boundaries between rooms (1-cell thick slabs)
4. Place sources inside rooms
5. Run the demo to verify: `python -m blocks.world.demo --config configs/environments/your_config.json`
