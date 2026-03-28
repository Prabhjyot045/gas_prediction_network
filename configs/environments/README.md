# Environment Configuration Schema

Environment JSON files define the 3D grid layout for HVAC simulations. Place configs in this directory and reference them with `--config`.

## Primary Config

**`university_floor.json`** — 5-room university building floor with corridors, VAV dampers, occupancy-driven heat sources, and shared cooling plant.

## Schema

### `grid` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nx` | int | — | Grid cells in X |
| `ny` | int | — | Grid cells in Y |
| `nz` | int | — | Grid cells in Z (height) |
| `dx` | float | 1.0 | Cell spacing in meters |

### `physics` (required)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `thermal_diffusivity` | float | — | alpha in m^2/s (typical indoor air: ~0.02) |
| `dt` | float/null | null | Time step in seconds. If null, auto-computed from CFL |
| `safety_factor` | float | 0.4 | Fraction of max stable dt (only used when dt is null) |
| `ambient_temperature` | float | 20.0 | Initial temperature for all cells |

### `rooms` (required)

Array of rectangular thermal zones. Each room has a comfort setpoint.

```json
{
  "name": "classroom_101",
  "bounds": {
    "x_min": 1, "x_max": 14,
    "y_min": 1, "y_max": 12,
    "z_min": 0, "z_max": 3
  },
  "setpoint": 22.0
}
```

### `hallways` (optional)

Open corridors connecting rooms. No setpoint — just open space.

```json
{
  "name": "main_corridor",
  "bounds": { "x_min": 14, "x_max": 17, "y_min": 1, "y_max": 29, "z_min": 0, "z_max": 3 }
}
```

### `vav_dampers` (required for HVAC)

Variable Air Volume dampers that control cooling airflow per zone.

```json
{
  "name": "vav_classroom_101",
  "zone": "classroom_101",
  "position": {"x": 7, "y": 6, "z": 2},
  "max_flow": 1.0,
  "initial_opening": 0.1
}
```

- `opening` ranges from 0.0 (closed) to 1.0 (fully open)
- Position must be inside the named zone

### `heat_sources` (optional)

Zone-wide heat injection with optional occupancy profiles.

```json
{
  "name": "occupancy_101",
  "zone": "classroom_101",
  "rate": 0.02,
  "schedule": {"start": 0, "end": null},
  "occupancy_profile": [
    {"time": 0.0, "rate": 0.0},
    {"time": 0.5, "rate": 0.02},
    {"time": 2.0, "rate": 0.0}
  ]
}
```

- `rate`: base heat injection rate (overridden by occupancy_profile if present)
- `occupancy_profile`: time-varying keyframes with step interpolation

### `cooling_plant` (required for HVAC)

```json
{
  "Q_total": 8.0,
  "supply_temperature": 12.0
}
```

- `Q_total`: maximum total cooling capacity (shared across all dampers)
- `supply_temperature`: chilled air temperature

### `sensors` (optional)

Configures sensor placement for the sensor network.

```json
{
  "placement": "grid",
  "spacing": 4,
  "z_levels": [1],
  "communication_radius": 6.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `placement` | string | "grid" | Strategy: `grid`, `random`, or `manual` |
| `spacing` | int | 3 | Grid spacing (grid mode only) |
| `count` | int | 20 | Number of sensors (random mode only) |
| `z_levels` | list[int] | [nz/2] | Z-levels to place sensors on |
| `communication_radius` | float | 5.0 | Max distance for communication edges |

### `noise` (optional)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensor_sigma` | float | 0.0 | Gaussian noise std dev for sensor readings |

### `network` (optional)

Centralized controller delay parameters (used by centralized policy baseline).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `polling_interval` | float | 5.0 | How often the central controller polls sensors |
| `jitter_sigma` | float | 0.5 | Random polling delay std dev |
| `compute_delay` | float | 1.0 | Central compute time per poll cycle |

## Creating a New Environment

1. Copy `university_floor.json` as a starting point
2. Define rooms with setpoints and hallways connecting them
3. Place VAV dampers inside rooms
4. Add heat sources with occupancy profiles
5. Configure sensor placement and cooling plant
6. Verify: `python -m blocks.world.demo --config configs/environments/your_config.json`
