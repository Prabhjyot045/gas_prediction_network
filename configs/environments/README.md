# Environment Configuration Schema

Environment JSON files define the 3D grid layout for HVAC simulations. Place configs in this directory and reference them with `--config`.

## Environments

Three environment configurations are provided, each representing a different building type and thermal challenge level. Together they test the system across a range of grid sizes, zone counts, heat load intensities, and occupancy patterns.

### At a Glance

| Property | HVAC Office | University Floor | Airport Terminal |
|---|---|---|---|
| **File** | `hvac_office.json` | `university_floor.json` | `airport.json` |
| **Grid** | 30 × 30 × 3 | 46 × 30 × 3 | 60 × 36 × 3 |
| **Zones** | 4 | 5 | 6 |
| **Hallways** | Cross corridor | 2 corridors + 7 doors | None |
| **α (m²/s)** | 0.02 | 0.02 | 0.025 |
| **Q_total** | 5.0 | 8.0 | 12.0 |
| **Sensor spacing** | 3 m | 4 m | 5 m |
| **Comm radius** | 5.0 m | 6.0 m | 7.0 m |
| **Peak heat rate** | 1.0 (server room) | 0.015 (occupancy) | 0.025 (occupancy + solar) |
| **Heat pattern** | Constant | Time-varying profiles | Time-varying + solar ramps |
| **Difficulty** | High (plant undersized) | Moderate (balanced) | High (bursty + large grid) |

---

### 1. HVAC Office (`hvac_office.json`) — Steady-State Stress Test

A compact 4-room office with a cross-shaped corridor.  The defining feature is a **server room** with continuous high heat output (rate = 1.0) and a low setpoint (18°C), competing for cooling with three standard offices (rate = 0.15–0.30, setpoint = 22°C).

**What it tests:**
- **Resource contention under fixed loads.** The server room alone demands most of the Q_total = 5.0 budget.  The system must learn to starve low-urgency offices in favor of the constantly overheating server room.
- **Steady-state allocation.** All heat sources are constant (`schedule: start=0, end=null`), so there are no occupancy transitions to detect.  This isolates the system's ability to find and maintain a stable proportional allocation.
- **Heterogeneous setpoints.** The server room's 18°C setpoint vs. 22°C offices forces the gossip protocol to negotiate across zones with fundamentally different thermal targets.
- **Dense sensor mesh.** 3 m spacing on a 30 × 30 grid produces a well-connected mesh (~40 nodes), testing gossip convergence on a small, high-connectivity topology.

| Zone | Setpoint | Heat Rate | Pattern |
|---|---|---|---|
| Office A | 22°C | 0.30 | Constant |
| Office B | 22°C | 0.20 | Constant |
| Office C | 22°C | 0.15 | Constant |
| Server Room | 18°C | 1.00 | Constant |

---

### 2. University Floor (`university_floor.json`) — Dynamic Occupancy Benchmark

The primary benchmark environment.  Five zones on a university building floor connected by corridors and doors, with time-varying occupancy profiles simulating a typical academic day (morning lectures, afternoon labs, scheduled meetings).

**What it tests:**
- **TTI prediction under occupancy transitions.** Heat loads ramp up and down as classes start and end.  The edge policy must detect these ramps via rolling-buffer slope estimation (dT/dt) and preemptively increase cooling *before* the setpoint is breached.
- **Temporal heterogeneity.** Different zones peak at different times: Classroom 101 heats up during morning lectures (t=200), the Computer Lab has an always-on server baseline plus occupancy surges (t=150), and the Conference Room peaks during scheduled meetings (t=400).  This tests the system's ability to dynamically redistribute cooling across zones as thermal priorities shift.
- **Corridor heat diffusion.** Two main corridors connect all zones, allowing heat to leak between rooms.  The system must account for inter-zone thermal coupling.
- **Balanced plant capacity.** Q_total = 8.0 is sufficient to handle all heat loads simultaneously.  This makes it a fair comparison: the edge vs. centralized performance gap is due to data freshness (AoI), not plant saturation.

| Zone | Setpoint | Peak Heat Rate | Pattern |
|---|---|---|---|
| Classroom 101 | 22°C | 0.015 | Morning lecture ramp (t=200–600) |
| Classroom 102 | 22°C | 0.014 | Afternoon lecture ramp (t=250–800, t=1200–1800) |
| Computer Lab | 20°C | 0.008 + 0.010 | Always-on servers + occupancy surge (t=150–600) |
| Conference Room | 22°C | 0.012 | Meeting schedule (t=400–1000) |
| Faculty Office | 22°C | 0.002 | Light, steady occupancy |

---

### 3. Airport Terminal (`airport.json`) — Large-Scale Stress Test

A large 6-gate airport terminal with high, bursty passenger occupancy surges and solar heat gain through terminal windows.  No hallways — gates are open-plan zones sharing the full terminal footprint.

**What it tests:**
- **Scalability.** The 60 × 36 grid is 2.5× larger than the HVAC office.  More voxels means longer diffusion times, more sensor nodes (5 m spacing on a larger grid), and more gossip messages required for convergence.
- **Bursty, extreme heat loads.** Gates 3–5 experience peak occupancy rates of 0.018–0.025 (boarding surges), which is 10–15× higher than the university's occupancy rates.  Combined with solar gain profiles that ramp over time, these loads push both policies beyond the plant's comfort-recovery capacity — **both will breach setpoints**.
- **Plant saturation.** Even with Q_total = 12.0, the total heat load during peak boarding can exceed what the VAV dampers can compensate.  This tests behavior under thermal saturation: edge can't prevent all overshoot, but should still reduce *cumulative* violation.
- **Message efficiency at scale.** With 6 zones and longer stable periods between boarding surges, the talk_threshold can suppress gossip for extended durations.  This tests whether the event-triggered communication strategy scales well.
- **Solar gain.** Gates 1 and 2 have additional solar heat profiles that ramp linearly over time, simulating sun exposure through terminal windows — a heat source that is independent of occupancy.

| Zone | Setpoint | Peak Heat Rate | Pattern |
|---|---|---|---|
| Gate 1 | 22°C | 0.004 + solar | Moderate occupancy surges + solar ramp |
| Gate 2 | 22°C | 0.007 + solar | Higher occupancy peaks + solar ramp |
| Gate 3 | 22°C | 0.021 | Heavy boarding surges (t=850–1600) |
| Gate 4 | 22°C | 0.025 | Heaviest load — major boarding event |
| Gate 5 | 22°C | 0.023 | Sustained high occupancy (t=150–1400) |
| Gate 6 | 22°C | 0.006 | Light, low-priority gate |

---

### How They Compare as Simulation Scenarios

```
Difficulty:   HVAC Office ━━━━━━━━━━━━━━━┓
                                          ┣━ High (different reasons)
              Airport Terminal ━━━━━━━━━━━┛
              University Floor ━━━━━━━━━━━━━ Moderate (balanced)
```

- **HVAC Office** is hard because the plant is **undersized** relative to the server room's constant heat output.  The system is always in a state of resource contention.  This tests *steady-state* allocation under scarcity.

- **University Floor** is moderate because the plant has **sufficient capacity**.  The challenge is *temporal* — detecting occupancy transitions fast enough to preempt comfort violations.  This is the fairest environment for measuring the AoI advantage.

- **Airport Terminal** is hard because heat loads are both **extreme** (10–15× university rates) and **bursty** (boarding surges hit suddenly).  The plant saturates during peaks, so the system's performance is bounded by capacity, not just data freshness.  This tests *graceful degradation* under overload.

---

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
