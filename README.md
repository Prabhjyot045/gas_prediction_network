# Aether-Edge: Decentralized Predictive HVAC Control

**ECE 659 Project** — A decentralized, predictive building management system for HVAC cooling allocation.

## Overview

Aether-Edge simulates a 3D HVAC environment where sensor nodes **predict** thermal zone breaches and **proactively** allocate cooling resources at the edge — before comfort violations occur. This contrasts with traditional centralized, reactive BMS controllers that suffer from network-induced Age of Information (AoI).

### Novel Component

**Urgency-weighted resource negotiation via gossip consensus.** Each sensor node estimates Time-To-Impact (TTI) from local temperature trends, converts it to urgency (`1/TTI`), then propagates that urgency through the sensor mesh so VAV dampers can proportionally allocate the building's shared cooling budget. No central coordinator required.

## Architecture

The project is built as independent, executable blocks:

| Block | Directory | Description | Status |
|-------|-----------|-------------|--------|
| 1 | `blocks/world/` | 3D heat diffusion + cooling engine | Complete |
| 2 | `blocks/visualization/` | 3D PyVista temperature rendering | Complete |
| 3 | `blocks/network/` | Sensor mesh topology (NetworkX) | Complete |
| 4 | `blocks/sensor/` | Rolling buffer + TTI + urgency gossip | Complete |
| 5 | `blocks/actuator/` | VAV dampers + proportional allocation | Complete |
| 6 | `blocks/simulation/` | Edge vs centralized simulation loop | Complete |
| 7 | `blocks/benchmark/` | Edge vs centralized comparison | Complete |
| - | `blocks/metrics/` | Metrics collection + experiment runner | Complete |

## Quick Start

```bash
# Activate environment
conda activate ece659

# Install dependencies
conda install -n ece659 numpy pytest matplotlib networkx jsonschema pyvista -c conda-forge

# Run all tests (179 tests)
python -m pytest blocks/ -v

# Run Block 1 demo (temperature heatmap)
python -m blocks.world.demo --config configs/environments/hvac_office.json

# Run Block 2 demo (3D PyVista visualization)
python -m blocks.visualization.demo --config configs/environments/hvac_office.json

# Run Block 3 demo (sensor network topology)
python -m blocks.network.demo --config configs/environments/hvac_office.json

# Run Block 4 demo (TTI + urgency gossip)
python -m blocks.sensor.demo --config configs/environments/hvac_office.json

# Run Block 5 demo (VAV damper actuation)
python -m blocks.actuator.demo --config configs/environments/hvac_office.json

# Run Block 6 demo (full simulation)
python -m blocks.simulation.demo --config configs/environments/hvac_office.json

# Run Block 7 demo (edge vs centralized benchmark)
python -m blocks.benchmark.demo --config configs/environments/hvac_office.json
```

## Environment Configuration

Environments are defined as JSON files in `configs/environments/`. This allows creating arbitrary room/hallway layouts, VAV damper placements, heat source schedules, and sensor configurations without code changes.

```json
{
  "grid": { "nx": 30, "ny": 30, "nz": 3, "dx": 1.0 },
  "physics": { "thermal_diffusivity": 0.02, "ambient_temperature": 20.0 },
  "rooms": [
    { "name": "office_A", "bounds": {...}, "setpoint": 22.0 }
  ],
  "hallways": [
    { "name": "corridor", "bounds": {...} }
  ],
  "vav_dampers": [
    { "name": "vav_A", "zone": "office_A", "position": {...}, "max_flow": 1.0, "initial_opening": 0.5 }
  ],
  "heat_sources": [
    { "name": "occupancy", "zone": "office_A", "rate": 0.5, "schedule": {"start": 0, "end": null} }
  ],
  "cooling_plant": { "Q_total": 5.0, "supply_temperature": 12.0 },
  "sensors": { "placement": "grid", "spacing": 3, "z_levels": [1], "communication_radius": 5.0 },
  "noise": { "sensor_sigma": 0.05 },
  "network": { "polling_interval": 5.0, "jitter_sigma": 0.5, "compute_delay": 1.0 }
}
```

## Physics

The core engine solves 3D heat diffusion using Forward-Time Central-Space (FTCS) finite differences:

```
T_new = T + alpha * dt/dx^2 * laplacian(T) + Q_heat - Q_cool
```

- **Thermal diffusivity**: `alpha ~ 0.02 m^2/s` for air
- **Stability constraint**: `dt <= dx^2 / (6 * alpha)` (auto-enforced)
- **Boundary conditions**: Neumann zero-flux at walls (insulated)
- **Heat sources**: Distributed uniformly across thermal zones (occupancy, equipment, solar)
- **Cooling model**: `T -= flow * (T - T_supply) * dt` at VAV damper positions
- **Resource constraint**: `sum(Q_cool_i) <= Q_total` (plant budget)

## Sensing and Prediction

Each sensor node maintains a **rolling buffer** of temperature readings and computes:

- **dT/dt**: Least-squares slope from buffer (more robust than finite difference)
- **Time-To-Impact (TTI)**: `(T_setpoint - T_current) / (dT/dt)` — predicts when setpoint will be breached
- **Urgency**: `1/TTI` — higher values mean more urgent need for cooling
- **Communication threshold**: Only gossip if `|dT/dt| > talk_threshold` (bandwidth savings)

## Actuation Policies

### Edge (Aether-Edge) — Decentralized
1. Each zone computes local urgency from TTI
2. Gossip: zones share urgency with neighbors over multiple rounds
3. After consensus, each zone knows neighborhood urgencies
4. Proportional allocation: `A_i = u_i / sum(u_j)` (urgency-weighted share of Q_total)
5. **Age of Information = 0** (all decisions are local)

### Centralized — Baseline
1. Central controller polls all sensors (adds configurable network delay)
2. Same proportional allocation formula, but with **stale data**
3. **Age of Information > 0** due to: polling_interval + jitter + compute_delay

## Benchmark Metrics

| Metric | Definition | Lower = Better? |
|--------|-----------|-----------------|
| Overshoot Error | max(T - T_setpoint) across all zones | Yes |
| Comfort Violation | Time-integrated temperature exceedance | Yes |
| Energy Usage | Total cooling energy over all steps | Yes |
| Age of Information | Average staleness of sensor data | Yes |
| Packet Overhead | Total gossip messages sent | Yes |

## Integration

Blocks share a single `Environment` instance as their interface. Integration tests in [blocks/test_integration.py](blocks/test_integration.py) verify:

- Shared `Environment` object identity across all blocks
- Temperature field integrity between World and Renderer
- Sensor positions validated against wall mask
- SensorField reads temperatures from World, computes TTI/urgency
- DamperController allocates cooling based on urgency consensus
- Simulation loop: World -> SensorField -> Actuator (no block drives another)
- Benchmark runs independent edge/centralized simulations for fair comparison
- `MetricsCollector` accumulates from all blocks
- Decoupling: Renderer never drives simulation, Network topology is static

```bash
# Run integration tests specifically
python -m pytest blocks/test_integration.py -v
```

## Project Structure

```
gas_prediction_network/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE
├── results/                               # Experiment outputs (gitignored)
├── blocks/
│   ├── __init__.py
│   ├── test_integration.py                # Integration tests (38 tests)
│   ├── world/                             # Block 1: Physics engine
│   │   ├── stability.py                   # FTCS + CFL stability constraints
│   │   ├── environment.py                 # JSON config loader (HVAC schema)
│   │   ├── world.py                       # 3D heat diffusion + cooling engine
│   │   ├── demo.py                        # Temperature heatmap demo
│   │   ├── test_world.py                  # Unit tests (37 tests)
│   │   └── README.md
│   ├── visualization/                     # Block 2: 3D rendering
│   │   ├── renderer.py                    # PyVista temperature renderer
│   │   ├── demo.py                        # Visualization demo
│   │   ├── test_visualization.py          # Unit tests (7 tests)
│   │   └── README.md
│   ├── network/                           # Block 3: Sensor mesh
│   │   ├── placement.py                   # Sensor placement strategies
│   │   ├── sensor_network.py              # NetworkX graph + topology metrics
│   │   ├── demo.py                        # Topology visualization demo
│   │   ├── test_network.py                # Unit tests (24 tests)
│   │   └── README.md
│   ├── sensor/                            # Block 4: TTI + urgency gossip
│   │   ├── node.py                        # SensorNode with rolling buffer + TTI
│   │   ├── gossip.py                      # NegotiationMessage dataclass
│   │   ├── sensor_field.py                # SensorField manager
│   │   ├── demo.py                        # TTI + gossip visualization demo
│   │   ├── test_sensor.py                 # Unit tests (29 tests)
│   │   └── README.md
│   ├── actuator/                          # Block 5: VAV damper control
│   │   ├── controller.py                  # DamperController (edge/centralized)
│   │   ├── demo.py                        # Actuation visualization demo
│   │   ├── test_actuator.py               # Unit tests (11 tests)
│   │   └── README.md
│   ├── simulation/                        # Block 6: Full integration
│   │   ├── simulation.py                  # Simulation loop (edge/centralized)
│   │   ├── demo.py                        # Full simulation demo
│   │   ├── test_simulation.py             # Unit tests (12 tests)
│   │   └── README.md
│   ├── benchmark/                         # Block 7: Edge vs centralized
│   │   ├── benchmark.py                   # Benchmark comparison runner
│   │   ├── demo.py                        # Side-by-side comparison demo
│   │   ├── test_benchmark.py              # Unit tests (7 tests)
│   │   └── README.md
│   └── metrics/                           # Metrics infrastructure
│       ├── collector.py                   # MetricsCollector
│       ├── experiment.py                  # ExperimentRunner (parameter sweeps)
│       ├── test_metrics.py                # Unit tests (14 tests)
│       └── README.md
├── configs/
│   ├── environments/
│   │   ├── README.md                      # JSON schema documentation
│   │   ├── default_maze.json              # Default 20x20x3 cross layout
│   │   └── hvac_office.json               # 30x30x3 four-room office
│   └── experiments/
│       ├── sweep_sensor_density.json      # Parameter sweep config
│       └── benchmark_default.json         # Default benchmark config
└── deprecated/
    └── main.py                            # Original 2D LBM prototype
```
