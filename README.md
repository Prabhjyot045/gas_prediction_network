# Vanguard Distributed Predictive Actuation (VDPA)

**ECE 659 Project** — A decentralized, predictive gas-flow control system.

## Overview

VDPA simulates a 3D sensor mesh that **predicts** the trajectory of gas diffusion and **proactively** triggers actuators (doors/vents) at the edge — before contamination arrives. This contrasts with traditional centralized, reactive safety systems.

### Novel Component

**Distributed predictive actuation via gossip-propagated gradient fields.** Each sensor node estimates gas flow velocity from local gradients, then propagates that prediction forward through the network so distant nodes can actuate before the gas reaches them. No central coordinator required.

## Architecture

The project is built as independent, executable blocks:

| Block | Directory | Description | Status |
|-------|-----------|-------------|--------|
| 1 | `blocks/world/` | 3D FTCS diffusion engine | Complete |
| 2 | `blocks/visualization/` | 3D PyVista rendering | Complete |
| 3 | `blocks/network/` | Sensor mesh topology (NetworkX) | Complete |
| 4 | `blocks/sensor/` | Event-driven edge nodes + Kalman + gossip | Complete |
| 5 | `blocks/actuator/` | Doors + trigger logic (predictive/reactive) | Complete |
| 6 | `blocks/simulation/` | Full integration loop | Complete |
| 7 | `blocks/benchmark/` | Centralized vs VDPA comparison | Complete |
| - | `blocks/metrics/` | Metrics collection + experiment runner | Complete |

## Quick Start

```bash
# Activate environment
conda activate ece659

# Install dependencies
conda install -n ece659 numpy pytest matplotlib networkx jsonschema pyvista filterpy -c conda-forge

# Run Block 1 demo (2D slice diffusion)
python -m blocks.world.demo --config configs/environments/default_maze.json

# Run Block 2 demo (3D PyVista visualization)
python -m blocks.visualization.demo --config configs/environments/default_maze.json

# Run Block 3 demo (sensor network topology)
python -m blocks.network.demo --config configs/environments/default_maze.json

# Run Block 4 demo (Kalman filter + gossip protocol)
python -m blocks.sensor.demo --config configs/environments/default_maze.json

# Run Block 5 demo (actuator with predictive or reactive policy)
python -m blocks.actuator.demo --config configs/environments/default_maze.json

# Run Block 6 demo (full simulation)
python -m blocks.simulation.demo --config configs/environments/default_maze.json

# Run Block 7 demo (VDPA vs centralized benchmark)
python -m blocks.benchmark.demo --config configs/environments/default_maze.json

# Run all tests (182 tests)
python -m pytest blocks/ -v
```

## Environment Configuration

Environments are defined as JSON files in `configs/environments/`. This allows creating arbitrary room/hallway layouts, sensor placements, and noise levels without code changes. See [configs/environments/README.md](configs/environments/README.md) for the full schema.

## Physics

The core engine solves the 3D diffusion equation using Forward-Time Central-Space (FTCS) finite differences:

```
phi_new = phi + D * dt/dx^2 * laplacian(phi)
```

- **Stability constraint**: `dt <= dx^2 / (6 * D)` (auto-enforced)
- **Boundary conditions**: Neumann zero-flux at walls (impermeable, gas reflects)
- **Door actuation**: Hard boundary — closing a door zeroes both the diffusion path and existing concentration in the door cells

## Metrics and Experiments

Every block exposes a `metrics() -> dict` method. The `MetricsCollector` accumulates these over time and exports to JSON/CSV. The `ExperimentRunner` drives parameter sweeps from JSON configs:

```bash
# Example: sweep sensor density and communication radius
# See configs/experiments/sweep_sensor_density.json
```

See [blocks/metrics/README.md](blocks/metrics/README.md) for full details.

## Integration

Blocks share a single `Environment` instance as their interface. Integration tests in [blocks/test_integration.py](blocks/test_integration.py) verify:

- Shared `Environment` object identity across all 7 blocks
- Door state changes propagate from World to Renderer's wall mesh
- Volume data exactly matches `World.phi` after stepping
- Sensor positions validated against wall mask
- SensorField reads from World, uses Network topology for gossip
- ActuatorController closes doors via World based on sensor predictions
- Simulation loop: World → SensorField → Actuator (no block drives another)
- Benchmark runs independent predictive/reactive simulations for fair comparison
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
│   ├── test_integration.py                # Integration tests (42 tests)
│   ├── world/                             # Block 1: Physics engine
│   │   ├── stability.py                   # FTCS stability constraint
│   │   ├── environment.py                 # JSON config loader
│   │   ├── world.py                       # 3D diffusion engine
│   │   ├── demo.py                        # Standalone runnable demo
│   │   ├── test_world.py                  # Unit tests (23 tests)
│   │   └── README.md
│   ├── visualization/                     # Block 2: 3D rendering
│   │   ├── renderer.py                    # PyVista volumetric renderer
│   │   ├── demo.py                        # Standalone visualization demo
│   │   ├── test_visualization.py          # Unit tests (8 tests)
│   │   └── README.md
│   ├── network/                           # Block 3: Sensor mesh
│   │   ├── placement.py                   # Sensor placement strategies
│   │   ├── sensor_network.py              # NetworkX graph + topology metrics
│   │   ├── demo.py                        # Topology visualization demo
│   │   ├── test_network.py                # Unit tests (24 tests)
│   │   └── README.md
│   ├── sensor/                            # Block 4: Kalman + gossip
│   │   ├── node.py                        # SensorNode with Kalman filter
│   │   ├── gossip.py                      # GossipMessage dataclass
│   │   ├── sensor_field.py                # SensorField manager
│   │   ├── demo.py                        # Kalman + gossip visualization demo
│   │   ├── test_sensor.py                 # Unit tests (36 tests)
│   │   └── README.md
│   ├── actuator/                          # Block 5: Door actuation
│   │   ├── controller.py                  # ActuatorController (predictive/reactive)
│   │   ├── demo.py                        # Actuation visualization demo
│   │   ├── test_actuator.py               # Unit tests (13 tests)
│   │   └── README.md
│   ├── simulation/                        # Block 6: Full integration
│   │   ├── simulation.py                  # Simulation loop
│   │   ├── demo.py                        # Full simulation demo
│   │   ├── test_simulation.py             # Unit tests (14 tests)
│   │   └── README.md
│   ├── benchmark/                         # Block 7: VDPA vs centralized
│   │   ├── benchmark.py                   # Benchmark comparison runner
│   │   ├── demo.py                        # Side-by-side comparison demo
│   │   ├── test_benchmark.py              # Unit tests (8 tests)
│   │   └── README.md
│   └── metrics/                           # Metrics infrastructure
│       ├── collector.py                   # MetricsCollector
│       ├── experiment.py                  # ExperimentRunner (parameter sweeps)
│       ├── test_metrics.py                # Unit tests (14 tests)
│       └── README.md
├── configs/
│   ├── environments/
│   │   ├── README.md                      # JSON schema documentation
│   │   └── default_maze.json              # Default 20x20x5 cross-shaped maze
│   └── experiments/
│       ├── sweep_sensor_density.json      # Parameter sweep config
│       └── benchmark_default.json         # Default benchmark config
└── deprecated/
    └── main.py                            # Original 2D LBM prototype
```
