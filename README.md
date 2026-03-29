# Aether-Edge: Decentralized Predictive Resource Management

**ECE 659 Project** — A decentralized, predictive building management system demonstrating edge-native resource allocation.

## Overview

Aether-Edge is a general-purpose **rate-of-change monitoring and prediction network** applied to HVAC building management. Sensor nodes monitor environmental metrics (temperature, but the architecture generalizes to CO2, humidity, occupancy, etc.), detect trends via rolling buffer analysis, and predict when comfort thresholds will be breached. The key contribution is that **prediction and actuation happen at the edge** — no central coordinator is needed, and decisions are made with zero data staleness.

The "better than centralized" argument is simple: **network latency kills responsiveness**. A centralized BMS must poll sensors, transmit data, compute a decision, and push commands back — all while the environment changes. Edge nodes act on fresh local data immediately. The Age of Information (AoI) difference is the core metric.

### Architecture: Interface + Sensor Network

The system enforces a clean separation between **what interacts with the environment** and **what does inference**:

**Interface Block** (`blocks/interface/`) — Environment I/O boundary
- **Input**: Reads `World.T` at sensor positions → feeds scalar `(timestamp, value)` pairs to the sensor network
- **Output**: Reads urgency from sensor network → routes fixed airflow budget → actuates vents
- This is the only code that touches `World.T` or damper controls

**Sensor Network Block** (`blocks/sensor_network/`) — Pure domain-agnostic inference
- Receives scalar values — knows nothing about temperature, HVAC, or vents
- Three inference layers:
  1. **Sensing**: Rolling buffer + least-squares slope estimation (dV/dt)
  2. **Prediction**: TTI = (threshold - current) / (rate of change) → urgency = 1/TTI
  3. **Communication**: Gossip protocol propagates urgency to neighbors
- Swap the interface to read CO2 or humidity — the sensor network works identically

**Vent Routing**: With fixed total airflow (`Q_total`), the interface redistributes cooling from empty/cool rooms to rooms with high urgency. Edge policy makes this decision locally (AoI=0); centralized baseline polls with network delay (AoI>0).

## Quick Start

```bash
conda activate ece659
conda install -n ece659 numpy pytest matplotlib networkx jsonschema pyvista -c conda-forge

# Run all tests
python -m pytest blocks/ -v

# Run the benchmark (edge vs centralized on university building)
python -m blocks.benchmark.demo --config configs/environments/university_floor.json
```

## Report Section Mapping

This section maps each component of the ECE 659 report to the specific files and code that implement it.

---

### 1. Introduction to the Problem

**The problem**: Centralized Building Management Systems (BMS) suffer from network-induced data staleness. When a central controller polls sensors, computes decisions, and pushes commands, the environment has already changed. In multi-zone buildings with shared cooling capacity, this delay causes comfort violations, wasted energy, and suboptimal resource allocation.

**Where it's demonstrated in code**:
- The centralized policy in [blocks/interface/interface.py](blocks/interface/interface.py) explicitly models polling delay, network jitter, and compute delay. Between polls, the controller uses *cached* (stale) data.
- The `age_of_information` field in each `VentAction` records exactly how stale the data was for every decision.
- The benchmark comparison in [blocks/benchmark/benchmark.py](blocks/benchmark/benchmark.py) directly compares AoI between the two approaches.

---

### 2. Motivation

**Why edge?**: The insight is that sensor networks can do more than just report readings — they can *predict* and *act locally*. If a node can estimate "this room will breach its setpoint in 30 seconds," it doesn't need to wait for a central server to figure that out. The prediction is derived from **rate of change**, which is a local computation.

**Why this generalizes**: The sensing layer (`RollingBuffer` + slope estimation) is not HVAC-specific. It monitors rate of change of any metric. You could swap temperature for CO2, humidity, occupancy density, or power draw — the prediction and gossip layers work the same way.

**Where it's demonstrated in code**:
- [blocks/sensor_network/node.py](blocks/sensor_network/node.py) — `RollingBuffer` class (least-squares slope estimation), `SensorNode.tti` property (threshold prediction), `SensorNode.urgency` property (actionable signal)
- The `talk_threshold` parameter controls bandwidth: nodes only gossip when they detect a meaningful rate of change (|dT/dt| > threshold)
- [blocks/sensor_network/gossip.py](blocks/sensor_network/gossip.py) — `NegotiationMessage` carries urgency, temperature, and rate of change — not raw sensor dumps

---

### 3. Goal and Objectives

**Goal**: Demonstrate that decentralized edge-native actuation with gossip-propagated predictions achieves comparable or better comfort outcomes with zero Age of Information.

**Objectives**:
1. Build a 3D thermal simulation environment with multi-zone HVAC physics
2. Implement a general-purpose rate-of-change monitoring and prediction network
3. Design a gossip-based consensus protocol for distributed resource allocation
4. Compare edge vs centralized approaches on comfort, energy, AoI, and message overhead

**Where each objective maps**:

| Objective | Primary Code | Tests |
|-----------|-------------|-------|
| 3D thermal simulation | [blocks/world/world.py](blocks/world/world.py) — FTCS heat diffusion + cooling | [blocks/world/test_world.py](blocks/world/test_world.py) (45 tests) |
| Rate-of-change monitoring | [blocks/sensor_network/node.py](blocks/sensor_network/node.py) — `RollingBuffer`, `slope()`, `dT_dt` | [blocks/sensor_network/test_sensor_network.py](blocks/sensor_network/test_sensor_network.py) |
| Prediction (TTI) | [blocks/sensor_network/node.py](blocks/sensor_network/node.py) — `tti`, `urgency` properties | Same test file, `TestSensorNodeTTI` class |
| Gossip consensus | [blocks/sensor_network/sensor_field.py](blocks/sensor_network/sensor_field.py) — `_run_gossip()` + [blocks/sensor_network/gossip.py](blocks/sensor_network/gossip.py) | Same test file, `TestSensorNodeGossip` class |
| Environment I/O (interface) | [blocks/interface/interface.py](blocks/interface/interface.py) — `read_sensors()`, `step()` | [blocks/interface/test_interface.py](blocks/interface/test_interface.py) (14 tests) |
| Vent routing | [blocks/interface/interface.py](blocks/interface/interface.py) — `_evaluate_edge()`, `_evaluate_centralized()` | Same test file, `TestEdgePolicy` / `TestCentralizedPolicy` classes |
| Edge vs centralized comparison | [blocks/benchmark/benchmark.py](blocks/benchmark/benchmark.py) — `compare()` | [blocks/benchmark/test_benchmark.py](blocks/benchmark/test_benchmark.py) (7 tests) |

---

### 4. Methodology: Design and Implementation

#### 4a. Physical Environment (Block 1)

The simulation models a university building floor with 5 thermal zones connected by corridors. The physics engine solves 3D heat diffusion:

```
dT/dt = alpha * nabla^2(T) + Q_heat - Q_cool
```

- **Files**: [blocks/world/environment.py](blocks/world/environment.py) (config parser), [blocks/world/world.py](blocks/world/world.py) (physics engine), [blocks/world/stability.py](blocks/world/stability.py) (CFL stability)
- **Config**: [configs/environments/university_floor.json](configs/environments/university_floor.json) — 5-room university floor (2 classrooms, computer lab, faculty office, study lounge)
- **Key design**: Environments are JSON-driven. The same code handles any room layout, heat profile, and damper placement.

#### 4b. Sensor Network — Topology + Inference + Gossip

The sensor network block (`blocks/sensor_network/`) combines mesh topology with the inference engine:

- **Topology**: Sensors placed on a grid, connected by a communication graph (edges within radio range)
- **Files**: [blocks/sensor_network/sensor_network.py](blocks/sensor_network/sensor_network.py), [blocks/sensor_network/placement.py](blocks/sensor_network/placement.py)
- **Metrics**: Connectivity, coverage, degree distribution, clustering coefficient

#### 4c. Sensing — Rate-of-Change Monitoring (Layer 1)

Each sensor node maintains a **rolling buffer** of the last 30 seconds of readings. The least-squares slope of this buffer gives `dT/dt` — the rate of change. This is the foundational signal for everything that follows.

- **File**: [blocks/sensor_network/node.py](blocks/sensor_network/node.py) — `RollingBuffer` class, `SensorNode.sense()` method
- **Key property**: `node.dT_dt` — the estimated rate of change at this sensor position
- **Key insight**: This layer is domain-agnostic. Replace temperature with CO2 or humidity and the buffer + slope logic is identical.

#### 4d. Prediction — Time-To-Impact (Layer 2)

TTI answers: "At the current rate, when will this zone breach its comfort setpoint?"

```
TTI = (T_setpoint - T_current) / (dT/dt)    when dT/dt > 0
TTI = infinity                                when stable or cooling
```

Urgency is the inverse: `urgency = 1/TTI`. Higher urgency = more imminent breach.

- **File**: [blocks/sensor_network/node.py](blocks/sensor_network/node.py) — `SensorNode.tti` property, `SensorNode.urgency` property
- **Gossip**: Urgency propagates to neighbors via `NegotiationMessage` ([blocks/sensor_network/gossip.py](blocks/sensor_network/gossip.py)), so each node knows the urgency landscape of its neighborhood.

#### 4e. Gossip Protocol (propagation)

Multi-round gossip spreads urgency information through the sensor mesh:
- Round 0: Nodes with |dT/dt| > talk_threshold generate messages
- Rounds 1+: Messages are forwarded one hop per round (up to max_hops)
- Result: Each node knows the urgency of its neighbors, enabling local resource allocation decisions

- **File**: [blocks/sensor_network/sensor_field.py](blocks/sensor_network/sensor_field.py) — `_run_gossip()` method
- **Bandwidth control**: `talk_threshold` prevents unnecessary chatter when conditions are stable

#### 4f. Environment Interface — I/O Boundary (Block 5)

The interface block is the **only** component that touches both the physical environment (World) and the inference network (SensorField). It enforces a clean separation:

- **Input side**: Reads `World.T` at sensor positions → feeds scalar `(timestamp, value)` pairs to the SensorField. The sensor network never sees the temperature array directly.
- **Output side**: Reads urgency from the SensorField → translates to vent routing commands → applies openings to World dampers.

This separation means the sensor network is genuinely domain-agnostic — you could swap the interface to read CO2 or humidity without changing a single line in the sensor block.

**Vent routing with fixed airflow budget**: With a fixed `Q_total`, the interface redistributes airflow from low-urgency rooms (empty/cool) to high-urgency rooms (occupied/heating). Closing vents to empty rooms saves energy while maintaining comfort in active rooms.

**Edge policy** (Aether-Edge):
- Each vent's opening = `urgency_i / total_urgency`
- All decisions are local — **AoI = 0**

**Centralized policy** (Baseline):
- Central controller polls sensors at `polling_interval` (+ random jitter)
- Same allocation, but with stale data
- **AoI = polling_interval + jitter + compute_delay** (~20 seconds, measured at 19.4s on university floor)

- **File**: [blocks/interface/interface.py](blocks/interface/interface.py)
- **Read sensors**: `read_sensors()` — extracts scalar values from World
- **Full cycle**: `step()` — read → feed sensor network → route airflow → actuate
- **Edge**: `_evaluate_edge()`
- **Centralized**: `_evaluate_centralized()`

#### 4g. Simulation Loop (Block 6)

Each timestep: `world.step()` -> `interface.step(world)`

The interface call handles the full cycle: read environment → feed sensor network → get urgency → route airflow → actuate vents. The sensor network never sees the World object.

The simulation accumulates comfort violation (time-integrated overshoot) and energy usage. Both edge and centralized modes use the same loop — the difference is entirely in the interface's data freshness.

- **File**: [blocks/simulation/simulation.py](blocks/simulation/simulation.py)

---

### 5. Experimental Work

#### 5a. Experimental Setup

The benchmark environment is a **university building floor** with 5 zones:

| Zone | Setpoint | Peak Heat Rate | Scenario |
|------|----------|----------------|----------|
| Classroom 101 | 22°C | 0.015 (occupancy profile) | Morning lecture |
| Classroom 102 | 22°C | 0.014 (occupancy profile) | Afternoon lecture |
| Computer Lab | 20°C | 0.008 (always-on) + 0.010 (occupancy) | Always-on servers |
| Conference Room | 22°C | 0.012 (meeting profile) | Scheduled meetings |
| Faculty Office | 22°C | 0.002 (light occupancy) | Quiet work |

- **Config**: [configs/environments/university_floor.json](configs/environments/university_floor.json)
- **Grid**: 46x30x3 voxels, 1m resolution
- **Sensors**: grid spacing=4m, communication radius=6.0m
- **Cooling plant**: Q_total=8.0, T_supply=12°C (shared budget across all 5 VAV dampers)
- **Centralized network**: polling_interval=15s, jitter_sigma=2s, compute_delay=3s

#### 5b. Benchmark Execution

The benchmark runs the same environment with both policies (edge and centralized) from identical initial conditions.

- **File**: [blocks/benchmark/benchmark.py](blocks/benchmark/benchmark.py)
- **Entry point**: `Benchmark.run()` calls `run_edge()` and `run_centralized()`, then `compare()`
- **Config**: [configs/experiments/benchmark_default.json](configs/experiments/benchmark_default.json)

#### 5c. Metrics Collected

| Metric | Code Location | What It Measures |
|--------|--------------|-----------------|
| Comfort violation | [simulation.py](blocks/simulation/simulation.py) `_comfort_violation_integral` | Time-integrated overshoot (lower = better comfort) |
| Cumulative energy | [simulation.py](blocks/simulation/simulation.py) `_energy_integral` | Total cooling energy consumed |
| Age of Information | [interface.py](blocks/interface/interface.py) `mean_age_of_information` | Average data staleness at decision time |
| Max overshoot | [world.py](blocks/world/world.py) `max_overshoot()` | Peak degrees above setpoint |
| Zone temperatures | [world.py](blocks/world/world.py) `zone_mean_temperature()` | Per-zone thermal state |
| Gossip messages | [interface.py](blocks/interface/interface.py) `total_messages` | Communication overhead |
| Comfort improvement % | [benchmark.py](blocks/benchmark/benchmark.py) `compare()` | Edge vs centralized comfort delta |
| Energy savings % | [benchmark.py](blocks/benchmark/benchmark.py) `compare()` | Edge vs centralized energy delta |

#### 5d. Results Output

- Per-policy metric time-series saved as JSON and CSV via `MetricsCollector`
- Comparison summary saved to `results/benchmark/comparison.json`
- **Metrics infrastructure**: [blocks/metrics/collector.py](blocks/metrics/collector.py), [blocks/metrics/experiment.py](blocks/metrics/experiment.py)

#### 5e. Parameter Sweeps

The `ExperimentRunner` supports sweeping any config parameter (sensor density, communication radius, gossip rounds, etc.) across a grid:

- **Config**: [configs/experiments/sweep_sensor_density.json](configs/experiments/sweep_sensor_density.json)
- **Code**: [blocks/metrics/experiment.py](blocks/metrics/experiment.py)

---

### 6. Testing and Validation

190 unit + integration tests verify correctness at every level:

| Test Suite | File | Count | What It Validates |
|-----------|------|-------|-------------------|
| World physics | [blocks/world/test_world.py](blocks/world/test_world.py) | 45 | Stability, diffusion, cooling, zones, occupancy |
| Visualization | [blocks/visualization/test_visualization.py](blocks/visualization/test_visualization.py) | 7 | Mesh construction, volume rendering |
| Sensor network | [blocks/sensor_network/test_sensor_network.py](blocks/sensor_network/test_sensor_network.py) | 51 | Placement, graph, buffer, TTI, urgency, gossip |
| Interface (I/O) | [blocks/interface/test_interface.py](blocks/interface/test_interface.py) | 14 | Read/actuate, edge/centralized policies, AoI |
| Simulation | [blocks/simulation/test_simulation.py](blocks/simulation/test_simulation.py) | 12 | Full loop, metrics recording |
| Benchmark | [blocks/benchmark/test_benchmark.py](blocks/benchmark/test_benchmark.py) | 7 | Comparison, independence |
| Metrics | [blocks/metrics/test_metrics.py](blocks/metrics/test_metrics.py) | 14 | Collection, export, sweeps |
| **Integration** | [blocks/test_integration.py](blocks/test_integration.py) | **40** | All blocks working together |

```bash
python -m pytest blocks/ -v
```

---

## Project Structure

```
gas_prediction_network/
├── README.md                              # This file (report mapping)
├── blocks/
│   ├── test_integration.py                # Cross-block integration tests (40)
│   ├── world/                             # Block 1: 3D thermal physics
│   │   ├── environment.py                 #   JSON config -> grid + zones + dampers
│   │   ├── world.py                       #   FTCS heat diffusion + cooling
│   │   └── stability.py                   #   CFL + diffusion stability
│   ├── visualization/                     # Block 2: 3D PyVista rendering
│   │   └── renderer.py                    #   Temperature volume + damper markers
│   ├── sensor_network/                    # Sensor network: topology + inference + gossip
│   │   ├── sensor_network.py              #   NetworkX graph + coverage metrics
│   │   ├── placement.py                   #   Grid/random/manual sensor placement
│   │   ├── node.py                        #   RollingBuffer, TTI, urgency (3 layers)
│   │   ├── gossip.py                      #   NegotiationMessage dataclass
│   │   └── sensor_field.py                #   Orchestrates sense -> predict -> gossip
│   ├── interface/                         # Environment I/O boundary
│   │   └── interface.py                   #   Read sensors + route airflow + actuate vents
│   ├── simulation/                        # Full integration loop
│   │   └── simulation.py                  #   world -> interface (read→infer→actuate)
│   ├── benchmark/                         # Edge vs centralized comparison
│   │   └── benchmark.py                   #   Run both, compare metrics
│   └── metrics/                           # Infrastructure: collection + sweeps
│       ├── collector.py                   #   MetricsCollector (JSON/CSV export)
│       └── experiment.py                  #   ExperimentRunner (parameter grids)
├── configs/
│   ├── environments/
│   │   ├── university_floor.json          #   Primary: 5-room university building
│   │   ├── hvac_office.json               #   4-room office layout
│   │   └── default_maze.json              #   3-room cross layout
│   └── experiments/
│       ├── benchmark_default.json         #   Default benchmark config
│       └── sweep_sensor_density.json      #   Sensor density parameter sweep
├── results/                               # Generated benchmark outputs
│   └── benchmark/                         #   JSON, CSV, GIFs, comparison chart
└── run_benchmark.py                       # Top-level entry point
```
