# Block 4: Sensor — Kalman Filter + Gossip Protocol

## Overview

Event-driven sensor nodes that read concentration from the World, apply Kalman filtering for noise reduction, estimate gas flow velocity from gradients, and propagate predictions through the sensor network via a gossip protocol. This is the core novel component of VDPA: **distributed predictive actuation via gossip-propagated gradient fields**.

## How to Run

```bash
conda activate ece659

# Demo: Kalman filter + gossip visualization
python -m blocks.sensor.demo --config configs/environments/default_maze.json

# More simulation steps
python -m blocks.sensor.demo --config configs/environments/default_maze.json --steps 300

# Multi-hop gossip (faster prediction propagation)
python -m blocks.sensor.demo --config configs/environments/default_maze.json --gossip-rounds 3

# Run tests
python -m pytest blocks/sensor/test_sensor.py -v
```

## Architecture

### SensorNode

Each node maintains:
- **Kalman filter** (FilterPy) with state `[φ, dφ/dt]` for noise-corrected concentration tracking
- **Spatial gradient** `∇φ` via central differences on the world's concentration field
- **Flow velocity estimate** `v̂ = -(dφ/dt) / |∇φ|² × ∇φ` — the apparent velocity of the gas front
- **Prediction table** mapping gossip origins to predicted arrival times

### Gossip Protocol

When a node detects gas above a threshold:
1. It creates a `GossipMessage` carrying its concentration, gradient, and velocity
2. The message is delivered to all NetworkX neighbors
3. Receiving nodes compute predicted arrival time: `T = timestamp + distance / |velocity|`
4. If the prediction is novel (earlier than existing), the message is forwarded further
5. Multi-hop propagation controlled by `gossip_rounds` (per step) and `max_hops` (total)

### SensorField

High-level manager that orchestrates the per-step cycle:
1. **Sense**: all nodes read `world.phi` at their position, add noise, run Kalman update
2. **Gradient**: compute spatial gradient and estimate flow velocity
3. **Gossip**: propagate predictions through the network

## Configuration

Block 4 uses the existing `sensors` and `noise` sections from the environment JSON:

```json
"sensors": {
  "placement": "grid",
  "spacing": 3,
  "z_levels": [2],
  "communication_radius": 5.0
},
"noise": {
  "sensor_sigma": 0.05,
  "source_rate_sigma": 0.0
}
```

Additional `SensorField` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gossip_rounds` | 1 | Gossip hops per simulation step (1 = realistic delay) |
| `detection_threshold` | 0.01 | Minimum concentration to trigger gossip |
| `max_hops` | 10 | Maximum relay hops for a single message |
| `process_noise_var` | 0.01 | Kalman process noise variance |
| `seed` | None | RNG seed for reproducible noise |

## API

```python
from blocks.world import Environment, World
from blocks.network import SensorNetwork
from blocks.sensor import SensorField

env = Environment("configs/environments/default_maze.json")
world = World(env)
network = SensorNetwork(env)
field = SensorField(env, network, gossip_rounds=1, seed=42)

# Per-step cycle
for step in range(200):
    world.step()
    field.step(world)

# Query predictions
arrivals = field.get_predicted_arrivals()  # {node_name: arrival_time}
alerts = field.get_alert_nodes(world.time, horizon=5.0)  # nodes expecting gas soon

# Metrics
print(field.metrics(world))

# Individual node access
for name, node in field.nodes.items():
    print(node.metrics())
```

## Metrics

`SensorField.metrics()` returns:

| Metric | Description |
|--------|-------------|
| `n_nodes` | Total sensor count |
| `n_detecting` | Nodes with concentration above threshold |
| `n_with_predictions` | Nodes that have received gossip predictions |
| `prediction_coverage` | Fraction of nodes with predictions |
| `total_messages_sent` | Cumulative gossip messages generated |
| `total_messages_received` | Cumulative messages received |
| `mean_filtered_concentration` | Average Kalman-filtered reading |
| `mean_velocity_magnitude` | Average estimated flow speed |
| `earliest_global_arrival` | Earliest predicted arrival across all nodes |
| `mean_predicted_arrival` | Mean predicted arrival time |
| `concentration_rmse` | RMSE vs ground truth (if world provided) |

`SensorNode.metrics()` returns per-node details including `filtered_concentration`, `gradient_magnitude`, `velocity_magnitude`, `earliest_predicted_arrival`, and message counts.

## Run Tests

```bash
python -m pytest blocks/sensor/test_sensor.py -v
```
