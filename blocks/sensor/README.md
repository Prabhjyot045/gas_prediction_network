# Block 4: Sensor — Rate-of-Change Monitoring + Gossip Protocol

## Overview

Event-driven sensor nodes that read environmental metrics (e.g., temperature) from the World, monitor their rate of change via a rolling buffer, predict Time-To-Impact (TTI) for comfort thresholds, and propagate urgency predictions through the sensor network via a gossip protocol. This forms the sensing and prediction layers of **Aether-Edge**: distributed predictive resource allocation.

## Architecture

### SensorNode

Each edge inference node separates concerns into three layers:
- **Sensing (Layer 1)**: Maintains a `RollingBuffer` of recent readings and computes the least-squares slope (`dT/dt`) to estimate the rate of change.
- **Prediction (Layer 2)**: Computes Time-To-Impact `TTI = (setpoint - current) / (dT/dt)`. Converts this into an actionable `urgency = 1 / TTI` score.
- **Communication (Layer 3)**: Creates `NegotiationMessage`s carrying the node's urgency, temperature, and rate of change.

### Gossip Protocol

When a node detects a meaningful rate of change (`|dT/dt| > talk_threshold`):
1. It creates a `NegotiationMessage` carrying its local urgency.
2. The message is delivered to all NetworkX neighbors.
3. Receiving nodes compute their highest known urgencies and update their tables.
4. If a message contains a higher urgency from a given origin than previously known, the message is forwarded further.
5. Multi-hop propagation is controlled by `gossip_rounds` (per step) and `max_hops` (total).

### SensorField

High-level manager that orchestrates the per-step cycle:
1. **Sense**: all nodes read temperature (with optional noise), update their rolling buffers, and compute `dT/dt`.
2. **Gradient**: compute spatial gradients via central differences.
3. **Gossip**: calculate TTI/urgencies and propagate predictions through the network.

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
  "sensor_sigma": 0.05
}
```

Additional `SensorField` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gossip_rounds` | 1 | Gossip hops per simulation step |
| `talk_threshold` | 0.01 | Minimum `\|dT/dt\|` to trigger gossip message generation |
| `max_hops` | 10 | Maximum relay hops for a single message |
| `buffer_seconds` | 30.0 | Time window for the rolling buffer |
| `seed` | None | RNG seed for reproducible noise |

## API

```python
from blocks.world import Environment, World
from blocks.network import SensorNetwork
from blocks.sensor import SensorField

env = Environment("configs/environments/university_floor.json")
world = World(env)
network = SensorNetwork(env)
field = SensorField(env, network, gossip_rounds=1, seed=42)

# Per-step cycle
for step in range(200):
    world.step()
    field.step(world)

# Query predictions
urgencies = field.get_urgencies()  # {node_name: urgency}
alerts = field.get_alert_nodes(horizon=30.0)  # nodes expecting breach within 30s

# Metrics
print(field.metrics(world))
```

## Metrics

`SensorField.metrics()` returns aggregation statistics including `n_heating`, `n_with_urgency`, `urgency_coverage`, `total_messages_sent`, `mean_dT_dt`, and `mean_tti`.

`SensorNode.metrics()` returns per-node details including `filtered_temperature`, `dT_dt`, `tti`, `urgency`, `gradient_magnitude`, and message counts.

## Run Tests

```bash
python -m pytest blocks/sensor/test_sensor.py -v
```
