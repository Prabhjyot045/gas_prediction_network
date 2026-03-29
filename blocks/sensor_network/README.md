# Sensor Network — Topology + Inference + Gossip

## Overview

Unified sensor network block combining mesh topology (placement, communication graph, coverage metrics) with the inference engine (rolling buffer, TTI prediction, urgency gossip). This is the core "intelligent edge" of Aether-Edge — domain-agnostic nodes that monitor rate of change, predict threshold breaches, and propagate urgency via gossip.

The interface layer (`blocks/interface/`) is the only code that touches the physical environment. This block receives scalar `(timestamp, value)` pairs and outputs urgency scores.

## Architecture

### SensorNetwork (Topology)

- Places sensors via configurable strategy (grid, random, manual)
- Builds a NetworkX communication graph (edges within radio range)
- Exposes rich topology metrics: connectivity, coverage, degree distribution, clustering

### SensorNode (Per-Node Inference)

Three-layer inference pipeline:
- **Layer 1 — Sensing**: `RollingBuffer` of recent readings + least-squares slope (`dV/dt`)
- **Layer 2 — Prediction**: `TTI = (setpoint - current) / (dV/dt)` → `urgency = 1/TTI`
- **Layer 3 — Communication**: `NegotiationMessage` carries urgency to graph neighbors

### SensorField (Orchestrator)

Manages all nodes and runs the per-step cycle:
1. **Sense**: Feed scalar readings to each node's rolling buffer
2. **Predict**: Compute TTI and urgency at each node
3. **Gossip**: Propagate urgency through the mesh (multi-hop, bandwidth-aware)

### Gossip Protocol

- Nodes with `|dV/dt| > talk_threshold` generate messages (bandwidth control)
- Messages forwarded one hop per gossip round, up to `max_hops`
- Each node builds a neighbor urgency table for local resource allocation

## Configuration

```json
"sensors": {
  "placement": "grid",
  "spacing": 4,
  "z_levels": [1],
  "communication_radius": 6.0
},
"noise": {
  "sensor_sigma": 0.05
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gossip_rounds` | 1 | Gossip hops per simulation step |
| `talk_threshold` | 0.01 | Minimum `\|dV/dt\|` to trigger gossip |
| `max_hops` | 10 | Maximum relay hops per message |
| `buffer_seconds` | 30.0 | Rolling buffer time window |
| `seed` | None | RNG seed for reproducible noise |

## API

```python
from blocks.world import Environment, World
from blocks.sensor_network import SensorNetwork, SensorField

env = Environment("configs/environments/university_floor.json")
world = World(env)
network = SensorNetwork(env)
field = SensorField(env, network, gossip_rounds=2, seed=42)

# Feed scalar readings (via interface in production)
readings = {name: float(world.T[node.position]) for name, node in field.nodes.items()}
field.step(readings, timestamp=world.time)

# Query predictions
urgencies = field.get_urgencies()       # {node_name: urgency}
alerts = field.get_alert_nodes(horizon=30.0)

# Topology metrics
print(network.metrics())  # n_nodes, coverage, clustering, etc.

# Inference metrics
print(field.metrics())    # n_heating, urgency_coverage, messages, etc.
```

## Metrics

**SensorNetwork.metrics()**: `n_nodes`, `n_edges`, `communication_radius`, `is_connected`, `connected_components`, `average_degree`, `diameter`, `average_path_length`, `clustering_coefficient`, `coverage`, `degree_distribution`

**SensorField.metrics()**: `n_nodes`, `n_heating`, `n_with_urgency`, `urgency_coverage`, `total_messages_sent`, `total_messages_received`, `mean_value`, `mean_dT_dt`, `min_tti`, `mean_tti`

## Run Tests

```bash
python -m pytest blocks/sensor_network/test_sensor_network.py -v
```
