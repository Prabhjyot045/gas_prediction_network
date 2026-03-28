# Sensor Network — Topology + Inference + Gossip

## Overview

The sensor block combines **mesh topology** (placement, communication graph) and **domain-agnostic inference** (rate-of-change monitoring, TTI prediction, gossip consensus) into a single unified package. Sensor nodes receive scalar `(timestamp, value)` pairs from the interface layer and perform three-layer inference without any knowledge of the physical environment.

### Three Inference Layers

1. **Sensing**: Rolling buffer + least-squares slope estimation (dV/dt)
2. **Prediction**: TTI = (threshold - current) / (rate of change), urgency = 1/TTI
3. **Communication**: Multi-hop gossip propagates urgency to neighbors for distributed consensus

### Mesh Topology

Sensors are placed via configurable strategies (grid, random, manual) and connected by a NetworkX communication graph where edges link nodes within a configurable radius.

## How to Run

```bash
conda activate ece659

# Run tests (all sensor + topology tests)
python -m pytest blocks/sensor/test_sensor.py -v

# Network topology tests (backward compat location)
python -m pytest blocks/network/test_network.py -v
```

## Modules

| File | Purpose |
|------|---------|
| `node.py` | `RollingBuffer`, `SensorNode` — per-node inference (buffer, TTI, urgency) |
| `gossip.py` | `NegotiationMessage` — inter-node urgency message |
| `sensor_field.py` | `SensorField` — orchestrates sense -> predict -> gossip cycle |
| `sensor_network.py` | `SensorNetwork` — NetworkX graph + topology metrics |
| `placement.py` | Grid/random/manual sensor placement strategies |
| `test_sensor.py` | 27 unit tests for inference pipeline |

## Configuration

The `sensors` section of the environment JSON controls placement and communication:

```json
"sensors": {
  "placement": "grid",
  "spacing": 4,
  "z_levels": [1],
  "communication_radius": 6.0
}
```

### Placement Strategies

| Strategy | Fields | Description |
|----------|--------|-------------|
| `grid` | `spacing`, `z_levels` | Uniform grid at given spacing on specified z-levels |
| `random` | `count`, `seed`, `z_levels` | N random non-wall cells (reproducible with seed) |
| `manual` | `nodes` | Explicit list: `[{"name": "s1", "position": {"x":3, "y":3, "z":1}}]` |

All positions are validated against the wall mask.

### SensorField Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gossip_rounds` | 1 | Gossip hops per simulation step |
| `talk_threshold` | 0.01 | Minimum \|dV/dt\| to trigger gossip |
| `max_hops` | 10 | Maximum relay hops for a single message |
| `buffer_seconds` | 30.0 | Rolling buffer window in seconds |
| `seed` | None | RNG seed for reproducible noise |

## API

```python
from blocks.world import Environment, World
from blocks.sensor import SensorNetwork, SensorField

env = Environment("configs/environments/university_floor.json")
world = World(env)
network = SensorNetwork(env)
field = SensorField(env, network, gossip_rounds=2, seed=42)

# Domain-agnostic step: feed scalar readings
readings = {name: float(world.T[node.position]) for name, node in field.nodes.items()}
field.step(readings, timestamp=world.time)

# Query predictions
urgencies = field.get_urgencies()    # {node_name: urgency}
ttis = field.get_ttis()              # {node_name: TTI_seconds}
alerts = field.get_alert_nodes()     # nodes predicting breach within 30s

# Topology info
print(network.n_nodes, network.n_edges)
print(network.is_connected())
print(network.metrics())

# Inference metrics
print(field.metrics())
```

## Topology Metrics

`SensorNetwork.metrics()` returns:

| Metric | Description |
|--------|-------------|
| `n_nodes` | Total sensor count |
| `n_edges` | Communication links |
| `is_connected` | Whether all nodes can reach each other |
| `connected_components` | Number of disconnected subgraphs |
| `average_degree` | Mean connections per node |
| `diameter` | Longest shortest path (inf if disconnected) |
| `average_path_length` | Mean shortest path length |
| `clustering_coefficient` | Local clustering average |
| `coverage` | Fraction of non-wall cells within sensing radius |
| `degree_distribution` | Histogram: `{degree: count}` |

## Inference Metrics

`SensorField.metrics()` returns:

| Metric | Description |
|--------|-------------|
| `n_nodes` | Total sensor count |
| `n_heating` | Nodes with dV/dt above threshold |
| `n_with_urgency` | Nodes predicting setpoint breach |
| `urgency_coverage` | Fraction of nodes with urgency |
| `total_messages_sent` | Cumulative gossip messages generated |
| `total_messages_received` | Cumulative messages received |
| `mean_value` | Average filtered reading across nodes |
| `mean_dT_dt` | Average rate of change |
| `min_tti` / `mean_tti` | Time-to-impact statistics |
