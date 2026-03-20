# Block 3: Network — Sensor Mesh Topology

## Overview

Builds a sensor communication network from the environment config using NetworkX. Sensors are placed in non-wall cells via configurable strategies (grid, random, manual), and edges connect nodes within a communication radius. Exposes rich topology metrics for reporting.

## How to Run

```bash
conda activate ece659

# Demo: topology visualization
python -m blocks.network.demo --config configs/environments/default_maze.json

# Override communication radius
python -m blocks.network.demo --config configs/environments/default_maze.json --comm-radius 4.0

# Run tests
python -m pytest blocks/network/test_network.py -v
```

## Sensor Configuration (JSON)

Add a `"sensors"` section to your environment JSON:

```json
"sensors": {
  "placement": "grid",
  "spacing": 3,
  "z_levels": [2],
  "communication_radius": 5.0
}
```

### Placement Strategies

| Strategy | Fields | Description |
|----------|--------|-------------|
| `grid` | `spacing`, `z_levels` | Uniform grid at given spacing on specified z-levels |
| `random` | `count`, `seed`, `z_levels` | N random non-wall cells (reproducible with seed) |
| `manual` | `nodes` | Explicit list: `[{"name": "s1", "position": {"x":3, "y":3, "z":2}}]` |

All positions are validated against the wall mask.

## Metrics

`SensorNetwork.metrics()` returns a dict with:

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
| `coverage` | Fraction of non-wall cells within sensing radius of any node |
| `degree_distribution` | Histogram: `{degree: count}` |

## API

```python
from blocks.world import Environment
from blocks.network import SensorNetwork

env = Environment("configs/environments/default_maze.json")
net = SensorNetwork(env)

# Topology info
print(net.n_nodes, net.n_edges)
print(net.is_connected())
print(net.metrics())

# Access the NetworkX graph directly
for node in net.graph.nodes:
    print(node, net.graph.degree(node))

# Sensor positions as Nx3 array
positions = net.node_positions_array()

# Override communication radius
net2 = SensorNetwork(env, comm_radius=3.0)
```
