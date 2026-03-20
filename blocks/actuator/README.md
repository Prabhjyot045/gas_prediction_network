# Block 5: Actuator — Door Actuation Controller

## Overview

Manages door actuations based on sensor data. Supports two policies for fair comparison:

- **Predictive (VDPA)**: close doors when gossip-propagated predictions indicate gas will arrive within a configurable time horizon. Acts *before* gas reaches the door.
- **Reactive (centralized)**: close doors when Kalman-filtered sensor readings near the door exceed a concentration threshold. Acts *after* gas has arrived.

## How to Run

```bash
conda activate ece659

# Demo: predictive policy
python -m blocks.actuator.demo --config configs/environments/default_maze.json

# Demo: reactive policy for comparison
python -m blocks.actuator.demo --config configs/environments/default_maze.json --policy reactive

# Adjust prediction horizon
python -m blocks.actuator.demo --config configs/environments/default_maze.json --horizon 3.0

# Run tests
python -m pytest blocks/actuator/test_actuator.py -v
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy` | `"predictive"` | `"predictive"` (VDPA) or `"reactive"` (centralized) |
| `horizon` | 5.0 | Seconds ahead to look for predictions (predictive only) |
| `concentration_threshold` | 0.5 | Trigger threshold (reactive only) |
| `proximity_radius` | 3.0 | Max distance (grid cells) from door center to consider sensors |

## API

```python
from blocks.actuator import ActuatorController

ac = ActuatorController(env, sensor_field, policy="predictive", horizon=5.0)

# Each step: evaluate and actuate
closed_doors = ac.evaluate(world)

# Metrics
print(ac.metrics())
print(f"Response time: {ac.response_time}")
```

## Metrics

| Metric | Description |
|--------|-------------|
| `policy` | Active policy name |
| `doors_closed` | Number of unique doors closed |
| `total_actuations` | Total actuation events |
| `first_detection_time` | When gas was first detected by any sensor |
| `first_actuation_time` | When the first door was closed |
| `response_time` | Time between detection and first actuation |
| `actuation_log` | Detailed log of each actuation event |
