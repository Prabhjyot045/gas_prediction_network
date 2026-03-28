# Interface — Environment I/O Boundary

## Overview

The interface block is the **only** component that touches both the physical environment (World) and the inference network (SensorField). It enforces a clean separation between what interacts with the environment and what does inference.

**Input side**: Reads `World.T` at sensor positions → feeds scalar `(timestamp, value)` pairs to the SensorField.

**Output side**: Reads urgency from the SensorField → translates to vent routing commands → applies openings to World dampers.

## Policies

### Edge (Aether-Edge)

- Each vent's opening = `urgency_i / total_urgency`
- Decisions are local, based on gossip-propagated urgencies
- **Age of Information = 0** (fresh data always)

### Centralized (Baseline)

- Central controller polls sensors at `polling_interval` (+ random jitter)
- Same allocation algorithm, but data is stale between polls
- **Age of Information = polling_interval + jitter + compute_delay** (~7.6s typical)

## How to Run

```bash
conda activate ece659

# Run tests
python -m pytest blocks/interface/test_interface.py -v
```

## API

```python
from blocks.world import Environment, World
from blocks.sensor import SensorNetwork, SensorField
from blocks.interface.interface import EnvironmentInterface

env = Environment("configs/environments/university_floor.json")
world = World(env)
network = SensorNetwork(env)
field = SensorField(env, network, gossip_rounds=2, seed=42)
iface = EnvironmentInterface(env, field, policy="edge", seed=42)

# Full cycle: read → infer → actuate
for _ in range(100):
    world.step()
    openings = iface.step(world)  # returns {damper_name: opening}

# Metrics
print(iface.metrics())
print(f"AoI: {iface.mean_age_of_information:.2f}s")
print(f"Energy: {iface.total_energy:.4f}")
```

## Vent Routing

With fixed total airflow (`Q_total`), the interface redistributes cooling:
- Rooms with no urgency (empty/cool) → minimal opening (0.1)
- Rooms with high urgency (occupied/heating) → proportionally more of the budget
- `opening_i = urgency_i / sum(urgency_all)`

## Metrics

`EnvironmentInterface.metrics()` returns:

| Metric | Description |
|--------|-------------|
| `policy` | Active policy ("edge" or "centralized") |
| `damper_openings` | Current opening per damper |
| `total_actions` | Total vent adjustment count |
| `total_energy` | Cumulative cooling energy |
| `mean_age_of_information` | Average data staleness |
| `total_gossip_messages` | Communication overhead |
