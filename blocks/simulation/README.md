# Block 6: Simulation — Full Integration Loop

## Overview

Ties all Aether-Edge blocks into a single simulation loop:

```
World (physics) → Interface (read) → SensorField (infer) → Interface (actuate) → World
```

The Interface block is the only component that touches both the physical environment (World) and the inference network (SensorField). The sensor network never sees the World directly.

## How to Run

```bash
conda activate ece659

# Run simulation
python -m blocks.simulation.demo --config configs/environments/university_floor.json

# Centralized baseline
python -m blocks.simulation.demo --config configs/environments/university_floor.json --policy centralized

# Run tests
python -m pytest blocks/simulation/test_simulation.py -v
```

## API

```python
from blocks.simulation import Simulation

# Direct construction
sim = Simulation(
    "configs/environments/university_floor.json",
    actuator_policy="edge",
    gossip_rounds=2,
    seed=42,
)

# Run
sim.run(n_steps=500, record_every=10)

# Results
print(sim.summary())
print(f"Comfort violation: {sim.cumulative_comfort_violation:.4f}")
print(f"Energy: {sim.cumulative_energy:.4f}")

# Export
sim.collector.save_json("results/run.json")

# From JSON config
sim = Simulation.from_config("configs/simulations/my_run.json")
```

## Step Order

Each `sim.step()` executes:
1. `world.step()` — advance thermal physics
2. `interface.step(world)` — read environment → feed sensors → route airflow → actuate vents
3. Accumulate comfort violation and energy integrals

## Recorded Metrics

Per-record (at each `record_every` interval):

| Field | Source |
|-------|--------|
| `step`, `time`, `mean_temperature`, `max_overshoot` | World |
| `temperature_rmse`, `n_heating`, `urgency_coverage`, `total_gossip_messages` | SensorField |
| `mean_age_of_information`, `cumulative_comfort_violation`, `cumulative_energy` | Interface + accumulated |
