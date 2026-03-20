# Block 6: Simulation — Full Integration

## Overview

Ties all VDPA blocks into a single simulation loop: World (physics) → SensorField (Kalman + gossip) → ActuatorController (door decisions) → MetricsCollector (recording). Supports both predictive and reactive policies via configuration.

## How to Run

```bash
conda activate ece659

# Full VDPA simulation
python -m blocks.simulation.demo --config configs/environments/default_maze.json

# Reactive baseline
python -m blocks.simulation.demo --config configs/environments/default_maze.json --policy reactive

# Save results
python -m blocks.simulation.demo --config configs/environments/default_maze.json --save results/demo_run

# Run tests
python -m pytest blocks/simulation/test_simulation.py -v
```

## API

```python
from blocks.simulation import Simulation

# Direct construction
sim = Simulation(
    "configs/environments/default_maze.json",
    actuator_policy="predictive",
    actuator_horizon=5.0,
    gossip_rounds=2,
    seed=42,
)

# Run
sim.run(n_steps=500, record_every=10)

# Results
print(sim.summary())
print(f"Contamination: {sim.cumulative_contamination:.4f}")

# Export
sim.collector.save_json("results/run.json")
sim.collector.save_csv("results/scalars.csv")

# From JSON config
sim = Simulation.from_config("configs/simulations/my_run.json")
```

## Simulation Config Schema

```json
{
    "environment": "configs/environments/default_maze.json",
    "actuator": {
        "policy": "predictive",
        "horizon": 5.0,
        "concentration_threshold": 0.5,
        "proximity_radius": 3.0
    },
    "sensor_field": {
        "gossip_rounds": 1,
        "detection_threshold": 0.01,
        "max_hops": 10
    },
    "simulation": {
        "seed": 42,
        "name": "my_experiment"
    }
}
```

## Step Order

Each `sim.step()` executes in this order:
1. `world.step()` — advance diffusion physics
2. `sensor_field.step(world)` — sense, filter, gossip
3. `actuator.evaluate(world)` — check predictions, close doors
4. Accumulate contamination integral

## Recorded Metrics

Per-record (at each `record_every` interval):

| Field | Source |
|-------|--------|
| `step`, `time`, `total_mass`, `contaminated_volume`, `peak_concentration` | World |
| `concentration_rmse`, `n_detecting`, `prediction_coverage`, `total_gossip_messages` | SensorField |
| `doors_closed`, `cumulative_contamination` | Actuator + accumulated |

Scalar time-series (for plotting): `total_mass`, `contaminated_volume`, `cumulative_contamination`, `peak_concentration`, `concentration_rmse`, `prediction_coverage`, `n_detecting`.
