# Metrics Infrastructure — MetricsCollector + ExperimentRunner

## Overview

Lightweight metrics and experiment infrastructure for collecting, serializing, and comparing results across simulation runs. Every block exposes a `metrics() -> dict` method that feeds into the `MetricsCollector`.

## MetricsCollector

Accumulates structured metrics from simulation runs.

```python
from blocks.metrics import MetricsCollector

mc = MetricsCollector("my_experiment")
mc.set_metadata(config="default_maze.json", strategy="vdpa")

# Per-step recording
for step in range(100):
    world.step()
    mc.record(world.metrics(), step=step)
    mc.record_scalar("total_mass", world.total_mass(), step=step)

# Export
mc.save_json("results/run_001.json")
mc.save_csv("results/run_001.csv")
df = mc.to_dataframe()  # requires pandas
```

### Export Formats

| Method | Format | Best for |
|--------|--------|----------|
| `save_json()` | JSON | Full data with metadata, programmatic analysis |
| `save_csv()` | CSV | Quick plots in Excel/matplotlib, scalar time-series |
| `to_dataframe()` | pandas DataFrame | In-notebook analysis |

## ExperimentRunner

Drives parameter sweeps from a JSON config file.

```bash
# Run a sensor density sweep
python -c "
from blocks.metrics import ExperimentRunner
from blocks.world.environment import Environment
from blocks.world.world import World
from blocks.network import SensorNetwork

def run(env_config, sim_config, collector):
    import tempfile, json
    path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(env_config, path); path.close()
    env = Environment(path.name)
    world = World(env)
    net = SensorNetwork(env)
    collector.record(net.metrics(), step=0)
    n_steps = sim_config.get('n_steps', 100)
    for i in range(n_steps):
        world.step()
    collector.record(world.metrics(), step=n_steps)

runner = ExperimentRunner('configs/experiments/sweep_sensor_density.json')
runner.run(run)
"
```

### Experiment Config Format

```json
{
  "name": "sensor_density_sweep",
  "base_environment": "configs/environments/default_maze.json",
  "parameters": {
    "sensors.spacing": [2, 3, 4, 5, 6],
    "sensors.communication_radius": [3.0, 5.0, 7.0]
  },
  "simulation": {
    "n_steps": 300,
    "metrics_every": 30
  },
  "output_dir": "results/sensor_density_sweep"
}
```

- `parameters`: dot-notation keys that patch the base environment JSON
- Cartesian product of all parameter values = total number of runs
- Each run gets its own `MetricsCollector` and output JSON

## Metrics Convention

Every block exposes metrics via a `metrics() -> dict` method:

| Block | Method | Key metrics |
|-------|--------|-------------|
| World | `world.metrics()` | `step`, `time`, `total_mass`, `contaminated_volume`, `peak_concentration` |
| Network | `network.metrics()` | `n_nodes`, `n_edges`, `diameter`, `coverage`, `clustering_coefficient` |
| Sensor | `node.metrics()` / `sensor_field.metrics(world)` | `filtered_concentration`, `gradient_magnitude`, `earliest_predicted_arrival`, `messages_sent`, `concentration_rmse`, `prediction_coverage` |
| Actuator | `actuator.metrics()` | `doors_closed`, `response_time`, `first_detection_time`, `first_actuation_time`, `actuation_log` |
| Benchmark | `benchmark.run()` | `cumulative_contamination`, `contamination_reduction_pct`, `response_time` |

## Run Tests

```bash
python -m pytest blocks/metrics/test_metrics.py -v
```
