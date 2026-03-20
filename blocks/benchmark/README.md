# Block 7: Benchmark — VDPA vs Centralized Comparison

## Overview

Runs the same environment with both actuation policies (VDPA predictive vs centralized reactive) and compares results. The primary metric is **time-integrated contamination volume** — lower is better, meaning the actuation system prevented more gas spread.

## How to Run

```bash
conda activate ece659

# Run benchmark comparison
python -m blocks.benchmark.demo --config configs/environments/default_maze.json

# Save results to disk
python -m blocks.benchmark.demo --config configs/environments/default_maze.json --save results/benchmark

# From experiment config
python -c "
from blocks.benchmark import Benchmark
bm = Benchmark.from_config('configs/experiments/benchmark_default.json')
results = bm.run()
import json; print(json.dumps(results, indent=2, default=str))
"

# Run tests
python -m pytest blocks/benchmark/test_benchmark.py -v
```

## API

```python
from blocks.benchmark import Benchmark

bm = Benchmark(
    env_config="configs/environments/default_maze.json",
    n_steps=500,
    record_every=10,
    predictive_horizon=5.0,
    gossip_rounds=2,
    reactive_threshold=0.5,
    seed=42,
    output_dir="results/benchmark",
)

# Run both and compare
comparison = bm.run()
print(comparison["comparison"]["contamination_reduction_pct"])

# Or run individually
pred_sim = bm.run_predictive()
react_sim = bm.run_reactive()
```

## Comparison Metrics

| Metric | Description |
|--------|-------------|
| `cumulative_contamination` | Time-integrated contaminated volume (primary) |
| `response_time` | Time from first detection to first door closure |
| `first_detection_time` | When gas was first detected |
| `first_actuation_time` | When the first door was closed |
| `doors_closed` | Total doors closed during simulation |
| `total_mass_final` | Total gas in system at end |
| `peak_concentration` | Maximum concentration reached |
| `contaminated_volume` | Contaminated cells at end |
| `contamination_reduction_pct` | Percentage improvement of predictive over reactive |

## Benchmark Config Schema

```json
{
    "environment": "configs/environments/default_maze.json",
    "n_steps": 500,
    "record_every": 10,
    "predictive": {
        "horizon": 5.0,
        "gossip_rounds": 2
    },
    "reactive": {
        "threshold": 0.5
    },
    "proximity_radius": 3.0,
    "detection_threshold": 0.01,
    "seed": 42,
    "output_dir": "results/benchmark"
}
```

## Output Files

When `output_dir` is set, the benchmark saves:
- `comparison.json` — side-by-side metric comparison
- `predictive_metrics.json` — full VDPA run data
- `predictive_scalars.csv` — VDPA time-series (for plotting)
- `reactive_metrics.json` — full reactive run data
- `reactive_scalars.csv` — reactive time-series
