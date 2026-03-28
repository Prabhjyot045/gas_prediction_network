# Block 7: Benchmark — Edge vs Centralized Comparison

## Overview

Runs the same environment with both actuation policies (Aether-Edge decentralized vs centralized reactive) and compares results. The primary metrics are **comfort violation**, **energy consumption**, and **Age of Information** — demonstrating that edge-native decisions outperform centralized control on data freshness.

## How to Run

```bash
conda activate ece659

# Run benchmark comparison
python -m blocks.benchmark.demo --config configs/environments/university_floor.json

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
    env_config="configs/environments/university_floor.json",
    n_steps=500,
    record_every=10,
    seed=42,
    output_dir="results/benchmark",
)

# Run both and compare
comparison = bm.run()
print(comparison["comparison"]["comfort_improvement_pct"])
print(comparison["comparison"]["energy_savings_pct"])

# Or run individually
edge_sim = bm.run_edge()
cent_sim = bm.run_centralized()
```

## Comparison Metrics

| Metric | Description | Lower = Better? |
|--------|-------------|-----------------|
| `cumulative_comfort_violation` | Time-integrated temperature exceedance | Yes |
| `cumulative_energy` | Total cooling energy consumed | Yes |
| `mean_aoi` | Average Age of Information at decision time | Yes |
| `max_overshoot` | Peak degrees above setpoint | Yes |
| `total_gossip_messages` | Communication overhead | Yes |
| `comfort_improvement_pct` | Edge vs centralized comfort delta | Higher = edge wins |
| `energy_savings_pct` | Edge vs centralized energy delta | Higher = edge wins |

## Output Files

When `output_dir` is set, the benchmark saves:
- `comparison.json` — side-by-side metric comparison
- `edge_metrics.json` — full edge policy run data
- `centralized_metrics.json` — full centralized run data
