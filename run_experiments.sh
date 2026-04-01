#!/usr/bin/env bash
# run_experiments.sh — Full experimental coverage: 5 seeds per config
# Usage:
#   bash run_experiments.sh university_floor
#   bash run_experiments.sh airport
#   bash run_experiments.sh big_university
#
# Covers the three key claims:
#   1. Edge achieves lower comfort violation than centralized (all environments)
#   2. Edge AoI ≈ 0s vs centralized ~20s (all environments)
#   3. Gossip scales to large meshes (big_university)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_experiments.sh <config_name>"
    echo "  config_name: university_floor | airport | big_university"
    exit 1
fi

CONFIG_NAME="$1"
CONFIG_PATH="configs/environments/${CONFIG_NAME}.json"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    exit 1
fi

CONFIGS=("$CONFIG_PATH")
SEEDS=(42 7 13 99 2025)
STEPS=2000

# GIF parameters — tuned for 2000-step runs:
#   200 warm-up steps let thermal gradients develop before recording starts
#   200 frames × 10 steps/frame covers the full 2000-step window
GIF_FRAMES=200
GIF_STEPS_PER_FRAME=10
GIF_PRE_STEPS=200
GIF_FPS=15
GIF_Z_SLICE=1   # mid-floor slice where occupancy effects are strongest

RESULTS_BASE="results/experiments_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$RESULTS_BASE/summary.tsv"

mkdir -p "$RESULTS_BASE"
echo -e "config\tseed\tedge_comfort\tcent_comfort\tcomfort_pct\tedge_energy\tcent_energy\tenergy_pct\tedge_aoi\tcent_aoi\tedge_msgs\tcent_msgs" \
    > "$SUMMARY_FILE"

total=$((${#CONFIGS[@]} * ${#SEEDS[@]}))
run=0

for cfg in "${CONFIGS[@]}"; do
    cfg_name=$(basename "$cfg" .json)
    for seed in "${SEEDS[@]}"; do
        run=$((run + 1))
        out_dir="$RESULTS_BASE/${cfg_name}_seed${seed}"
        echo ""
        echo "[$run/$total] $cfg_name  seed=$seed  → $out_dir"

        conda run -n ece659 python run_benchmark.py \
            --config "$cfg" \
            --steps "$STEPS" \
            --seed "$seed" \
            --output "$out_dir" \
            --frames "$GIF_FRAMES" \
            --steps-per-frame "$GIF_STEPS_PER_FRAME" \
            --pre-steps "$GIF_PRE_STEPS" \
            --fps "$GIF_FPS" \
            --z-slice "$GIF_Z_SLICE"

        # Extract key scalars from comparison.json into the summary TSV
        python3 - "$out_dir/comparison.json" "$cfg_name" "$seed" >> "$SUMMARY_FILE" <<'PYEOF'
import json, sys
path, cfg, seed = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    d = json.load(f)
e, c, cmp = d["edge"], d["centralized"], d["comparison"]
print(
    cfg, seed,
    f"{e['cumulative_comfort_violation']:.4f}",
    f"{c['cumulative_comfort_violation']:.4f}",
    f"{cmp['comfort_improvement_pct']:.2f}",
    f"{e['cumulative_energy']:.2f}",
    f"{c['cumulative_energy']:.2f}",
    f"{cmp['energy_savings_pct']:.2f}",
    f"{e['mean_aoi']:.2f}",
    f"{c['mean_aoi']:.2f}",
    e['total_messages'],
    c['total_messages'],
    sep="\t"
)
PYEOF
    done
done

# ── Parameter sweeps (university_floor only) ─────────────────────
if [[ "$CONFIG_NAME" == "university_floor" ]]; then
    echo ""
    echo "[sweep] Running parameter sweeps → $RESULTS_BASE/sweep/"
    conda run -n ece659 python -m blocks.metrics.demo \
        --sweep configs/experiments/sweep_sensor_density.json \
        --output "$RESULTS_BASE/sweep" \
        --seed 42 \
        --z-slice "$GIF_Z_SLICE"
fi

echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETE — $total runs  [$CONFIG_NAME]"
echo "Results: $RESULTS_BASE/"
echo ""
echo "Summary (TSV):"
column -t -s $'\t' "$SUMMARY_FILE"
echo ""
echo "Key claims to verify from summary:"
echo "  • comfort_pct > 0   → Edge reduces comfort violations vs centralized"
echo "  • edge_aoi ≈ 0      → Edge operates on fresh data (zero AoI)"
echo "  • cent_aoi ≈ 20     → Centralized carries ~20s staleness"
echo "  • Consistent across all 3 environments and all 5 seeds"
