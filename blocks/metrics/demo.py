"""
Metrics Demo — Parameter Sweep from experiment config.

Reads a sweep config JSON, runs Benchmark for each parameter value,
and saves one combined plot per parameter.

Usage:
    python -m blocks.metrics.demo --sweep configs/experiments/sweep_sensor_density.json
    python -m blocks.metrics.demo --sweep configs/experiments/sweep_sensor_density.json --z-slice 1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from blocks.benchmark.benchmark import Benchmark


# ── Config helpers ────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_param(config: dict, param_path: str, value) -> dict:
    """Set a nested config value using dot-notation, e.g. 'sensors.spacing'."""
    cfg = copy.deepcopy(config)
    keys = param_path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return cfg


def write_temp_config(config: dict) -> str:
    """Write config dict to a temp JSON file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(config, tmp)
    tmp.flush()
    return tmp.name


# ── Run one benchmark (edge + centralized) for a single param value ───────────
def run_benchmark(
    env_config_path: str,
    n_steps: int,
    record_every: int,
    gossip_rounds: int = 2,  # 1. Add this as a parameter (default to 2)
    seed: int = 42,
) -> tuple:
    """Run edge + centralized benchmark. Returns (edge_sim, cent_sim, comparison)."""
    bm = Benchmark(
        env_config=env_config_path,
        n_steps=n_steps,
        record_every=record_every,
        gossip_rounds=gossip_rounds, # 2. Use the variable here
        seed=seed,
    )
    edge_sim = bm.run_edge()
    cent_sim = bm.run_centralized()
    comparison = bm.compare(edge_sim, cent_sim)
    return edge_sim, cent_sim, comparison


# ── Plot one parameter sweep ──────────────────────────────────────────────────

def plot_parameter_sweep(
    param_name: str,
    param_values: list,
    results: list[tuple],
    output_path: Path,
    z_slice: int = 1,
) -> None:
    n_vals = len(param_values)

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f0f1a")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.92, bottom=0.06, left=0.07, right=0.97,
    )

    ax_violation_bar = fig.add_subplot(gs[0, 0])
    ax_energy_bar    = fig.add_subplot(gs[0, 1])
    ax_overshoot_bar = fig.add_subplot(gs[0, 2])
    ax_violation_ts  = fig.add_subplot(gs[1, :])
    ax_energy_ts     = fig.add_subplot(gs[2, :])
    ax_overshoot_ts  = fig.add_subplot(gs[3, :])

    cmap_edge    = plt.cm.cool
    cmap_central = plt.cm.autumn
    x_labels     = [str(v) for v in param_values]
    x_pos        = np.arange(n_vals)
    bar_w        = 0.35

    # Collect final scalars
    edge_violations, cent_violations = [], []
    edge_energies,   cent_energies   = [], []
    edge_overshoots, cent_overshoots = [], []

    for edge_sim, cent_sim, comp in results:
        e = comp["edge"]
        c = comp["centralized"]
        edge_violations.append(e["cumulative_comfort_violation"])
        cent_violations.append(c["cumulative_comfort_violation"])
        edge_energies.append(e["cumulative_energy"])
        cent_energies.append(c["cumulative_energy"])
        edge_overshoots.append(e["max_overshoot"])
        cent_overshoots.append(c["max_overshoot"])

    # ── Row 0: bar charts ─────────────────────────────────────────────
    for ax, e_vals, c_vals, ylabel, title in [
        (ax_violation_bar, edge_violations, cent_violations,
         "Cumulative Violation",  "Comfort Violation"),
        (ax_energy_bar,    edge_energies,   cent_energies,
         "Cumulative Energy",     "Energy Use"),
        (ax_overshoot_bar, edge_overshoots, cent_overshoots,
         "Max Overshoot (°C)",    "Peak Overshoot"),
    ]:
        bars_e = ax.bar(x_pos - bar_w / 2, e_vals, bar_w,
                        label="Edge",        color="#00d4ff", alpha=0.85)
        bars_c = ax.bar(x_pos + bar_w / 2, c_vals, bar_w,
                        label="Centralized", color="#ff6b6b", alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        _style_ax(ax, param_name, ylabel, title)
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")
        for bar in list(bars_e) + list(bars_c):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h * 1.01,
                f"{h:.2f}", ha="center", va="bottom",
                color="white", fontsize=6,
            )

    # ── Rows 1-3: time-series ─────────────────────────────────────────
    for idx, (edge_sim, cent_sim, _) in enumerate(results):
        frac    = idx / max(n_vals - 1, 1)
        color_e = cmap_edge(0.3 + 0.7 * frac)
        color_c = cmap_central(0.3 + 0.7 * frac)
        lbl     = f"{param_name}={param_values[idx]}"

        s_e, v_e = edge_sim.collector.scalar_series("cumulative_comfort_violation")
        s_c, v_c = cent_sim.collector.scalar_series("cumulative_comfort_violation")
        ax_violation_ts.plot(s_e, v_e, color=color_e, lw=1.6,
                             label=f"Edge {lbl}")
        ax_violation_ts.plot(s_c, v_c, color=color_c, lw=1.6, ls="--",
                             label=f"Central {lbl}")

        s_e, e_e = edge_sim.collector.scalar_series("cumulative_energy")
        s_c, e_c = cent_sim.collector.scalar_series("cumulative_energy")
        ax_energy_ts.plot(s_e, e_e, color=color_e, lw=1.6, label=f"Edge {lbl}")
        ax_energy_ts.plot(s_c, e_c, color=color_c, lw=1.6, ls="--",
                          label=f"Central {lbl}")

        s_e, o_e = edge_sim.collector.scalar_series("max_overshoot")
        s_c, o_c = cent_sim.collector.scalar_series("max_overshoot")
        ax_overshoot_ts.plot(s_e, o_e, color=color_e, lw=1.6,
                             label=f"Edge {lbl}")
        ax_overshoot_ts.plot(s_c, o_c, color=color_c, lw=1.6, ls="--",
                             label=f"Central {lbl}")

    _style_ax(ax_violation_ts, "Step", "Cumulative Comfort Violation",
              "Comfort Violation Over Time")
    _style_ax(ax_energy_ts,    "Step", "Cumulative Energy",
              "Energy Usage Over Time")
    _style_ax(ax_overshoot_ts, "Step", "Max Overshoot (°C)",
              "Peak Overshoot Over Time")

    for ax in [ax_violation_ts, ax_energy_ts, ax_overshoot_ts]:
        ax.legend(
            fontsize=6.5, facecolor="#1a1a2e", labelcolor="white",
            edgecolor="#555", ncol=min(n_vals * 2, 6), loc="upper left",
        )

    param_label = param_name.replace(".", " → ")
    fig.suptitle(
        f"Sweep: {param_label}  |  values: {param_values}",
        color="white", fontsize=13, fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path}")


def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#2a2a3e", linewidth=0.5, linestyle="--")

def save_summary_table(
    all_results: dict,
    output_dir: Path,
) -> None:
    output_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for param_path, value_results in all_results.items():
        summary[param_path] = {}
        for val, (edge_sim, cent_sim, comparison) in value_results.items():
            e = comparison["edge"]
            c = comparison["centralized"]
            summary[param_path][str(val)] = {
                "edge": {
                    "cumulative_violation": e["cumulative_comfort_violation"],
                    "cumulative_energy":    e["cumulative_energy"],
                    "max_overshoot":        e["max_overshoot"],
                    "mean_aoi":             e["mean_aoi"],
                    "total_messages":       e["total_messages"],
                },
                "centralized": {
                    "cumulative_violation": c["cumulative_comfort_violation"],
                    "cumulative_energy":    c["cumulative_energy"],
                    "max_overshoot":        c["max_overshoot"],
                    "mean_aoi":             c["mean_aoi"],
                    "total_messages":       c["total_messages"],
                },
                "comparison": {
                    "comfort_improvement_pct": comparison["comparison"]["comfort_improvement_pct"],
                    "energy_savings_pct":      comparison["comparison"]["energy_savings_pct"],
                },
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary table → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aether-Edge Metrics Demo — parameter sweep"
    )
    p.add_argument(
        "--sweep", type=str, required=True,
        help="Path to sweep experiment JSON "
             "(e.g. configs/experiments/sensor_density_sweep.json)"
    )
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--z-slice", type=int, default=1)
    p.add_argument(
        "--output", type=str, default=None,
        help="Override output directory from sweep config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Step 1: load sweep config ─────────────────────────────────────
    sweep_cfg  = load_json(args.sweep)
    sweep_name = sweep_cfg["name"]
    base_env   = sweep_cfg["base_environment"]
    parameters = sweep_cfg["parameters"]
    n_steps    = sweep_cfg["simulation"]["n_steps"]
    rec_every  = sweep_cfg["simulation"]["metrics_every"]
    output_dir = Path(args.output) if args.output else Path(sweep_cfg.get("output_dir", "results/sensor_metrics"))

    print(f"\n{'='*60}")
    print(f"Sweep : {sweep_name}")
    print(f"Env   : {base_env}")
    print(f"Params: {list(parameters.keys())}")
    print(f"Steps : {n_steps}  |  record every: {rec_every}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 2: load base environment config ──────────────────────────
    base_config = load_json(base_env)
    print(f"Loaded base config: {base_env}\n")

    # ── Steps 3 & 4: sweep each parameter independently ──────────────
    for param_path, values in parameters.items():
        print(f"\n{'─'*60}")
        print(f"Parameter : {param_path}")
        print(f"Values    : {values}")
        print(f"{'─'*60}")

        param_results  = []
        valid_values   = []

        for val in values:
            print(f"\n  [{param_path} = {val}]")

            patched_cfg  = apply_param(base_config, param_path, val)
            tmp_env_path = write_temp_config(patched_cfg)

            print(f"    Running benchmark ({n_steps} steps)...")
            try:
                # 1. Check if the parameter we are sweeping is gossip_rounds
                # If not, we'll default to 2 (or whatever you want the baseline to be)
                current_gossip = 2
                if param_path == "gossip_rounds":
                    current_gossip = val
                print(current_gossip)
                edge_sim, cent_sim, comparison = run_benchmark(
                    env_config_path=tmp_env_path,
                    n_steps=n_steps,
                    record_every=rec_every,
                    gossip_rounds=current_gossip, # 2. Pass it in!
                    seed=args.seed,
                )

                param_results.append((edge_sim, cent_sim, comparison))
                valid_values.append(val)

                e = comparison["edge"]
                c = comparison["centralized"]
                print(
                    f"    Edge    — "
                    f"violation={e['cumulative_comfort_violation']:.4f}  "
                    f"energy={e['cumulative_energy']:.4f}  "
                    f"overshoot={e['max_overshoot']:.4f}"
                )
                print(
                    f"    Central — "
                    f"violation={c['cumulative_comfort_violation']:.4f}  "
                    f"energy={c['cumulative_energy']:.4f}  "
                    f"overshoot={c['max_overshoot']:.4f}"
                )

            except Exception as exc:
                print(f"    ✗ FAILED: {exc}")

        if not param_results:
            print(f"  All runs failed for {param_path}, skipping plot.")
            continue

        # One plot file per parameter
        safe_name   = param_path.replace(".", "_")
        output_path = output_dir / f"{safe_name}.png"

        print(f"\n  Plotting {len(param_results)} results...")
        plot_parameter_sweep(
            param_name   = param_path,
            param_values = valid_values,
            results      = param_results,
            output_path  = output_path,
            z_slice      = args.z_slice,
        )

    print(f"\n{'='*60}")
    print(f"Done. Plots saved to: {output_dir}/")
    print(f"{'='*60}\n")

    all_results = {}  # add this before the parameter loop

    for param_path, values in parameters.items():
        all_results[param_path] = {}
        ...
        # inside the try block, after appending to param_results:
        all_results[param_path][val] = (edge_sim, cent_sim, comparison)

    # after the loop:
    save_summary_table(all_results, output_dir)


if __name__ == "__main__":
    main()