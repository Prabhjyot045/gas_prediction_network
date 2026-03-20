"""
MetricsCollector — accumulates structured metrics from simulation runs.

Each block pushes metrics via record(). Results can be exported to JSON,
CSV, or pandas DataFrame for reporting.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


class MetricsCollector:
    """Accumulates metrics from simulation runs for reporting."""

    def __init__(self, name: str = "experiment"):
        self.name = name
        self._records: list[dict[str, Any]] = []
        self._scalars: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._metadata: dict[str, Any] = {}
        self._start_time = time.time()

    # ── Recording ─────────────────────────────────────────────────────────

    def set_metadata(self, **kwargs: Any) -> None:
        """Set experiment-level metadata (config params, description, etc.)."""
        self._metadata.update(kwargs)

    def record(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append a metrics snapshot. Auto-adds step if provided."""
        entry = dict(metrics)
        if step is not None:
            entry.setdefault("step", step)
        self._records.append(entry)

    def record_scalar(self, name: str, value: float, step: int) -> None:
        """Record a single named scalar at a given step (time-series)."""
        self._scalars[name].append((step, value))

    # ── Access ────────────────────────────────────────────────────────────

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def scalar_series(self, name: str) -> tuple[list[int], list[float]]:
        """Return (steps, values) for a named scalar time-series."""
        if name not in self._scalars:
            return [], []
        pairs = self._scalars[name]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def scalar_names(self) -> list[str]:
        """Return all recorded scalar names."""
        return list(self._scalars.keys())

    def latest(self) -> dict[str, Any]:
        """Return the most recent record."""
        return self._records[-1] if self._records else {}

    # ── Export ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return all collected data as a JSON-serializable dict."""
        return {
            "name": self.name,
            "metadata": self._metadata,
            "elapsed_seconds": round(time.time() - self._start_time, 2),
            "n_records": len(self._records),
            "records": self._records,
            "scalars": {
                name: {"steps": s, "values": v}
                for name, (s, v) in (
                    (n, self.scalar_series(n)) for n in self._scalars
                )
            },
        }

    def save_json(self, path: str | Path) -> Path:
        """Save all data to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2, default=str)
        return path

    def save_csv(self, path: str | Path) -> Path:
        """Save scalar time-series to CSV (step as first column)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all steps across all scalars
        all_steps = set()
        for name in self._scalars:
            for step, _ in self._scalars[name]:
                all_steps.add(step)
        steps = sorted(all_steps)

        # Build lookup
        scalar_data: dict[str, dict[int, float]] = {}
        for name in self._scalars:
            scalar_data[name] = {s: v for s, v in self._scalars[name]}

        names = sorted(self._scalars.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + names)
            for step in steps:
                row = [step] + [scalar_data.get(n, {}).get(step, "") for n in names]
                writer.writerow(row)
        return path

    def to_dataframe(self) -> Any:
        """Convert records to a pandas DataFrame. Requires pandas."""
        import pandas as pd
        return pd.DataFrame(self._records)
