"""
DamperController — VAV damper actuation with edge and centralized policies.

Two policies:
- **edge** (Aether-Edge): Each damper opening is computed locally from
  gossip-propagated urgencies. Proportional allocation: A_i = u_i / sum(u).
  Zero network delay for local decisions.
- **centralized**: A central controller polls all sensors, computes allocation,
  and pushes commands back. Simulated polling interval, jitter, and compute
  delay cause stale data (Age of Information).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World
    from blocks.sensor.sensor_field import SensorField


@dataclass
class DamperAction:
    """Record of a damper adjustment."""
    damper_name: str
    time: float
    step: int
    opening: float
    policy: str
    age_of_information: float  # staleness of data used for this decision


class DamperController:
    """Manages VAV damper openings based on sensor data.

    Maps each damper to nearby sensors, then evaluates a policy each step
    to compute continuous damper openings A ∈ [0, 1].
    """

    def __init__(
        self,
        env: Environment,
        sensor_field: SensorField,
        policy: str = "edge",
        proximity_radius: float = 3.0,
        # Centralized delay parameters
        polling_interval: float = 5.0,
        jitter_sigma: float = 0.5,
        compute_delay: float = 1.0,
        seed: int | None = None,
    ):
        if policy not in ("edge", "centralized"):
            raise ValueError(f"Unknown policy: {policy!r}. Use 'edge' or 'centralized'.")

        self.env = env
        self.sensor_field = sensor_field
        self.policy = policy
        self.proximity_radius = proximity_radius

        # Centralized delay model
        self.polling_interval = polling_interval
        self.jitter_sigma = jitter_sigma
        self.compute_delay = compute_delay
        self._rng = np.random.default_rng(seed)
        self._last_poll_time: float = -float("inf")
        self._cached_openings: dict[str, float] = {}

        # Map dampers → nearby sensor names
        self.damper_sensors: dict[str, list[str]] = self._map_dampers_to_sensors()

        # Tracking
        self.actions: list[DamperAction] = []
        self._total_aoi: float = 0.0
        self._total_decisions: int = 0
        self._total_energy: float = 0.0

    # ── Setup ──────────────────────────────────────────────────────────

    def _damper_center(self, damper) -> np.ndarray:
        return np.array(damper.position, dtype=float)

    def _map_dampers_to_sensors(self) -> dict[str, list[str]]:
        """Find sensors within proximity_radius of each damper."""
        mapping: dict[str, list[str]] = {}
        dx = self.env.dx

        for damper_name, damper in self.env.dampers.items():
            center = self._damper_center(damper)
            nearby: list[str] = []

            for sensor_name, pos in self.sensor_field.network.positions.items():
                dist = float(np.linalg.norm(
                    (np.array(pos, dtype=float) - center) * dx
                ))
                if dist <= self.proximity_radius * dx:
                    nearby.append(sensor_name)

            mapping[damper_name] = nearby

        return mapping

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, world: World) -> dict[str, float]:
        """Evaluate policy and set damper openings.

        Returns dict of {damper_name: new_opening}.
        """
        if self.policy == "edge":
            openings = self._evaluate_edge(world)
        else:
            openings = self._evaluate_centralized(world)

        # Apply openings to world
        for name, opening in openings.items():
            world.set_damper(name, opening)

        # Track energy
        self._total_energy += world.total_cooling_energy()

        return openings

    def _evaluate_edge(self, world: World) -> dict[str, float]:
        """Edge policy: urgency-weighted proportional allocation.

        Each damper's opening = local_urgency / total_known_urgency.
        Uses gossip-propagated urgencies — zero delay.
        """
        openings: dict[str, float] = {}

        # Collect urgency per damper from nearby sensors
        damper_urgencies: dict[str, float] = {}
        for damper_name, sensors in self.damper_sensors.items():
            if not sensors:
                damper_urgencies[damper_name] = 0.0
                continue
            # Use max urgency from nearby sensors (including their neighbor knowledge)
            max_urg = 0.0
            for s_name in sensors:
                node = self.sensor_field.nodes[s_name]
                max_urg = max(max_urg, node.urgency)
                # Also consider neighbor urgencies propagated via gossip
                for u in node.neighbor_urgencies.values():
                    max_urg = max(max_urg, u)
            damper_urgencies[damper_name] = max_urg

        # Proportional allocation
        total_urgency = sum(damper_urgencies.values())
        for damper_name, urg in damper_urgencies.items():
            if total_urgency > 1e-10:
                opening = urg / total_urgency
            else:
                # No urgency anywhere — minimal opening
                opening = 0.1
            openings[damper_name] = max(0.0, min(1.0, opening))

            self.actions.append(DamperAction(
                damper_name=damper_name,
                time=world.time,
                step=world.step_count,
                opening=openings[damper_name],
                policy="edge",
                age_of_information=0.0,
            ))
            self._total_decisions += 1

        return openings

    def _evaluate_centralized(self, world: World) -> dict[str, float]:
        """Centralized policy: poll → compute → push with simulated delays.

        Only updates damper settings at polling_interval. Between polls,
        dampers hold their previous positions. Data is stale by
        polling_interval + jitter + compute_delay.
        """
        time_since_poll = world.time - self._last_poll_time
        effective_interval = self.polling_interval + self._rng.normal(0, self.jitter_sigma)
        effective_interval = max(0.1, effective_interval)

        if time_since_poll < effective_interval and self._cached_openings:
            # Reuse cached openings (no new poll yet)
            return self._cached_openings

        # New poll cycle
        aoi = time_since_poll + self.compute_delay
        self._last_poll_time = world.time

        # Same allocation logic but using stale data
        damper_urgencies: dict[str, float] = {}
        for damper_name, sensors in self.damper_sensors.items():
            if not sensors:
                damper_urgencies[damper_name] = 0.0
                continue
            max_urg = 0.0
            for s_name in sensors:
                node = self.sensor_field.nodes[s_name]
                max_urg = max(max_urg, node.urgency)
            damper_urgencies[damper_name] = max_urg

        total_urgency = sum(damper_urgencies.values())
        openings: dict[str, float] = {}
        for damper_name, urg in damper_urgencies.items():
            if total_urgency > 1e-10:
                opening = urg / total_urgency
            else:
                opening = 0.1
            openings[damper_name] = max(0.0, min(1.0, opening))

            self.actions.append(DamperAction(
                damper_name=damper_name,
                time=world.time,
                step=world.step_count,
                opening=openings[damper_name],
                policy="centralized",
                age_of_information=aoi,
            ))
            self._total_aoi += aoi
            self._total_decisions += 1

        self._cached_openings = openings
        return openings

    # ── Metrics ────────────────────────────────────────────────────────

    @property
    def mean_age_of_information(self) -> float:
        """Average AoI across all decisions."""
        if self._total_decisions == 0:
            return 0.0
        return self._total_aoi / self._total_decisions

    @property
    def total_energy(self) -> float:
        return self._total_energy

    @property
    def total_messages(self) -> int:
        """Total gossip messages (only meaningful for edge policy)."""
        return sum(n.messages_sent for n in self.sensor_field.nodes.values())

    def metrics(self) -> dict[str, Any]:
        """Return actuator metrics for reporting."""
        current_openings = {
            name: round(d.opening, 4)
            for name, d in self.env.dampers.items()
        }
        return {
            "policy": self.policy,
            "damper_openings": current_openings,
            "total_actions": len(self.actions),
            "total_energy": round(self._total_energy, 6),
            "mean_age_of_information": round(self.mean_age_of_information, 4),
            "total_gossip_messages": self.total_messages,
        }
