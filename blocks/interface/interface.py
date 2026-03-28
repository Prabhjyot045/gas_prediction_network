"""
EnvironmentInterface — boundary between physical environment and inference network.

This is the I/O layer that separates what the sensor network knows (scalar
values and urgency) from the physical world (temperature fields, vent dampers).

**Input side**: Reads the World's temperature field at sensor positions and
feeds scalar (timestamp, value) pairs to the SensorField.

**Output side**: Reads urgency from the SensorField and translates it into
vent routing commands — redistributing fixed airflow from low-urgency rooms
to high-urgency rooms.

Supports two policies:
- **edge**: Decisions are local, based on gossip-propagated urgencies.
  Age of Information = 0.
- **centralized**: A simulated central controller polls sensors with
  network delay. Same allocation algorithm, but data is stale.
  Age of Information = polling_interval + jitter + compute_delay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World
    from blocks.sensor.sensor_field import SensorField
    from blocks.sensor.sensor_network import SensorNetwork


@dataclass
class VentAction:
    """Record of a vent adjustment."""
    damper_name: str
    time: float
    step: int
    opening: float
    policy: str
    age_of_information: float  # staleness of data used for this decision


class EnvironmentInterface:
    """Boundary between physical environment and the inference network.

    Reads World.T at sensor positions → feeds values to SensorField.
    Reads urgency from SensorField → routes airflow to vents → applies to World.
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
        self._last_poll_time: float = 0.0
        self._cached_openings: dict[str, float] = {}

        # Map dampers → nearby sensor names
        self.damper_sensors: dict[str, list[str]] = self._map_dampers_to_sensors()

        # Tracking
        self.actions: list[VentAction] = []
        self._total_aoi: float = 0.0
        self._total_decisions: int = 0
        self._total_energy: float = 0.0

    # ── Setup ──────────────────────────────────────────────────────────

    def _map_dampers_to_sensors(self) -> dict[str, list[str]]:
        """Find sensors within proximity_radius of each damper."""
        mapping: dict[str, list[str]] = {}
        dx = self.env.dx

        for damper_name, damper in self.env.dampers.items():
            center = np.array(damper.position, dtype=float)
            nearby: list[str] = []

            for sensor_name, pos in self.sensor_field.network.positions.items():
                dist = float(np.linalg.norm(
                    (np.array(pos, dtype=float) - center) * dx
                ))
                if dist <= self.proximity_radius * dx:
                    nearby.append(sensor_name)

            mapping[damper_name] = nearby

        return mapping

    # ── Input: Read environment → feed sensor network ─────────────────

    def read_sensors(self, world: World) -> dict[str, float]:
        """Read temperature at each sensor node's position from the World."""
        readings: dict[str, float] = {}
        for name, node in self.sensor_field.nodes.items():
            readings[name] = float(world.T[node.position])
        return readings

    # ── Full step: read → infer → actuate ─────────────────────────────

    def step(self, world: World) -> dict[str, float]:
        """Full interface cycle: read environment → feed sensors → actuate vents.

        Returns dict of {damper_name: new_opening}.
        """
        # 1. Read environment → scalar values
        readings = self.read_sensors(world)
        timestamp = world.time

        # 2. Feed to sensor network (pure inference — no World coupling)
        self.sensor_field.step(readings, timestamp)

        # 3. Translate urgency → vent routing → apply to World
        if self.policy == "edge":
            openings = self._evaluate_edge(world)
        else:
            openings = self._evaluate_centralized(world)

        # 4. Apply vent openings to environment
        for name, opening in openings.items():
            world.set_damper(name, opening)

        # Track energy
        self._total_energy += world.total_cooling_energy()

        return openings

    # ── Output: Vent routing policies ─────────────────────────────────

    def _evaluate_edge(self, world: World) -> dict[str, float]:
        """Edge policy: urgency-weighted proportional allocation.

        Each vent's opening = local_urgency / total_known_urgency.
        Uses gossip-propagated urgencies — zero delay, AoI = 0.

        Rooms with no urgency (empty/cool) get minimal airflow.
        Rooms with high urgency get proportionally more of the fixed budget.
        """
        openings: dict[str, float] = {}

        for damper_name, sensors in self.damper_sensors.items():
            if not sensors:
                openings[damper_name] = 0.1
                continue
            
            # Local urgency is solely based on the sensors physically controlling this damper
            local_urg = 0.0
            known_urgencies: dict[str, float] = {}
            
            for s_name in sensors:
                node = self.sensor_field.nodes[s_name]
                local_urg = max(local_urg, node.urgency)
                
                # Collect all global gossip this cluster of sensors heard
                for origin, u in node.neighbor_urgencies.items():
                    known_urgencies[origin] = max(known_urgencies.get(origin, 0.0), u)
                    
            # A damper's perceived global urgency is its own + what it heard from others
            total_urgency = local_urg + sum(known_urgencies.values())

            if total_urgency > 1e-10:
                opening = local_urg / total_urgency
                # Cap the opening if the absolute urgency is low to prevent overcooling
                opening = min(opening, local_urg * 20.0)
            else:
                opening = 0.1

            openings[damper_name] = max(0.0, min(1.0, opening))

            self.actions.append(VentAction(
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

        Only updates vent settings at polling_interval. Between polls,
        vents hold their previous positions. Data is stale by
        polling_interval + jitter + compute_delay.
        """
        time_since_poll = world.time - self._last_poll_time
        effective_interval = self.polling_interval + self._rng.normal(0, self.jitter_sigma)
        effective_interval = max(0.1, effective_interval)

        if time_since_poll < effective_interval and self._cached_openings:
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
                # Cap the opening if the absolute urgency is low
                opening = min(opening, urg * 20.0)
            else:
                opening = 0.1
            openings[damper_name] = max(0.0, min(1.0, opening))

            self.actions.append(VentAction(
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

    def temperature_rmse(self, world: World) -> float:
        """RMSE between noisy sensor readings and true temperature."""
        errors_sq = []
        for node in self.sensor_field.nodes.values():
            true_val = float(world.T[node.position])
            est_val = node.filtered_value
            errors_sq.append((true_val - est_val) ** 2)

        if not errors_sq:
            return 0.0
        return float(np.sqrt(np.mean(errors_sq)))

    def metrics(self) -> dict[str, Any]:
        """Return interface metrics for reporting."""
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
