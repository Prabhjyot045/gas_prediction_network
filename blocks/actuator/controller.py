"""
ActuatorController — door actuation logic with predictive and reactive policies.

Two policies are supported:
- **predictive** (VDPA): close a door when gossip-propagated predictions
  indicate gas will arrive at nearby sensors within a configurable horizon.
- **reactive** (centralized baseline): close a door when sensors near the door
  detect concentration above a threshold — i.e. gas has already arrived.

Both policies use the same SensorField and noise model so comparisons are fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World
    from blocks.sensor.sensor_field import SensorField


@dataclass
class Actuation:
    """Record of a single door actuation event."""
    door_name: str
    time: float
    step: int
    policy: str
    trigger_sensor: str
    trigger_value: float  # predicted arrival (predictive) or concentration (reactive)


class ActuatorController:
    """Manages door actuations based on sensor data.

    Maps each door to nearby sensors, then evaluates a policy each step
    to decide whether to close doors.
    """

    def __init__(
        self,
        env: Environment,
        sensor_field: SensorField,
        policy: str = "predictive",
        horizon: float = 5.0,
        concentration_threshold: float = 0.5,
        proximity_radius: float = 3.0,
    ):
        if policy not in ("predictive", "reactive"):
            raise ValueError(f"Unknown policy: {policy!r}. Use 'predictive' or 'reactive'.")

        self.env = env
        self.sensor_field = sensor_field
        self.policy = policy
        self.horizon = horizon
        self.concentration_threshold = concentration_threshold
        self.proximity_radius = proximity_radius

        # Map doors → nearby sensor names
        self.door_sensors: dict[str, list[str]] = self._map_doors_to_sensors()

        # Actuation log
        self.actuations: list[Actuation] = []
        self._first_detection_time: float | None = None
        self._first_actuation_time: float | None = None

    # ── Setup ──────────────────────────────────────────────────────────

    def _door_center(self, door) -> np.ndarray:
        """Compute the center position of a door's bounding box."""
        s = door.slices
        return np.array([
            (s[0].start + s[0].stop) / 2.0,
            (s[1].start + s[1].stop) / 2.0,
            (s[2].start + s[2].stop) / 2.0,
        ])

    def _map_doors_to_sensors(self) -> dict[str, list[str]]:
        """Find sensors within proximity_radius of each door's center."""
        mapping: dict[str, list[str]] = {}
        dx = self.env.dx

        for door_name, door in self.env.doors.items():
            center = self._door_center(door)
            nearby: list[str] = []

            for sensor_name, pos in self.sensor_field.network.positions.items():
                dist = float(np.linalg.norm(
                    (np.array(pos, dtype=float) - center) * dx
                ))
                if dist <= self.proximity_radius * dx:
                    nearby.append(sensor_name)

            mapping[door_name] = nearby

        return mapping

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, world: World) -> list[str]:
        """Evaluate policy and actuate doors if needed.

        Returns list of door names that were closed this step.
        """
        self._update_detection_time(world)

        if self.policy == "predictive":
            return self._evaluate_predictive(world)
        else:
            return self._evaluate_reactive(world)

    def _evaluate_predictive(self, world: World) -> list[str]:
        """VDPA: close doors when gossip predictions indicate gas approaching."""
        closed = []

        for door_name, sensors in self.door_sensors.items():
            if self.env.get_door_state(door_name) == "closed":
                continue

            for sensor_name in sensors:
                node = self.sensor_field.nodes[sensor_name]
                arrival = node.earliest_predicted_arrival
                if arrival < world.time + self.horizon:
                    world.close_door(door_name)

                    act = Actuation(
                        door_name=door_name,
                        time=world.time,
                        step=world.step_count,
                        policy="predictive",
                        trigger_sensor=sensor_name,
                        trigger_value=arrival,
                    )
                    self.actuations.append(act)
                    closed.append(door_name)

                    if self._first_actuation_time is None:
                        self._first_actuation_time = world.time
                    break

        return closed

    def _evaluate_reactive(self, world: World) -> list[str]:
        """Centralized reactive: close doors when nearby sensors detect gas."""
        closed = []

        for door_name, sensors in self.door_sensors.items():
            if self.env.get_door_state(door_name) == "closed":
                continue

            for sensor_name in sensors:
                node = self.sensor_field.nodes[sensor_name]
                if node.filtered_concentration > self.concentration_threshold:
                    world.close_door(door_name)

                    act = Actuation(
                        door_name=door_name,
                        time=world.time,
                        step=world.step_count,
                        policy="reactive",
                        trigger_sensor=sensor_name,
                        trigger_value=node.filtered_concentration,
                    )
                    self.actuations.append(act)
                    closed.append(door_name)

                    if self._first_actuation_time is None:
                        self._first_actuation_time = world.time
                    break

        return closed

    def _update_detection_time(self, world: World) -> None:
        """Track when gas is first detected by any sensor."""
        if self._first_detection_time is not None:
            return
        threshold = self.sensor_field.detection_threshold
        for node in self.sensor_field.nodes.values():
            if node.filtered_concentration >= threshold:
                self._first_detection_time = world.time
                return

    # ── Metrics ────────────────────────────────────────────────────────

    @property
    def doors_closed(self) -> int:
        """Number of unique doors that have been closed."""
        return len({a.door_name for a in self.actuations})

    @property
    def first_detection_time(self) -> float | None:
        return self._first_detection_time

    @property
    def first_actuation_time(self) -> float | None:
        return self._first_actuation_time

    @property
    def response_time(self) -> float | None:
        """Time between first detection and first actuation."""
        if self._first_detection_time is None or self._first_actuation_time is None:
            return None
        return self._first_actuation_time - self._first_detection_time

    def metrics(self) -> dict[str, Any]:
        """Return actuator metrics for reporting."""
        return {
            "policy": self.policy,
            "doors_closed": self.doors_closed,
            "total_actuations": len(self.actuations),
            "first_detection_time": self._first_detection_time,
            "first_actuation_time": self._first_actuation_time,
            "response_time": self.response_time,
            "actuation_log": [
                {
                    "door": a.door_name,
                    "time": round(a.time, 6),
                    "step": a.step,
                    "trigger_sensor": a.trigger_sensor,
                    "trigger_value": round(a.trigger_value, 6),
                }
                for a in self.actuations
            ],
        }
