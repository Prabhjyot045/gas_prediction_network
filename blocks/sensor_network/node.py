"""
SensorNode — individual edge inference node with rolling buffer and TTI.

**Domain-agnostic**: This module knows NOTHING about the physical environment.
Nodes receive scalar (timestamp, value) pairs from an external interface and
perform pure inference:

  Layer 1 (Sensing): Rate-of-change monitoring.
    - Rolling buffer of readings with least-squares slope estimation (dV/dt).
    - Works for any time-series metric (temperature, CO2, humidity, power, etc.).

  Layer 2 (Prediction): Time-To-Impact estimation.
    - TTI = (threshold - current) / (rate of change)
    - Converts raw trends into actionable "urgency" scores (urgency = 1/TTI).

  Layer 3 (Communication): Gossip-based consensus.
    - Urgency propagates to neighbors via NegotiationMessages.
    - Bandwidth-efficient: only gossip when |dV/dt| > talk_threshold.

The interface layer (blocks/interface/) is responsible for reading the physical
environment and feeding values to these nodes, then translating urgency
decisions back into actuator commands.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .gossip import NegotiationMessage


class RollingBuffer:
    """Fixed-size circular buffer storing (timestamp, value) pairs.

    Supports least-squares slope estimation for computing dT/dt.
    """

    def __init__(self, max_samples: int):
        self.max_samples = max(2, max_samples)
        self._times: deque[float] = deque(maxlen=self.max_samples)
        self._values: deque[float] = deque(maxlen=self.max_samples)

    def append(self, t: float, value: float) -> None:
        self._times.append(t)
        self._values.append(value)

    @property
    def size(self) -> int:
        return len(self._times)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_samples

    def slope(self) -> float:
        """Compute dT/dt via least-squares linear fit.

        Returns 0.0 if fewer than 2 samples.
        """
        n = self.size
        if n < 2:
            return 0.0

        times = np.array(self._times)
        values = np.array(self._values)

        t_mean = np.mean(times)
        v_mean = np.mean(values)
        dt = times - t_mean
        dv = values - v_mean
        denom = np.dot(dt, dt)
        if denom < 1e-15:
            return 0.0
        return float(np.dot(dt, dv) / denom)

    @property
    def latest_value(self) -> float:
        if not self._values:
            return 0.0
        return self._values[-1]


class SensorNode:
    """A single edge inference node in the Aether-Edge network."""

    def __init__(
        self,
        name: str,
        position: tuple[int, int, int],
        dt: float,
        setpoint: float = 22.0,
        buffer_seconds: float = 30.0,
        sensor_sigma: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.name = name
        self.position = position
        self.dt = dt
        self.setpoint = setpoint
        self.sensor_sigma = sensor_sigma

        max_samples = max(2, int(buffer_seconds / dt))
        self.buffer = RollingBuffer(max_samples)

        self.raw_reading: float = 0.0
        self.filtered_value: float = 0.0
        self.dT_dt: float = 0.0

        self.inbox: list[NegotiationMessage] = []
        self.neighbor_urgencies: dict[str, float] = {}
        self.messages_sent: int = 0
        self.messages_received: int = 0

        self._rng = rng if rng is not None else np.random.default_rng()

    def sense(self, value: float, timestamp: float) -> None:
        """Process a scalar reading, add noise, update rolling buffer."""
        if self.sensor_sigma > 0:
            noise = self._rng.normal(0, self.sensor_sigma)
            self.raw_reading = value + noise
        else:
            self.raw_reading = value

        self.buffer.append(timestamp, self.raw_reading)
        self.filtered_value = self.raw_reading
        self.dT_dt = self.buffer.slope()

    @property
    def tti(self) -> float:
        """Time-To-Impact: seconds until setpoint breach at current rate."""
        if self.dT_dt <= 1e-10:
            return float("inf")

        gap = self.setpoint - self.filtered_value
        if gap <= 0:
            return 0.0

        return gap / self.dT_dt

    @property
    def urgency(self) -> float:
        """Urgency = 1/TTI. Higher means more urgent need for cooling."""
        t = self.tti
        if t <= 0:
            return float("inf")
        if t == float("inf"):
            return 0.0
        return 1.0 / t

    def create_negotiation_message(
        self, timestamp: float, talk_threshold: float = 0.01
    ) -> NegotiationMessage | None:
        """Create a negotiation message if value is changing significantly."""
        if abs(self.dT_dt) < talk_threshold:
            return None

        self.messages_sent += 1
        return NegotiationMessage(
            origin_node=self.name,
            origin_position=self.position,
            sender_node=self.name,
            timestamp=timestamp,
            value=self.filtered_value,
            dT_dt=self.dT_dt,
            urgency=self.urgency,
            hops=0,
        )

    def receive_negotiation(self, message: NegotiationMessage) -> NegotiationMessage | None:
        """Process an incoming negotiation message."""
        self.messages_received += 1
        self.inbox.append(message)

        origin = message.origin_node
        existing = self.neighbor_urgencies.get(origin, 0.0)
        if message.urgency > existing:
            self.neighbor_urgencies[origin] = message.urgency
            return message.forward(self.name)

        return None

    @property
    def max_neighbor_urgency(self) -> float:
        """Maximum urgency reported by any neighbor (including self)."""
        own = self.urgency
        if not self.neighbor_urgencies:
            return own
        return max(own, max(self.neighbor_urgencies.values()))

    @property
    def total_known_urgency(self) -> float:
        """Sum of all known urgencies (self + neighbors) for proportional allocation."""
        total = self.urgency
        total += sum(self.neighbor_urgencies.values())
        return total

    def clear_inbox(self) -> None:
        """Clear the message inbox (call at start of each step)."""
        self.inbox.clear()

    def metrics(self) -> dict[str, Any]:
        """Return node-level metrics for reporting."""
        return {
            "name": self.name,
            "position": self.position,
            "raw_reading": round(self.raw_reading, 4),
            "filtered_value": round(self.filtered_value, 4),
            "dT_dt": round(self.dT_dt, 6),
            "tti": self.tti if self.tti != float("inf") else None,
            "urgency": round(self.urgency, 6) if self.urgency != float("inf") else None,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "n_neighbor_urgencies": len(self.neighbor_urgencies),
            "buffer_fill": self.buffer.size,
        }
