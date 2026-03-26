"""
SensorNode — individual edge inference node with rolling buffer and TTI.

Each sensor node:
1. Reads temperature from the world at its grid position
2. Applies Gaussian noise to simulate imperfect sensing
3. Maintains a 30-second rolling buffer for local trend analysis
4. Computes dT/dt via least-squares fit on the buffer
5. Computes Time-To-Impact (TTI): time until setpoint breach
6. Computes spatial gradient for flow direction inference
7. Generates and processes negotiation messages for distributed consensus
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

        # Least squares: slope = Σ(t-t̄)(v-v̄) / Σ(t-t̄)²
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
        dx: float,
        dt: float,
        setpoint: float = 22.0,
        buffer_seconds: float = 30.0,
        sensor_sigma: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.name = name
        self.position = position
        self.dx = dx
        self.dt = dt
        self.setpoint = setpoint
        self.sensor_sigma = sensor_sigma

        # ── Rolling buffer ──
        max_samples = max(2, int(buffer_seconds / dt))
        self.buffer = RollingBuffer(max_samples)

        # ── Local state ──
        self.raw_reading: float = 0.0
        self.filtered_temperature: float = 0.0
        self.dT_dt: float = 0.0
        self.gradient: np.ndarray = np.zeros(3)

        # ── Gossip state ──
        self.inbox: list[NegotiationMessage] = []
        self.neighbor_urgencies: dict[str, float] = {}  # origin -> urgency
        self.messages_sent: int = 0
        self.messages_received: int = 0

        # ── RNG ──
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── Sensing ────────────────────────────────────────────────────────

    def sense(self, T: np.ndarray, timestamp: float) -> float:
        """Read temperature, add noise, update rolling buffer.

        Returns the (noisy) temperature reading.
        """
        true_value = float(T[self.position])

        if self.sensor_sigma > 0:
            noise = self._rng.normal(0, self.sensor_sigma)
            self.raw_reading = true_value + noise
        else:
            self.raw_reading = true_value

        self.buffer.append(timestamp, self.raw_reading)
        self.filtered_temperature = self.raw_reading
        self.dT_dt = self.buffer.slope()

        return self.filtered_temperature

    # ── TTI (Time-To-Impact) ──────────────────────────────────────────

    @property
    def tti(self) -> float:
        """Time-To-Impact: seconds until setpoint breach at current rate.

        Returns inf if temperature is falling, stable, or already below setpoint
        with no upward trend.
        """
        if self.dT_dt <= 1e-10:
            return float("inf")

        gap = self.setpoint - self.filtered_temperature
        if gap <= 0:
            # Already above setpoint
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

    # ── Gradient ──────────────────────────────────────────────────────

    def compute_gradient(self, T: np.ndarray, walls: np.ndarray) -> np.ndarray:
        """Compute spatial gradient nabla(T) using central differences.

        Respects wall boundaries (Neumann: zero gradient at walls).
        """
        center_val = T[self.position]
        grad = np.zeros(3)

        for axis in range(3):
            pos = self.position[axis]
            size = T.shape[axis]

            fwd_idx = list(self.position)
            if pos + 1 < size:
                fwd_idx[axis] = pos + 1
                fwd_val = center_val if walls[tuple(fwd_idx)] else T[tuple(fwd_idx)]
            else:
                fwd_val = center_val

            bwd_idx = list(self.position)
            if pos - 1 >= 0:
                bwd_idx[axis] = pos - 1
                bwd_val = center_val if walls[tuple(bwd_idx)] else T[tuple(bwd_idx)]
            else:
                bwd_val = center_val

            grad[axis] = (fwd_val - bwd_val) / (2.0 * self.dx)

        self.gradient = grad
        return grad

    # ── Gossip ─────────────────────────────────────────────────────────

    def create_negotiation_message(
        self, timestamp: float, talk_threshold: float = 0.01
    ) -> NegotiationMessage | None:
        """Create a negotiation message if temperature is changing significantly.

        Only gossip if |dT/dt| > talk_threshold (saves bandwidth).
        """
        if abs(self.dT_dt) < talk_threshold:
            return None

        self.messages_sent += 1
        return NegotiationMessage(
            origin_node=self.name,
            origin_position=self.position,
            sender_node=self.name,
            timestamp=timestamp,
            temperature=self.filtered_temperature,
            dT_dt=self.dT_dt,
            gradient=self.gradient.copy(),
            urgency=self.urgency,
            hops=0,
        )

    def receive_negotiation(self, message: NegotiationMessage) -> NegotiationMessage | None:
        """Process an incoming negotiation message.

        Updates neighbor urgency table. Returns a forwarded copy if the
        urgency from this origin is higher than previously known.
        """
        self.messages_received += 1
        self.inbox.append(message)

        origin = message.origin_node
        existing = self.neighbor_urgencies.get(origin, 0.0)
        if message.urgency > existing:
            self.neighbor_urgencies[origin] = message.urgency
            return message.forward(self.name)

        return None

    # ── Properties ─────────────────────────────────────────────────────

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

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Return node-level metrics for reporting."""
        return {
            "name": self.name,
            "position": self.position,
            "raw_reading": round(self.raw_reading, 4),
            "filtered_temperature": round(self.filtered_temperature, 4),
            "dT_dt": round(self.dT_dt, 6),
            "tti": self.tti if self.tti != float("inf") else None,
            "urgency": round(self.urgency, 6) if self.urgency != float("inf") else None,
            "gradient_magnitude": round(float(np.linalg.norm(self.gradient)), 6),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "n_neighbor_urgencies": len(self.neighbor_urgencies),
            "buffer_fill": self.buffer.size,
        }
