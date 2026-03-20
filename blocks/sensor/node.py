"""
SensorNode — individual sensor node with Kalman filter and gossip participation.

Each sensor node:
1. Reads concentration from the world at its grid position
2. Applies Gaussian noise to simulate imperfect sensing
3. Runs a Kalman filter to smooth readings and estimate dφ/dt
4. Computes spatial gradient from neighboring cells
5. Estimates flow velocity of the gas front
6. Generates and processes gossip messages for distributed prediction
"""

from __future__ import annotations

from typing import Any

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from .gossip import GossipMessage


class SensorNode:
    """A single sensor node in the VDPA network."""

    def __init__(
        self,
        name: str,
        position: tuple[int, int, int],
        dx: float,
        dt: float,
        sensor_sigma: float = 0.0,
        process_noise_var: float = 0.01,
        rng: np.random.Generator | None = None,
    ):
        self.name = name
        self.position = position
        self.dx = dx
        self.dt = dt
        self.sensor_sigma = sensor_sigma

        # ── Kalman filter ──
        # State: [concentration, d(concentration)/dt]
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0.0, 0.0])
        self.kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.kf.H = np.array([[1.0, 0.0]])
        self.kf.R = np.array([[max(sensor_sigma ** 2, 1e-10)]])
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_var)
        self.kf.P *= 10.0

        # ── Local state ──
        self.raw_reading: float = 0.0
        self.filtered_concentration: float = 0.0
        self.filtered_rate: float = 0.0       # dφ/dt from Kalman
        self.gradient: np.ndarray = np.zeros(3)
        self.velocity: np.ndarray = np.zeros(3)

        # ── Gossip state ──
        self.inbox: list[GossipMessage] = []
        self.predictions: dict[str, float] = {}  # origin -> arrival_time
        self.messages_sent: int = 0
        self.messages_received: int = 0

        # ── RNG ──
        self._rng = rng if rng is not None else np.random.default_rng()

    # ── Sensing ────────────────────────────────────────────────────────

    def sense(self, phi: np.ndarray) -> float:
        """Read concentration, add noise, update Kalman filter.

        Returns the Kalman-filtered concentration estimate.
        """
        true_value = float(phi[self.position])

        if self.sensor_sigma > 0:
            noise = self._rng.normal(0, self.sensor_sigma)
            self.raw_reading = max(0.0, true_value + noise)
        else:
            self.raw_reading = true_value

        self.kf.predict()
        self.kf.update(np.array([self.raw_reading]))

        self.filtered_concentration = float(self.kf.x[0])
        self.filtered_rate = float(self.kf.x[1])

        return self.filtered_concentration

    # ── Gradient and velocity ──────────────────────────────────────────

    def compute_gradient(self, phi: np.ndarray, walls: np.ndarray) -> np.ndarray:
        """Compute spatial gradient ∇φ using central differences.

        Uses the world phi field for finite differences at this node's
        position, respecting wall boundaries (Neumann: zero gradient at walls).
        """
        center_val = phi[self.position]
        grad = np.zeros(3)

        for axis in range(3):
            pos = self.position[axis]
            size = phi.shape[axis]

            # Forward neighbor
            fwd_idx = list(self.position)
            if pos + 1 < size:
                fwd_idx[axis] = pos + 1
                fwd_val = center_val if walls[tuple(fwd_idx)] else phi[tuple(fwd_idx)]
            else:
                fwd_val = center_val

            # Backward neighbor
            bwd_idx = list(self.position)
            if pos - 1 >= 0:
                bwd_idx[axis] = pos - 1
                bwd_val = center_val if walls[tuple(bwd_idx)] else phi[tuple(bwd_idx)]
            else:
                bwd_val = center_val

            grad[axis] = (fwd_val - bwd_val) / (2.0 * self.dx)

        self.gradient = grad
        return grad

    def compute_velocity(self) -> np.ndarray:
        """Estimate apparent flow velocity from temporal and spatial gradients.

        v̂ = -(dφ/dt) / |∇φ|² × ∇φ

        Derived from the advection equation for the concentration front:
        dφ/dt + v · ∇φ = 0.  When gradient is negligible, velocity is zero.
        """
        grad_mag_sq = float(np.dot(self.gradient, self.gradient))

        if grad_mag_sq < 1e-12:
            self.velocity = np.zeros(3)
        else:
            self.velocity = -(self.filtered_rate / grad_mag_sq) * self.gradient

        return self.velocity

    # ── Gossip ─────────────────────────────────────────────────────────

    def create_gossip_message(
        self, timestamp: float, detection_threshold: float = 0.01
    ) -> GossipMessage | None:
        """Create a gossip message if this node detects significant gas."""
        if self.filtered_concentration < detection_threshold:
            return None

        self.messages_sent += 1
        return GossipMessage(
            origin_node=self.name,
            origin_position=self.position,
            sender_node=self.name,
            timestamp=timestamp,
            concentration=self.filtered_concentration,
            gradient=self.gradient.copy(),
            velocity=self.velocity.copy(),
            hops=0,
        )

    def receive_gossip(self, message: GossipMessage) -> GossipMessage | None:
        """Process an incoming gossip message.

        Computes predicted arrival time from the message's origin.
        Returns a forwarded copy if the prediction is novel (earlier than
        any existing prediction from that origin), otherwise None.
        """
        self.messages_received += 1
        self.inbox.append(message)

        # Compute predicted arrival time at this node
        speed = float(np.linalg.norm(message.velocity))
        if speed < 1e-10:
            arrival = float("inf")
        else:
            displacement = (
                np.array(self.position, dtype=float)
                - np.array(message.origin_position, dtype=float)
            )
            distance = float(np.linalg.norm(displacement)) * self.dx

            # Only predict if gas is moving toward us
            if np.dot(message.velocity, displacement) > 1e-12:
                arrival = message.timestamp + distance / speed
            else:
                arrival = float("inf")

        # Update prediction table if this is earlier
        origin = message.origin_node
        existing = self.predictions.get(origin, float("inf"))
        if arrival < existing:
            self.predictions[origin] = arrival
            return message.forward(self.name)

        return None

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def earliest_predicted_arrival(self) -> float:
        """Earliest predicted arrival time from any gossip source."""
        if not self.predictions:
            return float("inf")
        return min(self.predictions.values())

    @property
    def concentration_uncertainty(self) -> float:
        """Kalman filter uncertainty (variance) on concentration estimate."""
        return float(self.kf.P[0, 0])

    def clear_inbox(self) -> None:
        """Clear the message inbox (call at start of each step)."""
        self.inbox.clear()

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Return node-level metrics for reporting."""
        return {
            "name": self.name,
            "position": self.position,
            "raw_reading": round(self.raw_reading, 6),
            "filtered_concentration": round(self.filtered_concentration, 6),
            "filtered_rate": round(self.filtered_rate, 6),
            "concentration_uncertainty": round(self.concentration_uncertainty, 6),
            "gradient_magnitude": round(float(np.linalg.norm(self.gradient)), 6),
            "velocity_magnitude": round(float(np.linalg.norm(self.velocity)), 6),
            "earliest_predicted_arrival": self.earliest_predicted_arrival,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "n_predictions": len(self.predictions),
        }
