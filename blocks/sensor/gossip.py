"""
NegotiationMessage — inter-node gossip message for HVAC consensus.

Nodes share their thermal urgency (1/TTI) and local state with neighbors
so the mesh can converge on a fair cooling allocation without a central
coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NegotiationMessage:
    """A gossip message carrying thermal urgency and local state."""
    origin_node: str
    origin_position: tuple[int, int, int]
    sender_node: str
    timestamp: float
    temperature: float
    dT_dt: float          # rate of temperature change
    gradient: np.ndarray  # spatial gradient direction
    urgency: float        # 1/TTI — higher means more urgent
    hops: int = 0

    def forward(self, relay_node: str) -> NegotiationMessage:
        """Create a forwarded copy with incremented hop count."""
        return NegotiationMessage(
            origin_node=self.origin_node,
            origin_position=self.origin_position,
            sender_node=relay_node,
            timestamp=self.timestamp,
            temperature=self.temperature,
            dT_dt=self.dT_dt,
            gradient=self.gradient.copy(),
            urgency=self.urgency,
            hops=self.hops + 1,
        )
