"""
NegotiationMessage — inter-node gossip message for distributed consensus.

Nodes share their urgency (1/TTI) and local state with neighbors so the
mesh can converge on a fair resource allocation without a central coordinator.

Domain-agnostic: messages carry value and rate-of-change, not HVAC-specific data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NegotiationMessage:
    """A gossip message carrying urgency and local state."""
    origin_node: str
    origin_position: tuple[int, int, int]
    sender_node: str
    timestamp: float
    value: float              # current reading at origin
    dT_dt: float              # rate of change
    urgency: float            # 1/TTI — higher means more urgent
    hops: int = 0

    def forward(self, relay_node: str) -> NegotiationMessage:
        """Create a forwarded copy with incremented hop count."""
        return NegotiationMessage(
            origin_node=self.origin_node,
            origin_position=self.origin_position,
            sender_node=relay_node,
            timestamp=self.timestamp,
            value=self.value,
            dT_dt=self.dT_dt,
            urgency=self.urgency,
            hops=self.hops + 1,
        )
