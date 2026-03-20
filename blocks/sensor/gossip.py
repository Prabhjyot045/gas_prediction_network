"""
GossipMessage — data structure for inter-node gossip communication.

The gossip protocol enables distributed prediction: when a node detects
gas, it broadcasts gradient and velocity estimates to neighbors. These
propagate through the network so distant nodes can actuate preemptively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GossipMessage:
    """A gossip message carrying gas detection and prediction data.

    Messages originate at nodes that detect gas above a threshold, then
    relay hop-by-hop through the sensor network. Each receiving node
    uses the velocity estimate to predict when gas will arrive at its
    own position.
    """

    origin_node: str                        # node that originally detected gas
    origin_position: tuple[int, int, int]   # grid position of origin
    sender_node: str                        # node relaying this message
    timestamp: float                        # world time of detection
    concentration: float                    # concentration at origin
    gradient: np.ndarray                    # spatial gradient at origin (3,)
    velocity: np.ndarray                    # estimated flow velocity at origin (3,)
    hops: int = 0                           # relay hop count

    def forward(self, relay_node: str) -> GossipMessage:
        """Create a forwarded copy for relay to the next hop."""
        return GossipMessage(
            origin_node=self.origin_node,
            origin_position=self.origin_position,
            sender_node=relay_node,
            timestamp=self.timestamp,
            concentration=self.concentration,
            gradient=self.gradient.copy(),
            velocity=self.velocity.copy(),
            hops=self.hops + 1,
        )
