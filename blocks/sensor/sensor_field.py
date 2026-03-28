"""
SensorField — collection of sensor nodes with gossip-based distributed consensus.

**Domain-agnostic**: This module performs pure inference. It receives scalar
readings from an external interface and orchestrates the three-layer pipeline:

  sense (rate-of-change monitoring)
    -> predict (TTI + urgency computation)
    -> gossip (multi-hop urgency propagation for neighborhood consensus)

The gossip protocol is bandwidth-efficient: nodes only transmit when they
detect a meaningful rate of change (|dV/dt| > talk_threshold). This makes
the network self-regulating — quiet areas stay quiet on the network.

The interface layer (blocks/interface/) is responsible for reading the physical
environment, feeding values here, and translating urgency back into actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .node import SensorNode
from .gossip import NegotiationMessage

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from .sensor_network import SensorNetwork


class SensorField:
    """Collection of sensor nodes with gossip-based distributed consensus."""

    def __init__(
        self,
        env: Environment,
        network: SensorNetwork,
        gossip_rounds: int = 1,
        talk_threshold: float = 0.01,
        max_hops: int = 10,
        buffer_seconds: float = 30.0,
        seed: int | None = None,
    ):
        self.env = env
        self.network = network
        self.gossip_rounds = gossip_rounds
        self.talk_threshold = talk_threshold
        self.max_hops = max_hops

        # Create reproducible RNGs for each node
        master_seq = np.random.SeedSequence(seed)
        child_seeds = master_seq.spawn(len(network.positions))

        # Determine per-node setpoint from zone mask
        self.nodes: dict[str, SensorNode] = {}
        for i, (name, pos) in enumerate(network.positions.items()):
            zone_name = str(env.zone_mask[pos])
            setpoint = env.rooms[zone_name].setpoint if zone_name in env.rooms else 25.0
            self.nodes[name] = SensorNode(
                name=name,
                position=pos,
                dt=env.dt,
                setpoint=setpoint,
                buffer_seconds=buffer_seconds,
                sensor_sigma=env.sensor_sigma,
                rng=np.random.default_rng(child_seeds[i]),
            )

        self._step_count = 0

    # ── Main loop ──────────────────────────────────────────────────────

    def step(self, readings: dict[str, float], timestamp: float) -> None:
        """Run one full sense -> predict -> gossip cycle.

        Parameters
        ----------
        readings : dict[str, float]
            Mapping of node_name -> scalar value, provided by the interface layer.
        timestamp : float
            Current simulation time.
        """
        # Clear inboxes from previous step
        for node in self.nodes.values():
            node.clear_inbox()

        # 1. Sense: feed each node its reading
        for name, value in readings.items():
            if name in self.nodes:
                self.nodes[name].sense(value, timestamp)

        # 2. Gossip propagation
        self._run_gossip(timestamp)

        self._step_count += 1

    def _run_gossip(self, timestamp: float) -> None:
        """Execute gossip propagation with configurable hop rounds.

        Round 0: nodes with |dV/dt| > talk_threshold generate messages.
        Round 1+: forwarded messages propagate one additional hop each round.
        """
        pending: dict[str, list[NegotiationMessage]] = {}

        for name, node in self.nodes.items():
            msg = node.create_negotiation_message(
                timestamp=timestamp,
                talk_threshold=self.talk_threshold,
            )
            if msg is not None:
                for neighbor in self.network.neighbors(name):
                    pending.setdefault(neighbor, []).append(msg)

        for _ in range(self.gossip_rounds):
            if not pending:
                break

            next_pending: dict[str, list[NegotiationMessage]] = {}

            for recipient, messages in pending.items():
                node = self.nodes[recipient]
                for msg in messages:
                    if msg.hops >= self.max_hops:
                        continue
                    forwarded = node.receive_negotiation(msg)
                    if forwarded is not None:
                        for neighbor in self.network.neighbors(recipient):
                            if neighbor != msg.sender_node:
                                next_pending.setdefault(neighbor, []).append(forwarded)

            pending = next_pending

    # ── Query helpers ──────────────────────────────────────────────────

    def get_urgencies(self) -> dict[str, float]:
        """Return {node_name: urgency} for all nodes."""
        return {name: node.urgency for name, node in self.nodes.items()}

    def get_ttis(self) -> dict[str, float]:
        """Return {node_name: TTI} for all nodes."""
        return {name: node.tti for name, node in self.nodes.items()}

    def get_alert_nodes(self, horizon: float = 30.0) -> list[str]:
        """Return names of nodes predicting setpoint breach within horizon seconds."""
        return [
            name for name, node in self.nodes.items()
            if node.tti < horizon
        ]

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Return aggregate metrics for reporting."""
        n_heating = sum(
            1 for n in self.nodes.values()
            if n.dT_dt > self.talk_threshold
        )
        n_with_urgency = sum(
            1 for n in self.nodes.values()
            if n.urgency > 0
        )

        finite_ttis = [
            n.tti for n in self.nodes.values()
            if n.tti < float("inf")
        ]

        total_sent = sum(n.messages_sent for n in self.nodes.values())
        total_received = sum(n.messages_received for n in self.nodes.values())

        result: dict[str, Any] = {
            "step": self._step_count,
            "n_nodes": len(self.nodes),
            "n_heating": n_heating,
            "n_with_urgency": n_with_urgency,
            "urgency_coverage": round(
                n_with_urgency / len(self.nodes), 4
            ) if self.nodes else 0.0,
            "total_messages_sent": total_sent,
            "total_messages_received": total_received,
            "mean_value": round(float(np.mean([
                n.filtered_value for n in self.nodes.values()
            ])), 4) if self.nodes else 0.0,
            "mean_dT_dt": round(float(np.mean([
                n.dT_dt for n in self.nodes.values()
            ])), 6) if self.nodes else 0.0,
        }

        if finite_ttis:
            result["min_tti"] = round(min(finite_ttis), 4)
            result["mean_tti"] = round(float(np.mean(finite_ttis)), 4)

        return result
