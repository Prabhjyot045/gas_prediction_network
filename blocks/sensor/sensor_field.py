"""
SensorField — manages all sensor nodes and coordinates gossip rounds.

SensorField is the high-level interface for Block 4. It:
1. Creates SensorNode instances from the SensorNetwork topology
2. Runs the sense → filter → gossip cycle each simulation step
3. Supports multi-hop gossip propagation (configurable rounds per step)
4. Exposes aggregate metrics for reporting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .node import SensorNode
from .gossip import GossipMessage

if TYPE_CHECKING:
    from blocks.world.environment import Environment
    from blocks.world.world import World
    from blocks.network.sensor_network import SensorNetwork


class SensorField:
    """Collection of sensor nodes with gossip-based distributed prediction."""

    def __init__(
        self,
        env: Environment,
        network: SensorNetwork,
        gossip_rounds: int = 1,
        detection_threshold: float = 0.01,
        max_hops: int = 10,
        process_noise_var: float = 0.01,
        seed: int | None = None,
    ):
        self.env = env
        self.network = network
        self.gossip_rounds = gossip_rounds
        self.detection_threshold = detection_threshold
        self.max_hops = max_hops

        # Create reproducible RNGs for each node
        master_seq = np.random.SeedSequence(seed)
        child_seeds = master_seq.spawn(len(network.positions))

        self.nodes: dict[str, SensorNode] = {}
        for i, (name, pos) in enumerate(network.positions.items()):
            self.nodes[name] = SensorNode(
                name=name,
                position=pos,
                dx=env.dx,
                dt=env.dt,
                sensor_sigma=env.sensor_sigma,
                process_noise_var=process_noise_var,
                rng=np.random.default_rng(child_seeds[i]),
            )

        self._step_count = 0

    # ── Main loop ──────────────────────────────────────────────────────

    def step(self, world: World) -> None:
        """Run one full sense → filter → gradient → gossip cycle."""
        phi = world.phi
        walls = self.env.walls
        timestamp = world.time

        # Clear inboxes from previous step
        for node in self.nodes.values():
            node.clear_inbox()

        # 1. Sense: all nodes read concentration + Kalman update
        for node in self.nodes.values():
            node.sense(phi)

        # 2. Compute gradients and velocities
        for node in self.nodes.values():
            node.compute_gradient(phi, walls)
            node.compute_velocity()

        # 3. Gossip propagation
        self._run_gossip(timestamp)

        self._step_count += 1

    def _run_gossip(self, timestamp: float) -> None:
        """Execute gossip propagation with configurable hop rounds.

        Round 0: detecting nodes generate messages → deliver to 1-hop neighbors.
        Round 1+: forwarded messages propagate one additional hop each round.
        """
        # Generate initial messages from all detecting nodes
        pending: dict[str, list[GossipMessage]] = {}

        for name, node in self.nodes.items():
            msg = node.create_gossip_message(
                timestamp=timestamp,
                detection_threshold=self.detection_threshold,
            )
            if msg is not None:
                for neighbor in self.network.neighbors(name):
                    pending.setdefault(neighbor, []).append(msg)

        # Run gossip rounds
        for _ in range(self.gossip_rounds):
            if not pending:
                break

            next_pending: dict[str, list[GossipMessage]] = {}

            for recipient, messages in pending.items():
                node = self.nodes[recipient]
                for msg in messages:
                    if msg.hops >= self.max_hops:
                        continue
                    forwarded = node.receive_gossip(msg)
                    if forwarded is not None:
                        for neighbor in self.network.neighbors(recipient):
                            if neighbor != msg.sender_node:
                                next_pending.setdefault(neighbor, []).append(forwarded)

            pending = next_pending

    # ── Query helpers ──────────────────────────────────────────────────

    def get_predicted_arrivals(self) -> dict[str, float]:
        """Return {node_name: earliest_predicted_arrival} for all nodes."""
        return {
            name: node.earliest_predicted_arrival
            for name, node in self.nodes.items()
        }

    def get_alert_nodes(self, current_time: float, horizon: float) -> list[str]:
        """Return names of nodes predicting gas arrival within `horizon` seconds.

        These are the nodes that would trigger actuators in Block 5.
        """
        alerts = []
        for name, node in self.nodes.items():
            arrival = node.earliest_predicted_arrival
            if arrival < current_time + horizon:
                alerts.append(name)
        return alerts

    def concentration_rmse(self, world: World) -> float:
        """RMSE between Kalman-filtered estimates and true concentration."""
        errors_sq = []
        for node in self.nodes.values():
            true_val = float(world.phi[node.position])
            est_val = node.filtered_concentration
            errors_sq.append((true_val - est_val) ** 2)

        if not errors_sq:
            return 0.0
        return float(np.sqrt(np.mean(errors_sq)))

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics(self, world: World | None = None) -> dict[str, Any]:
        """Return aggregate metrics for reporting."""
        n_detecting = sum(
            1 for n in self.nodes.values()
            if n.filtered_concentration >= self.detection_threshold
        )
        n_with_predictions = sum(
            1 for n in self.nodes.values()
            if n.predictions
        )

        finite_arrivals = [
            n.earliest_predicted_arrival
            for n in self.nodes.values()
            if n.earliest_predicted_arrival < float("inf")
        ]

        total_sent = sum(n.messages_sent for n in self.nodes.values())
        total_received = sum(n.messages_received for n in self.nodes.values())

        result: dict[str, Any] = {
            "step": self._step_count,
            "n_nodes": len(self.nodes),
            "n_detecting": n_detecting,
            "n_with_predictions": n_with_predictions,
            "prediction_coverage": round(
                n_with_predictions / len(self.nodes), 4
            ) if self.nodes else 0.0,
            "total_messages_sent": total_sent,
            "total_messages_received": total_received,
            "mean_filtered_concentration": round(float(np.mean([
                n.filtered_concentration for n in self.nodes.values()
            ])), 6) if self.nodes else 0.0,
            "mean_velocity_magnitude": round(float(np.mean([
                np.linalg.norm(n.velocity) for n in self.nodes.values()
            ])), 6) if self.nodes else 0.0,
        }

        if finite_arrivals:
            result["earliest_global_arrival"] = round(min(finite_arrivals), 6)
            result["mean_predicted_arrival"] = round(float(np.mean(finite_arrivals)), 6)

        if world is not None:
            result["concentration_rmse"] = round(self.concentration_rmse(world), 6)

        return result
