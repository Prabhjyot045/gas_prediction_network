"""
Sensor mesh topology — NetworkX graph construction and topology metrics.

SensorNetwork takes an Environment, places sensors via the configured
strategy, and builds a communication graph where edges connect nodes
within a given radius. Exposes rich metrics for academic reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

from .placement import place_sensors

if TYPE_CHECKING:
    from blocks.world.environment import Environment


class SensorNetwork:
    """Sensor mesh topology built from an Environment config."""

    def __init__(self, env: Environment, comm_radius: float | None = None):
        self.env = env
        self.comm_radius = comm_radius or env.sensor_config.get("communication_radius", 5.0)

        raw_positions = place_sensors(env)
        self.positions: dict[str, tuple[int, int, int]] = {
            name: pos for name, pos in raw_positions
        }

        self.graph: nx.Graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        """Create a NetworkX graph with edges for nodes within comm_radius."""
        G = nx.Graph()

        for name, pos in self.positions.items():
            G.add_node(name, pos=pos, x=pos[0], y=pos[1], z=pos[2])

        dx = self.env.dx
        names = list(self.positions.keys())
        for i, n1 in enumerate(names):
            p1 = np.array(self.positions[n1], dtype=float)
            for n2 in names[i + 1:]:
                p2 = np.array(self.positions[n2], dtype=float)
                dist = float(np.linalg.norm((p1 - p2) * dx))
                if dist <= self.comm_radius * dx:
                    G.add_edge(n1, n2, weight=dist)

        return G

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def node_positions_array(self) -> np.ndarray:
        """Return Nx3 array of sensor positions (for visualization)."""
        if not self.positions:
            return np.empty((0, 3), dtype=float)
        return np.array(list(self.positions.values()), dtype=float)

    def neighbors(self, name: str) -> list[str]:
        """Return the names of a node's graph neighbors."""
        return list(self.graph.neighbors(name))

    def is_connected(self) -> bool:
        if self.n_nodes == 0:
            return False
        return nx.is_connected(self.graph)

    def connected_components(self) -> int:
        if self.n_nodes == 0:
            return 0
        return nx.number_connected_components(self.graph)

    def degree_distribution(self) -> dict[int, int]:
        """Return {degree: count} histogram."""
        hist: dict[int, int] = {}
        for _, deg in self.graph.degree():
            hist[deg] = hist.get(deg, 0) + 1
        return dict(sorted(hist.items()))

    def average_degree(self) -> float:
        if self.n_nodes == 0:
            return 0.0
        return 2.0 * self.n_edges / self.n_nodes

    def diameter(self) -> float:
        """Graph diameter (longest shortest path). Returns inf if disconnected."""
        if self.n_nodes <= 1:
            return 0.0
        if not self.is_connected():
            return float("inf")
        return float(nx.diameter(self.graph))

    def average_path_length(self) -> float:
        """Average shortest path length. Returns inf if disconnected."""
        if self.n_nodes <= 1:
            return 0.0
        if not self.is_connected():
            return float("inf")
        return nx.average_shortest_path_length(self.graph)

    def clustering_coefficient(self) -> float:
        """Average clustering coefficient."""
        if self.n_nodes == 0:
            return 0.0
        return nx.average_clustering(self.graph)

    def coverage(self, sensing_radius: float | None = None) -> float:
        """Fraction of non-wall cells within sensing_radius of any sensor."""
        if sensing_radius is None:
            sensing_radius = self.comm_radius

        if self.n_nodes == 0:
            return 0.0

        walls = self.env.walls
        non_wall = np.argwhere(~walls).astype(float)
        if len(non_wall) == 0:
            return 0.0

        sensor_pos = self.node_positions_array()
        dx = self.env.dx

        covered = 0
        for cell in non_wall:
            dists = np.linalg.norm((sensor_pos - cell) * dx, axis=1)
            if np.min(dists) <= sensing_radius * dx:
                covered += 1

        return covered / len(non_wall)

    def metrics(self) -> dict[str, Any]:
        """Return all topology metrics as a flat dict for reporting."""
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "communication_radius": self.comm_radius,
            "is_connected": self.is_connected(),
            "connected_components": self.connected_components(),
            "average_degree": round(self.average_degree(), 2),
            "diameter": self.diameter(),
            "average_path_length": round(self.average_path_length(), 4),
            "clustering_coefficient": round(self.clustering_coefficient(), 4),
            "coverage": round(self.coverage(), 4),
            "degree_distribution": self.degree_distribution(),
        }
