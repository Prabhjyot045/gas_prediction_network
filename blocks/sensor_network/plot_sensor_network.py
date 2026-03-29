"""
Sensor network visualization using SensorNetwork (NetworkX-based).

Plots:
- Environment (walls)
- Sensor nodes
- Communication edges (within radius)

Supports:
- Single z-slice
- Full graph projection
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sensor_network import SensorNetwork


def plot_network_slice(
    net: SensorNetwork,
    z: int | None = None,
    show_edges: bool = True,
) -> None:

    env = net.env

    if z is None:
        z = env.nz // 2

    if z < 0 or z >= env.nz:
        raise ValueError(f"Invalid z level: {z}")

    # --- walls ---
    wall_slice = env.walls[:, :, z]

    # 0 = floor, 1 = wall
    grid = np.zeros_like(wall_slice, dtype=int)
    grid[wall_slice] = 1

    fig, ax = plt.subplots(figsize=(6, 6))

    # 👇 custom colormap: 0=white (floor), 1=grey (walls)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["white", "grey"])

    ax.imshow(grid.T, origin="lower", cmap=cmap)

    # --- nodes ---
    pos2d = {}
    for n, data in net.graph.nodes(data=True):
        if data["z"] == z:
            pos2d[n] = (data["x"], data["y"])

            # 👇 sensors in RED
            ax.scatter(data["x"], data["y"], color="red", s=30)

    # --- edges ---
    if show_edges:
        for u, v in net.graph.edges():
            if u in pos2d and v in pos2d:
                x1, y1 = pos2d[u]
                x2, y2 = pos2d[v]

                #edges (light grey or black)
                ax.plot([x1, x2], [y1, y2], color="black", linewidth=1)

    ax.set_title(f"Sensor Network (z = {z})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.grid(True)

    plt.show()

def plot_network_projection(
    net: SensorNetwork,
    show_edges: bool = True,
) -> None:
    """
    Plot full 3D network projected onto XY plane.
    (All z-levels collapsed)
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6))

    # --- walls projection (any wall along z) ---
    walls = net.env.walls
    wall_proj = np.any(walls, axis=2)

    grid = np.zeros_like(wall_proj, dtype=int)
    grid[wall_proj] = 1

    # 👇 white floor, grey walls
    cmap = ListedColormap(["white", "grey"])
    ax.imshow(grid.T, origin="lower", cmap=cmap)

    # --- nodes ---
    pos2d = {}
    for n, data in net.graph.nodes(data=True):
        x, y = data["x"], data["y"]
        pos2d[n] = (x, y)

        # 👇 sensors in RED
        ax.scatter(x, y, color="red", s=30)

    # --- edges ---
    if show_edges:
        for u, v in net.graph.edges():
            x1, y1 = pos2d[u]
            x2, y2 = pos2d[v]

            # 👇 edges in black (or grey if you prefer)
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1, alpha=0.7)

    ax.set_title("Sensor Network (XY Projection)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.grid(False)  # cleaner look
    plt.show()
def plot_degree_histogram(net: SensorNetwork) -> None:
    """
    Plot degree distribution (topology insight).
    """

    hist = net.degree_distribution()

    degrees = list(hist.keys())
    counts = list(hist.values())

    fig, ax = plt.subplots()
    ax.bar(degrees, counts)

    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Degree Distribution")

    plt.show()