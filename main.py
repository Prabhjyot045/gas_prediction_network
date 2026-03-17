import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
nx_dim, ny_dim = 100, 100
nt         = 1000
PLOT_EVERY = 10

# Lower tau → slower diffusion → plume stays coherent over long distances
# D = (tau - 0.5) / 3  →  tau=0.8 gives D≈0.10  (was 0.43 at tau=1.8)
tau = 0.8

# D2Q9
c = np.array([
    [0,  0], [1,  0], [0,  1], [-1, 0], [0, -1],
    [1,  1], [-1, 1], [-1,-1], [ 1,-1]
], dtype=int)
w        = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# =============================================================================
# CHANNEL GEOMETRY
# =============================================================================
walls = np.ones((nx_dim, ny_dim), dtype=bool)

H_ROW_LO, H_ROW_HI = 44, 57   # horizontal arm rows (open)
H_COL_LO, H_COL_HI =  2, 73   # horizontal arm cols
V_ROW_LO, V_ROW_HI = 15, 57   # vertical arm rows  (open)
V_COL_LO, V_COL_HI = 58, 73   # vertical arm cols

walls[H_ROW_LO:H_ROW_HI, H_COL_LO:H_COL_HI] = False
walls[V_ROW_LO:V_ROW_HI, V_COL_LO:V_COL_HI] = False
walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True

wall_idx = np.where(walls)

# =============================================================================
# DRIFT VELOCITY FIELD
# =============================================================================
# Without advection the gas only diffuses isotropically and dilutes before
# reaching the bend.  We add a gentle bulk drift that:
#   • pushes rightward  (+col direction) in the horizontal arm
#   • pushes upward     (-row direction) in the vertical arm
#
# In D2Q9 the equilibrium for a passive scalar WITH bulk velocity u=(vr,vc) is:
#   f_eq_i = w_i * C * (1 + 3*(c_i[0]*vr + c_i[1]*vc))
#
# vr = drift in row direction, vc = drift in col direction.
# Keep |u| << 1/√3 ≈ 0.577 (lattice speed of sound) for stability.

DRIFT = 0.18   # lattice units — strong enough to carry the plume to the bend

# Build per-cell velocity maps (vr=row-drift, vc=col-drift)
vr = np.zeros((nx_dim, ny_dim))   # row velocity (positive = downward in array)
vc = np.zeros((nx_dim, ny_dim))   # col velocity (positive = rightward)

# Horizontal arm: drift rightward
vc[H_ROW_LO:H_ROW_HI, H_COL_LO:V_COL_LO] = DRIFT

# Transition / corner region: blend from rightward to upward
# rows 44-57, cols 58-73 is where both arms overlap
corner_r = slice(H_ROW_LO, H_ROW_HI)
corner_c = slice(V_COL_LO,  V_COL_HI)
vc[corner_r, corner_c] =  DRIFT * 0.3   # residual rightward at corner
vr[corner_r, corner_c] = -DRIFT * 0.85  # strong upward at corner

# Vertical arm: drift upward (negative row direction in numpy)
vr[V_ROW_LO:H_ROW_LO, V_COL_LO:V_COL_HI] = -DRIFT

# Zero velocity inside walls (safety)
vr[walls] = 0.0
vc[walls] = 0.0

# =============================================================================
# SENSOR MESH
# =============================================================================
sensor_spacing = 7
G = nx.Graph()

for x in range(3, nx_dim, sensor_spacing):
    for y in range(3, ny_dim, sensor_spacing):
        if not walls[x, y]:
            G.add_node((x, y))

for n1 in G.nodes:
    for n2 in G.nodes:
        if n1 >= n2:
            continue
        if np.hypot(n1[0]-n2[0], n1[1]-n2[1]) <= sensor_spacing * 1.5:
            G.add_edge(n1, n2)

pos = {n: (n[1], n[0]) for n in G.nodes}

# =============================================================================
# INITIALISE LBM
# =============================================================================
f = np.zeros((9, nx_dim, ny_dim))

# =============================================================================
# MAIN LOOP
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 9))

for it in range(nt):

    # Inject gas at left entrance — shorter burst, drift carries it forward
    if it < 300:
        f[0, H_ROW_LO+1:H_ROW_HI-1, 3:9] += 3.5

    C = f.sum(axis=0)
    C[wall_idx] = 0.0

    # ── VECTORIZED BGK Collision WITH drift velocity ──────────────────────────
    #
    #  Standard passive-scalar equilibrium (no velocity):
    #    f_eq_i = w_i * C
    #
    #  With bulk drift u = (vr, vc) at each cell:
    #    f_eq_i = w_i * C * (1 + 3*(c_i[0]*vr + c_i[1]*vc))
    #
    #  The extra term biases the equilibrium in the drift direction,
    #  so after relaxation mass is nudged that way each step — advection.
    #
    #  c[:, 0] and c[:, 1] are shape (9,); vr and vc are (nx, ny).
    #  Reshape to (9,1,1) for broadcasting across the full grid.

    cu = (c[:, 0, None, None] * vr[None, :, :]    # (9,nx,ny)
        + c[:, 1, None, None] * vc[None, :, :])

    f_eq = w[:, None, None] * C[None, :, :] * (1.0 + 3.0 * cu)
    f   -= (1.0 / tau) * (f - f_eq)

    f_old = f.copy()

    # Streaming
    for i in range(9):
        f[i] = np.roll(np.roll(f[i], c[i, 0], axis=0), c[i, 1], axis=1)

    # Bounce-back
    for i in range(9):
        f[i][wall_idx] = f_old[opposite[i]][wall_idx]

    # ── Render ───────────────────────────────────────────────────────────────
    if it % PLOT_EVERY == 0:
        ax.cla()

        ax.imshow(C, cmap='magma', origin='lower', vmin=0, vmax=10,
                  interpolation='bilinear', aspect='auto')

        wall_vis = np.ma.masked_where(~walls, np.ones_like(walls, dtype=float))
        ax.imshow(wall_vis, cmap='Greys', origin='lower',
                  alpha=0.85, vmin=0, vmax=1, aspect='auto')

        # Per-node local gradient
        node_colors, Q_X, Q_Y, Q_U, Q_V = [], [], [], [], []

        for node in G.nodes:
            rx, cy_n = node
            c_local = float(C[rx, cy_n])
            node_colors.append(c_local)

            gx = gy = 0.0
            for nbr in G.neighbors(node):
                nx_c, ny_c = nbr
                dx, dy = nx_c - rx, ny_c - cy_n
                dist = np.hypot(dx, dy)
                if dist > 0:
                    wg = (C[nx_c, ny_c] - c_local) / dist
                    gx += wg * dx
                    gy += wg * dy

            if c_local > 0.3:
                Q_X.append(cy_n); Q_Y.append(rx)
                Q_U.append(gy);   Q_V.append(gx)

        nx.draw(G, pos, ax=ax,
                node_color=node_colors, cmap='magma',
                node_size=45, edge_color='white', width=0.4,
                vmin=0, vmax=10, with_labels=False, alpha=0.7)

        if Q_X:
            ax.quiver(Q_X, Q_Y, Q_U, Q_V,
                      color='cyan', scale=60, headwidth=4,
                      headlength=6, alpha=0.9)

        for txt, xy, col in [
            ('GAS\nSOURCE', (5,  H_ROW_LO+6), 'lime'),
            ('BEND',        (65, 43),           'yellow'),
            ('EXIT',        (65, V_ROW_LO+3),   'orange'),
        ]:
            ax.annotate(txt, xy=xy, color=col, fontsize=9, fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

        D = (tau - 0.5) / 3
        ax.set_title(
            f'ECE659 – LBM L-Channel  |  τ={tau}  D={D:.3f}  drift={DRIFT}  |  '
            f'Step {it}/{nt}',
            fontsize=10)
        ax.set_xlabel('Y (column)')
        ax.set_ylabel('X (row)')
        plt.pause(0.001)

plt.show()