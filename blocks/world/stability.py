"""
FTCS Stability Constraint for 3D Diffusion.

The Forward-Time Central-Space scheme for the diffusion equation is
conditionally stable. In 3D the constraint is:

    dt <= dx^2 / (6 * D)

The factor 6 arises from 3 spatial dimensions, each contributing a
second-order central-difference stencil with coefficient 2/dx^2.
"""


def compute_max_dt(dx: float, D: float) -> float:
    """Return the maximum stable time step for 3D FTCS diffusion."""
    if dx <= 0:
        raise ValueError(f"Grid spacing dx must be positive, got {dx}")
    if D <= 0:
        raise ValueError(f"Diffusion coefficient D must be positive, got {D}")
    return dx**2 / (6.0 * D)


def compute_stable_dt(dx: float, D: float, safety_factor: float = 0.4) -> float:
    """Return a safe time step: safety_factor * dt_max."""
    if not 0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    return safety_factor * compute_max_dt(dx, D)


def validate_dt(dt: float, dx: float, D: float) -> None:
    """Raise ValueError if dt exceeds the stability limit."""
    dt_max = compute_max_dt(dx, D)
    if dt > dt_max:
        raise ValueError(
            f"Time step dt={dt:.6f} exceeds the maximum stable value "
            f"dt_max={dt_max:.6f} for dx={dx}, D={D}. "
            f"Reduce dt or increase dx."
        )


def fourier_number(dt: float, dx: float, D: float) -> float:
    """Mesh Fourier number Fo = D * dt / dx^2. Must be <= 1/6 for stability."""
    return D * dt / dx**2
