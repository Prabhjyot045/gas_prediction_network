"""
FTCS Stability Constraints for 3D Heat Diffusion + Advection.

Two constraints apply:
1. Diffusion: dt <= dx^2 / (6 * alpha)   [3D FTCS, factor 6 for 3 axes]
2. Advection (CFL): dt <= dx / |v_max|    [Courant-Friedrichs-Lewy]

The stricter of the two determines the maximum stable time step.
"""


def compute_max_dt_diffusion(dx: float, alpha: float) -> float:
    """Maximum stable dt for 3D FTCS diffusion."""
    if dx <= 0:
        raise ValueError(f"Grid spacing dx must be positive, got {dx}")
    if alpha <= 0:
        raise ValueError(f"Thermal diffusivity alpha must be positive, got {alpha}")
    return dx**2 / (6.0 * alpha)


def compute_max_dt_advection(dx: float, v_max: float) -> float:
    """Maximum stable dt for CFL advection condition."""
    if dx <= 0:
        raise ValueError(f"Grid spacing dx must be positive, got {dx}")
    if v_max <= 0:
        return float("inf")  # No advection → no CFL constraint
    return dx / v_max


def compute_stable_dt(
    dx: float, alpha: float, v_max: float = 0.0, safety_factor: float = 0.4
) -> float:
    """Return a safe time step satisfying both diffusion and CFL constraints."""
    if not 0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    dt_diff = compute_max_dt_diffusion(dx, alpha)
    dt_cfl = compute_max_dt_advection(dx, v_max)
    return safety_factor * min(dt_diff, dt_cfl)


def validate_dt(dx: float, alpha: float, dt: float, v_max: float = 0.0) -> None:
    """Raise ValueError if dt exceeds either stability limit."""
    dt_diff = compute_max_dt_diffusion(dx, alpha)
    if dt > dt_diff:
        raise ValueError(
            f"dt={dt:.6f} exceeds diffusion limit dt_max={dt_diff:.6f} "
            f"for dx={dx}, alpha={alpha}."
        )
    dt_cfl = compute_max_dt_advection(dx, v_max)
    if dt > dt_cfl:
        raise ValueError(
            f"dt={dt:.6f} exceeds CFL limit dt_max={dt_cfl:.6f} "
            f"for dx={dx}, v_max={v_max}."
        )


def fourier_number(dt: float, dx: float, alpha: float) -> float:
    """Mesh Fourier number Fo = alpha * dt / dx^2. Must be <= 1/6 for stability."""
    return alpha * dt / dx**2
