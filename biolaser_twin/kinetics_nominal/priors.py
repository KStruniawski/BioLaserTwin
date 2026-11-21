"""Alternative kinetic priors (Finke-Watzky, Avrami, monotone-O, smoothness) for R_dyn ablations."""

import numpy as np


def finke_watzky_trajectory(
    times_min: np.ndarray,
    c_total_um: float = 50.0,
    k1_per_min: float = 1e-4,
    k2_per_um_min: float = 1e-3,
) -> np.ndarray:
    """Closed-form Finke-Watzky 2-step nucleation-autocatalytic growth mass B(t) in uM."""
    t = np.asarray(times_min, dtype=float)
    a0 = float(c_total_um)
    if k2_per_um_min <= 1e-15:
        return np.clip(a0 * (1.0 - np.exp(-k1_per_min * np.maximum(0.0, t))), 0.0, a0)
    num = k1_per_min / k2_per_um_min + a0
    exp_term = np.exp(np.minimum((k1_per_min + k2_per_um_min * a0) * t, 500.0))
    b_t = a0 - num / (1.0 + (k1_per_min / (k2_per_um_min * a0)) * exp_term)
    return np.clip(b_t, 0.0, a0)


def finke_watzky_residual(
    times_min: np.ndarray,
    b_mass_um: np.ndarray,
    c_total_um: float,
    k1_per_min: float,
    k2_per_um_min: float,
) -> np.ndarray:
    """ODE residual dB/dt - (k1*(A0 - B) + k2*B*(A0 - B)) for Finke-Watzky R_dyn ablation."""
    t = np.asarray(times_min, dtype=float)
    db_dt = np.gradient(b_mass_um, t)
    rem = c_total_um - b_mass_um
    rhs = (k1_per_min + k2_per_um_min * b_mass_um) * rem
    return db_dt - rhs


def avrami_trajectory(
    times_min: np.ndarray,
    c_total_um: float = 50.0,
    k_avrami: float = 0.02,
    n_avrami: float = 3.0,
) -> np.ndarray:
    """Avrami sigmoidal crystallization/aggregation mass M(t) = C_total * (1 - exp(-(k*t)^n))."""
    t = np.maximum(0.0, np.asarray(times_min, dtype=float))
    return float(c_total_um) * (1.0 - np.exp(-((k_avrami * t) ** n_avrami)))


def monotone_oligomer_penalty(o_traj: np.ndarray) -> np.ndarray:
    """Misspecified monotone-O(t) prior penalty: penalizes negative dO/dt (for ablation studies)."""
    do = np.diff(o_traj)
    return np.maximum(0.0, -do)


def smoothness_residual(times_min: np.ndarray, y_traj: np.ndarray) -> np.ndarray:
    """Second-order temporal smoothness residual d^2y / dt^2."""
    t = np.asarray(times_min, dtype=float)
    dy_dt = np.gradient(y_traj, t, axis=0)
    return np.gradient(dy_dt, t, axis=0)
