"""Independent nominal 4-moment kinetic ODEs for inverse solvers (strictly isolated from simulator/)."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class NominalKineticParams:
    """Nominal lumped kinetic parameters exposed to inverse solvers and identifiability.

    Intentionally uses a reduced, non-identical parameterization relative to simulator.kinetics.MechanismSpec
    to enforce the anti-inverse-crime boundary.
    """

    k_n_eff: float = 1e-8
    k_plus_eff: float = 0.03
    k_2_eff: float = 5e-7
    k_c_eff: float = 0.1
    k_d_eff: float = 0.02
    q_nom: int = 4
    c_total_um: float = 50.0


def integrate_nominal_moment_ode(
    times_min: np.ndarray,
    params: NominalKineticParams | None = None,
    m0_um: float | None = None,
    o0_um: float = 0.0,
    p0_um: float = 0.0,
    mass0_um: float = 0.0,
    t0_min: float = 0.0,
) -> dict[str, np.ndarray]:
    """Integrate the nominal 4-moment ODE system (m, O, P, M) from t0_min and evaluate at times_min."""
    p = params or NominalKineticParams()
    t_obs = np.asarray(times_min, dtype=float)
    if t_obs.ndim != 1 or len(t_obs) < 1 or np.any(t_obs < t0_min) or np.any(np.diff(t_obs) <= 0.0):
        raise ValueError("times_min must be strictly increasing and >= t0_min.")
    m_init = p.c_total_um if m0_um is None else float(m0_um)
    y0 = np.array([m_init, float(o0_um), float(p0_um), float(mass0_um)], dtype=float)
    t_all = t_obs if abs(t_obs[0] - t0_min) < 1e-12 else np.concatenate([[t0_min], t_obs])

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        m, o, p_num, mass = y
        m_pos = m if m > 0.0 else 0.0
        mass_pos = mass if mass > 0.0 else 0.0
        j_n = p.k_n_eff * (m_pos**2)
        j_2 = p.k_2_eff * mass_pos * (m_pos**2)
        dm = -p.q_nom * (j_n + j_2) + p.q_nom * p.k_d_eff * o - 2.0 * p.k_plus_eff * m_pos * p_num
        do = j_n + j_2 - (p.k_c_eff + p.k_d_eff) * o
        dp = p.k_c_eff * o
        dmass = 2.0 * p.k_plus_eff * m_pos * p_num + p.q_nom * p.k_c_eff * o
        return np.array([dm, do, dp, dmass], dtype=float)

    sol = solve_ivp(rhs, (float(t0_min), float(t_all[-1])), y0, method="Radau", t_eval=t_all, rtol=1e-8, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"Nominal moment ODE failed: {sol.message}")
    start = 0 if len(t_all) == len(t_obs) else 1
    return {
        "times_min": t_obs,
        "m": sol.y[0, start:],
        "O": sol.y[1, start:],
        "P": sol.y[2, start:],
        "M": sol.y[3, start:],
    }


def nominal_moment_ode_residual(
    times_min: np.ndarray,
    m: np.ndarray,
    o: np.ndarray,
    p_num: np.ndarray,
    mass: np.ndarray,
    params: NominalKineticParams,
) -> np.ndarray:
    """Compute time-derivative collocation residuals R_dyn for the nominal 4-moment ODE."""
    t = np.asarray(times_min, dtype=float)
    dm_dt = np.gradient(m, t)
    do_dt = np.gradient(o, t)
    dp_dt = np.gradient(p_num, t)
    dmass_dt = np.gradient(mass, t)

    m_pos = np.maximum(m, 0.0)
    mass_pos = np.maximum(mass, 0.0)
    j_n = params.k_n_eff * (m_pos**2)
    j_2 = params.k_2_eff * mass_pos * (m_pos**2)

    r_m = dm_dt - (-params.q_nom * (j_n + j_2) + params.q_nom * params.k_d_eff * o - 2.0 * params.k_plus_eff * m_pos * p_num)
    r_o = do_dt - (j_n + j_2 - (params.k_c_eff + params.k_d_eff) * o)
    r_p = dp_dt - (params.k_c_eff * o)
    r_mass = dmass_dt - (2.0 * params.k_plus_eff * m_pos * p_num + params.q_nom * params.k_c_eff * o)
    return np.stack([r_m, r_o, r_p, r_mass], axis=1)
