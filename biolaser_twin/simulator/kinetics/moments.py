"""Moment extraction, biological targets (q1, q2), and reduced moment-ODE reference."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .mechanisms import MechanismSpec
from .population_balance import PBETrajectory, solve_pbe


@dataclass(frozen=True)
class KineticSummaryTargets:
    """Primary biological summary targets derived from an aggregation trajectory."""

    q1_oligomer_burden: float  # (1 / (T * m0)) * int_0^T q * O(t) dt
    q2_peak_time_norm: float  # t_peak / T
    t_peak_min: float
    o_peak_um: float


def compute_moment_from_bins(
    fibril_bins_um: np.ndarray,
    pivots_ell: np.ndarray,
    order: int,
) -> np.ndarray:
    """Compute the k-th discrete size moment M_k(t) = sum_i ell_i^k c_i(t)."""
    return fibril_bins_um @ (pivots_ell**order)


def extract_kinetic_targets(
    traj: PBETrajectory,
    include_off_pathway: bool = True,
) -> KineticSummaryTargets:
    """Extract normalized oligomer burden q1 and normalized oligomer peak time q2.

    Normalizes q1 by initial monomer m0 (falling back to c_total_um if m0 == 0).
    """
    t = traj.times_min
    t_span = float(t[-1] - t[0])
    if t_span <= 0.0:
        raise ValueError("Trajectory time span must be positive.")

    o_total = traj.oligomer_um + (traj.off_pathway_um if include_off_pathway else 0.0)
    o_mass = traj.q_oligomer * o_total
    integral_o = float(np.trapezoid(o_mass, t) if hasattr(np, "trapezoid") else np.trapz(o_mass, t))
    m0_ref = float(traj.monomer_um[0]) if traj.monomer_um[0] > 1e-12 else float(traj.c_total_um)
    q1 = integral_o / (t_span * max(m0_ref, 1e-12))

    peak_idx = int(np.argmax(o_total))
    t_peak = float(t[peak_idx])
    q2 = (t_peak - float(t[0])) / t_span

    return KineticSummaryTargets(
        q1_oligomer_burden=q1,
        q2_peak_time_norm=q2,
        t_peak_min=t_peak,
        o_peak_um=float(o_total[peak_idx]),
    )


def extract_truth_kinetic_targets(
    spec: MechanismSpec,
    t_end_min: float,
    t_start_min: float = 0.0,
    n_dense: int = 501,
    include_off_pathway: bool = True,
    **solve_kwargs,
) -> KineticSummaryTargets:
    """Compute cadence-independent truth (q1, q2) on a dense internal time grid."""
    dense_times = np.linspace(float(t_start_min), float(t_end_min), max(101, int(n_dense)))
    dense_traj = solve_pbe(spec, dense_times, **solve_kwargs)
    return extract_kinetic_targets(dense_traj, include_off_pathway=include_off_pathway)


def solve_reduced_moments_length_independent(
    spec: MechanismSpec,
    times_min: np.ndarray,
    m0_um: float | None = None,
    o0_um: float = 0.0,
    p0_um: float = 0.0,
    m_mass0_um: float = 0.0,
) -> dict[str, np.ndarray]:
    """Solve exact closed 4-moment ODEs (m, O, P, M) when fragmentation has no q-cutoff penalty.

    Used in verification tests to compare against the full PBE in the large-fibril limit (ell >> 2q)
    where boundary bond loss (2q - 1) * P is negligible relative to M.
    """
    t_eval = np.asarray(times_min, dtype=float)
    q = spec.q_oligomer
    m_init = spec.c_total_um if m0_um is None else float(m0_um)
    y0 = np.array([m_init, float(o0_um), float(p0_um), float(m_mass0_um)], dtype=float)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        m, o, p, mass = y
        m_pos = max(0.0, m)
        mass_pos = max(0.0, mass)
        j_n = spec.k_n * (m_pos**spec.n_c)
        sat = 1.0 + ((m_pos / spec.k_m_sat) ** spec.n_2 if np.isfinite(spec.k_m_sat) else 0.0)
        j_2 = spec.k_2 * mass_pos * (m_pos**spec.n_2) / sat if spec.k_2 > 0.0 else 0.0

        dm = -q * (j_n + j_2) + q * spec.k_d * o - 2.0 * spec.k_plus * m_pos * p
        do = j_n + j_2 - (spec.k_c + spec.k_d) * o
        dp = spec.k_c * o + spec.k_f * (mass_pos - (2.0 * q - 1.0) * max(0.0, p))
        dmass = 2.0 * spec.k_plus * m_pos * p + q * spec.k_c * o
        return np.array([dm, do, dp, dmass], dtype=float)

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), y0, method="Radau", t_eval=t_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"Reduced moment ODE solve failed: {sol.message}")
    return {
        "times_min": t_eval,
        "m": sol.y[0],
        "O": sol.y[1],
        "P": sol.y[2],
        "M": sol.y[3],
    }
