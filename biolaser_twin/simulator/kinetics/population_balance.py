"""Kumar-Ramkrishna fixed-pivot log-binned Population Balance Equation (PBE) solver."""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp

from .mechanisms import MechanismSpec, effective_rate_multiplier


@dataclass(frozen=True)
class PBETrajectory:
    """Full population-balance trajectory with exact stoichiometric mass conservation."""

    times_min: np.ndarray
    pivots_ell: np.ndarray
    monomer_um: np.ndarray
    oligomer_um: np.ndarray
    off_pathway_um: np.ndarray
    fibril_bins_um: np.ndarray  # shape: (n_times, n_bins)
    p_number_um: np.ndarray
    m_mass_um: np.ndarray
    m2_second_moment_um: np.ndarray
    mean_length: np.ndarray
    c_total_um: float
    max_rel_mass_error: float
    q_oligomer: int
    max_upper_boundary_mass_frac: float = 0.0


def build_log_pivot_grid(
    q_min: int = 4,
    l_max: float = 1e5,
    n_discrete: int = 96,
    n_log_bins: int = 224,
) -> np.ndarray:
    """Construct a hybrid discrete + Kumar-Ramkrishna log-binned pivot grid for ell >= q_min."""
    if q_min < 2:
        raise ValueError("q_min must be >= 2.")
    disc = np.arange(q_min, q_min + max(1, n_discrete), dtype=float)
    start_log = disc[-1] + 1.0
    if l_max <= start_log:
        return disc
    log_pts = np.geomspace(start_log, float(l_max), max(2, n_log_bins))
    pivots = list(disc)
    for pt in log_pts:
        if pt - pivots[-1] >= 1.0:
            pivots.append(float(pt))
    return np.asarray(pivots, dtype=float)


def build_conservative_fragmentation_matrix(
    pivots: np.ndarray,
    q_min: int,
) -> np.ndarray:
    """Build linear fragmentation matrix F such that sum_i F[i, j] * x_i == 0 (exact mass conservation).

    For x_j <= 256, sums over all discrete integer scission sites u in [q_min, floor(x_j - q_min)].
    For x_j > 256, integrates the uniform daughter density analytically across every target bin
    [x_i, x_{i+1}] so no small-length bins are skipped and both number and mass are exact.
    """
    n = len(pivots)
    mat = np.zeros((n, n), dtype=float)
    for j in range(n):
        xj = pivots[j]
        if xj < 2.0 * q_min:
            continue
        n_bonds = xj - 2.0 * q_min + 1.0
        mat[j, j] -= n_bonds

        if xj <= 256.0:
            u_vals = np.arange(q_min, int(np.floor(xj - q_min)) + 1, dtype=float)
            w = n_bonds / max(1, len(u_vals))
            for u in u_vals:
                for daughter in (u, xj - u):
                    idx = max(0, min(n - 2, int(np.searchsorted(pivots, daughter, side="right")) - 1))
                    x_lo, x_hi = pivots[idx], pivots[idx + 1]
                    alpha = (daughter - x_lo) / (x_hi - x_lo)
                    mat[idx, j] += w * (1.0 - alpha)
                    mat[idx + 1, j] += w * alpha
        else:
            # Uniform daughter density = 2 * n_bonds / (xj - 2*q_min) over [q_min, xj - q_min]
            u_min = float(q_min)
            u_max = float(xj - q_min)
            density = 2.0 * n_bonds / (u_max - u_min)
            for i in range(n - 1):
                x_lo, x_hi = pivots[i], pivots[i + 1]
                a = max(u_min, x_lo)
                b = min(u_max, x_hi)
                if b <= a:
                    continue
                span = b - a
                alpha_a = (a - x_lo) / (x_hi - x_lo)
                alpha_b = (b - x_lo) / (x_hi - x_lo)
                alpha_mid = 0.5 * (alpha_a + alpha_b)
                mass_in_seg = density * span
                mat[i, j] += mass_in_seg * (1.0 - alpha_mid)
                mat[i + 1, j] += mass_in_seg * alpha_mid

        # Enforce exact floating-point mass conservation for column j
        off_diag_mass = float(np.dot(pivots[:j], mat[:j, j]) + np.dot(pivots[j + 1 :], mat[j + 1 :, j]))
        mat[j, j] = -off_diag_mass / xj
    return mat


def _validate_pivots(pivots: np.ndarray, q_min: int) -> np.ndarray:
    arr = np.asarray(pivots, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError("pivots must be a 1D array with at least 2 bins.")
    if not np.all(np.isfinite(arr)) or np.any(np.diff(arr) <= 0.0):
        raise ValueError("pivots must be finite and strictly increasing.")
    if abs(arr[0] - float(q_min)) > 1e-9:
        raise ValueError(f"pivots[0] ({arr[0]}) must equal spec.q_oligomer ({q_min}).")
    return arr


def solve_pbe(
    spec: MechanismSpec,
    times_min: np.ndarray,
    pivots: np.ndarray | None = None,
    m0_um: float | None = None,
    o0_um: float = 0.0,
    p0_um: float = 0.0,
    p0_length_ell: float | None = None,
    tht_conc_mm: float = 0.0,
    temperature_k: float | Callable[[float], float] = 298.15,
    method: str = "Radau",
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> PBETrajectory:
    """Integrate the stiff population balance equations without clipping using solve_ivp."""
    t_eval = np.asarray(times_min, dtype=float)
    if t_eval.ndim != 1 or len(t_eval) < 2 or np.any(np.diff(t_eval) < 0.0):
        raise ValueError("times_min must be a 1D non-decreasing array of length >= 2.")

    q = spec.q_oligomer
    pivots = build_log_pivot_grid(q_min=q) if pivots is None else _validate_pivots(pivots, q)
    n_bins = len(pivots)
    dx = np.diff(pivots)

    m_init = spec.c_total_um if m0_um is None else float(m0_um)
    y0 = np.zeros(3 + n_bins, dtype=float)
    y0[0] = m_init
    y0[1] = float(o0_um)
    y0[2] = 0.0
    if p0_um > 0.0:
        seed_ell = float(q) if p0_length_ell is None else float(p0_length_ell)
        if seed_ell < pivots[0] or seed_ell > pivots[-1]:
            raise ValueError(f"p0_length_ell ({seed_ell}) is outside pivot grid [{pivots[0]}, {pivots[-1]}].")
        idx = max(0, min(n_bins - 2, int(np.searchsorted(pivots, seed_ell, side="right")) - 1))
        alpha = (seed_ell - pivots[idx]) / (pivots[idx + 1] - pivots[idx])
        y0[3 + idx] += p0_um * (1.0 - alpha)
        y0[3 + idx + 1] += p0_um * alpha

    c_total = float(y0[0] + q * y0[1] + q * y0[2] + np.dot(pivots, y0[3:]))
    frag_mat = build_conservative_fragmentation_matrix(pivots, q) if spec.k_f > 0.0 else None

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        m = y[0]
        o_on = y[1]
        o_off = y[2]
        c = y[3:]

        mult = effective_rate_multiplier(spec, tht_conc_mm=tht_conc_mm, temperature_k=temperature_k, t_min=t)
        k_n = spec.k_n * mult
        k_plus = spec.k_plus * mult
        k_2 = spec.k_2 * mult
        k_f = spec.k_f * mult
        k_c = spec.k_c * mult
        k_d = spec.k_d * mult
        k_off = spec.k_off * mult
        k_d_off = spec.k_d_off * mult

        m_pos = m if m > 0.0 else 0.0
        mass_m = float(np.dot(pivots, c))
        mass_pos = mass_m if mass_m > 0.0 else 0.0

        j_n = k_n * (m_pos**spec.n_c)
        if k_2 > 0.0 and m_pos > 0.0:
            m_n2 = m_pos**spec.n_2
            sat = 1.0 + ((m_pos / spec.k_m_sat) ** spec.n_2 if np.isfinite(spec.k_m_sat) else 0.0)
            j_2 = k_2 * mass_pos * (m_n2 / sat)
        else:
            j_2 = 0.0

        flux_elong = (2.0 * k_plus * m_pos / dx) * c[:-1]
        p_grow = float(np.sum(c[:-1]))

        dm = -q * (j_n + j_2) + q * k_d * o_on + q * k_d_off * o_off - 2.0 * k_plus * m_pos * p_grow
        do_on = j_n + j_2 - (k_c + k_d + k_off) * o_on
        do_off = k_off * o_on - k_d_off * o_off

        dc = np.zeros(n_bins, dtype=float)
        dc[0] += k_c * o_on
        dc[:-1] -= flux_elong
        dc[1:] += flux_elong
        if frag_mat is not None:
            dc += k_f * (frag_mat @ c)

        dy = np.empty_like(y)
        dy[0] = dm
        dy[1] = do_on
        dy[2] = do_off
        dy[3:] = dc
        return dy

    def jac(t: float, y: np.ndarray) -> np.ndarray:
        m = y[0]
        c = y[3:]
        mult = effective_rate_multiplier(spec, tht_conc_mm=tht_conc_mm, temperature_k=temperature_k, t_min=t)
        k_n = spec.k_n * mult
        k_plus = spec.k_plus * mult
        k_2 = spec.k_2 * mult
        k_f = spec.k_f * mult
        k_c = spec.k_c * mult
        k_d = spec.k_d * mult
        k_off = spec.k_off * mult
        k_d_off = spec.k_d_off * mult

        m_pos = m if m > 0.0 else 0.0
        mass_m = float(np.dot(pivots, c))
        mass_pos = mass_m if mass_m > 0.0 else 0.0

        djn_dm = k_n * spec.n_c * (m_pos ** (spec.n_c - 1.0)) if (k_n > 0.0 and m_pos > 0.0) else 0.0
        if k_2 > 0.0 and m_pos > 0.0:
            m_n2 = m_pos**spec.n_2
            sat = 1.0 + ((m_pos / spec.k_m_sat) ** spec.n_2 if np.isfinite(spec.k_m_sat) else 0.0)
            dj2_dm = k_2 * mass_pos * spec.n_2 * (m_pos ** (spec.n_2 - 1.0)) / (sat**2)
            dj2_dmass = k_2 * (m_n2 / sat)
        else:
            dj2_dm = 0.0
            dj2_dmass = 0.0

        p_grow = float(np.sum(c[:-1]))
        jmat = np.zeros((3 + n_bins, 3 + n_bins), dtype=float)

        # Row 0: dm
        jmat[0, 0] = -q * (djn_dm + dj2_dm) - 2.0 * k_plus * p_grow
        jmat[0, 1] = q * k_d
        jmat[0, 2] = q * k_d_off
        jmat[0, 3:] = -q * dj2_dmass * pivots
        jmat[0, 3 : 3 + n_bins - 1] -= 2.0 * k_plus * m_pos

        # Row 1: do_on
        jmat[1, 0] = djn_dm + dj2_dm
        jmat[1, 1] = -(k_c + k_d + k_off)
        jmat[1, 3:] = dj2_dmass * pivots

        # Row 2: do_off
        jmat[2, 1] = k_off
        jmat[2, 2] = -k_d_off

        # Rows 3..: dc
        jmat[3, 1] = k_c
        elong_m = (2.0 * k_plus / dx) * c[:-1]
        jmat[3 : 3 + n_bins - 1, 0] -= elong_m
        jmat[4 : 3 + n_bins, 0] += elong_m

        rate_dx = 2.0 * k_plus * m_pos / dx
        idx_r = np.arange(n_bins - 1)
        jmat[3 + idx_r, 3 + idx_r] -= rate_dx
        jmat[4 + idx_r, 3 + idx_r] += rate_dx

        if frag_mat is not None:
            jmat[3:, 3:] += k_f * frag_mat
        return jmat

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method=method,
        t_eval=t_eval,
        jac=jac if method in ("Radau", "BDF") else None,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"PBE solve_ivp failed: {sol.message}")

    y_out = sol.y.T
    m_arr = y_out[:, 0]
    o_arr = y_out[:, 1]
    o_off_arr = y_out[:, 2]
    c_arr = y_out[:, 3:]

    p_arr = np.sum(c_arr, axis=1)
    mass_arr = c_arr @ pivots
    m2_arr = c_arr @ (pivots**2)
    mean_len = np.where(p_arr > 1e-15, mass_arr / np.maximum(p_arr, 1e-30), 0.0)

    total_mass_t = m_arr + q * o_arr + q * o_off_arr + mass_arr
    max_rel_err = float(np.max(np.abs(total_mass_t - c_total)) / max(c_total, 1e-12))
    max_upper_frac = float(np.max(c_arr[:, -1] * pivots[-1]) / max(c_total, 1e-12))

    return PBETrajectory(
        times_min=t_eval,
        pivots_ell=pivots,
        monomer_um=m_arr,
        oligomer_um=o_arr,
        off_pathway_um=o_off_arr,
        fibril_bins_um=c_arr,
        p_number_um=p_arr,
        m_mass_um=mass_arr,
        m2_second_moment_um=m2_arr,
        mean_length=mean_len,
        c_total_um=c_total,
        max_rel_mass_error=max_rel_err,
        q_oligomer=q,
        max_upper_boundary_mass_frac=max_upper_frac,
    )
