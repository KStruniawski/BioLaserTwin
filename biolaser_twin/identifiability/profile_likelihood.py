"""Profile-likelihood identifiability for log10(k_+) and CMA-ES indistinguishable-pair search (S2/S3)."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, least_squares

from .observability import (
    KineticObservableConfig,
    integrate_nominal_5moment_ode,
    nominal_kinetic_forward_observables,
    observation_noise_std,
)


@dataclass(frozen=True)
class ProfileLikelihoodResult:
    """Nuisance-profiled likelihood curve and 95% CI width for log10(k_+)."""

    grid_values: np.ndarray
    chi2_profile: np.ndarray
    mle_value: float
    ci_low: float
    ci_high: float
    ci_width: float
    status: str  # {"resolved", "boundary_censored", "unbounded", "failed"}
    bounded_within_1_decade: bool
    chi2_min: float = 0.0


@dataclass(frozen=True)
class IndistinguishablePair:
    """Two kinetic parameterizations with divergent oligomer burdens q1 yet matched optical curves."""

    theta_a: np.ndarray
    theta_b: np.ndarray
    q1_a: float
    q1_b: float
    abs_delta_q1: float
    rel_optical_distance: float
    feasible: bool


def profile_likelihood_log10_kp(
    y_obs: np.ndarray,
    times_min: np.ndarray,
    theta_init: np.ndarray,
    cfg: KineticObservableConfig | None = None,
    kp_grid: np.ndarray | None = None,
    bounds_low: np.ndarray | None = None,
    bounds_high: np.ndarray | None = None,
    noise_std: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    inflate_dispersion: bool = False,
) -> ProfileLikelihoodResult:
    """Compute nuisance-profiled chi^2(log10 k_+) with <=0.05-decade grid, brentq, and frozen noise weights."""
    c = cfg or KineticObservableConfig()
    d = len(theta_init)
    if bounds_low is None:
        base_lo = [-11.0, -3.5, -9.0, -4.0]
        if d >= 5:
            base_lo.append(-2.0)
        if d >= 6:
            base_lo.append(-1.0)
        if d >= 7:
            base_lo.append(0.0)
        bounds_low = np.array(base_lo[:d], dtype=float)
    if bounds_high is None:
        base_hi = [-4.5, 0.5, -4.0, 1.0]
        if d >= 5:
            base_hi.append(2.0)
        if d >= 6:
            base_hi.append(1.0)
        if d >= 7:
            base_hi.append(0.3)
        bounds_high = np.array(base_hi[:d], dtype=float)
    if kp_grid is None:
        kp_grid = np.linspace(float(bounds_low[1]), float(bounds_high[1]), 81)
    else:
        kp_grid = np.asarray(kp_grid, dtype=float)
    # Unconditionally enforce max step <= 0.05 decades across ALL grid paths (default or custom bounds)
    if len(kp_grid) >= 2 and np.max(np.diff(kp_grid)) > 0.05:
        n_sub = max(21, int(np.ceil((kp_grid[-1] - kp_grid[0]) / 0.05)) + 1)
        kp_grid = np.unique(np.sort(np.concatenate([kp_grid, np.linspace(float(kp_grid[0]), float(kp_grid[-1]), n_sub)])))

    # Freeze noise weights at observed-data sigma_obs (or externally estimated simulator sigma)
    scales_obs = (
        observation_noise_std(y_obs, c)
        if noise_std is None
        else np.broadcast_to(np.maximum(np.asarray(noise_std, dtype=float), 1e-12), y_obs.shape)
    )
    mask_flat = np.ones(y_obs.size, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)

    def _res_full(th: np.ndarray) -> np.ndarray:
        pred = nominal_kinetic_forward_observables(th, times_min, c)
        res = ((pred - y_obs) / scales_obs).reshape(-1)
        return res[mask_flat]

    mle_fit = least_squares(_res_full, np.clip(theta_init, bounds_low, bounds_high), bounds=(bounds_low, bounds_high), max_nfev=100)
    mle_kp = float(mle_fit.x[1])
    if kp_grid[0] <= mle_kp <= kp_grid[-1]:
        kp_grid = np.unique(np.sort(np.append(kp_grid, mle_kp)))
    chi2_raw = np.empty(len(kp_grid), dtype=float)
    nuisance_idx = [i for i in range(d) if i != 1]
    b_lo_n = bounds_low[nuisance_idx]
    b_hi_n = bounds_high[nuisance_idx]
    prev_n = mle_fit.x[nuisance_idx].copy()
    init_n = np.clip(theta_init[nuisance_idx], b_lo_n, b_hi_n)

    # Sweep across grid with continuation + multi-start
    def _eval_profile_at(kp_val: float, warm_start: np.ndarray) -> tuple[float, np.ndarray]:
        def _res_nuis(th_n: np.ndarray) -> np.ndarray:
            th_full = np.empty(d, dtype=float)
            th_full[1] = float(kp_val)
            for pos, orig_i in enumerate(nuisance_idx):
                th_full[orig_i] = th_n[pos]
            return _res_full(th_full)

        best_chi2 = float("inf")
        best_x = warm_start
        for start in (warm_start, init_n):
            try:
                fit_n = least_squares(_res_nuis, start, bounds=(b_lo_n, b_hi_n), max_nfev=45)
                val = float(np.sum(fit_n.fun**2))
                if val < best_chi2:
                    best_chi2 = val
                    best_x = fit_n.x
            except RuntimeError:
                continue
        return best_chi2, best_x

    nuisance_solutions: list[np.ndarray] = []
    for idx, kp_val in enumerate(kp_grid):
        val, prev_n = _eval_profile_at(float(kp_val), prev_n)
        chi2_raw[idx] = val
        nuisance_solutions.append(prev_n.copy())

    chi2_min = min(float(np.sum(mle_fit.fun**2)), float(np.min(chi2_raw)))
    chi2_prof = np.maximum(0.0, chi2_raw - chi2_min)
    mle_kp = float(kp_grid[int(np.argmin(chi2_prof))]) if float(np.min(chi2_raw)) < float(np.sum(mle_fit.fun**2)) else float(mle_fit.x[1])

    dof = max(1, int(np.sum(mask_flat)) - d)
    disp_scale = max(1.0, chi2_min / dof) if inflate_dispersion else 1.0
    threshold = 3.841 * disp_scale  # 95% chi^2(df=1) with optional quasi-likelihood dispersion inflation
    in_mask = chi2_prof <= threshold
    if not np.any(in_mask):
        return ProfileLikelihoodResult(
            grid_values=kp_grid,
            chi2_profile=chi2_prof,
            mle_value=mle_kp,
            ci_low=float("nan"),
            ci_high=float("nan"),
            ci_width=float("inf"),
            status="failed",
            bounded_within_1_decade=False,
            chi2_min=chi2_min,
        )

    idx_in = np.where(in_mask)[0]
    i_lo, i_hi = int(idx_in[0]), int(idx_in[-1])
    left_censored = i_lo == 0
    right_censored = i_hi == (len(kp_grid) - 1)

    # Exact brentq root-finding on crossing intervals (warm-started from nearest grid solution)
    if not left_censored:
        def _f_lo(kv: float) -> float:
            v, _ = _eval_profile_at(kv, nuisance_solutions[i_lo])
            return (v - chi2_min) - threshold

        try:
            ci_lo = float(brentq(_f_lo, float(kp_grid[i_lo - 1]), float(kp_grid[i_lo]), xtol=1e-3, maxiter=20))
        except (ValueError, RuntimeError):
            ci_lo = float(kp_grid[i_lo - 1])
    else:
        ci_lo = float(kp_grid[0])

    if not right_censored:
        def _f_hi(kv: float) -> float:
            v, _ = _eval_profile_at(kv, nuisance_solutions[i_hi])
            return (v - chi2_min) - threshold

        try:
            ci_hi = float(brentq(_f_hi, float(kp_grid[i_hi]), float(kp_grid[i_hi + 1]), xtol=1e-3, maxiter=20))
        except (ValueError, RuntimeError):
            ci_hi = float(kp_grid[i_hi + 1])
    else:
        ci_hi = float(kp_grid[-1])

    width = float(ci_hi - ci_lo)
    if left_censored and right_censored:
        status = "unbounded"
    elif left_censored or right_censored:
        status = "boundary_censored"
    elif np.any(~in_mask[i_lo : i_hi + 1]):
        status = "disconnected"
    else:
        status = "resolved"

    return ProfileLikelihoodResult(
        grid_values=kp_grid,
        chi2_profile=chi2_prof,
        mle_value=mle_kp,
        ci_low=ci_lo,
        ci_high=ci_hi,
        ci_width=width,
        status=status,
        bounded_within_1_decade=bool(status == "resolved" and width < 1.0),
        chi2_min=chi2_min,
    )


def _dense_nominal_q1(theta_vec: np.ndarray, t_end_min: float, cfg: KineticObservableConfig, n_dense: int = 151) -> float:
    t_dense = np.linspace(cfg.t0_min, float(t_end_min), n_dense)
    mom = integrate_nominal_5moment_ode(
        t_dense,
        k_n=10.0 ** float(theta_vec[0]),
        k_plus=10.0 ** float(theta_vec[1]),
        k_2=10.0 ** float(theta_vec[2]),
        k_c=10.0 ** float(theta_vec[3]),
        k_d=cfg.k_d_nom,
        q_nom=cfg.q_nom,
        c_total_um=cfg.c_total_um,
        t0_min=cfg.t0_min,
    )
    o_mass = float(cfg.q_nom) * np.maximum(0.0, mom["O"])
    t_span = max(float(t_end_min - cfg.t0_min), 1e-9)
    area = float(np.trapezoid(o_mass, t_dense) if hasattr(np, "trapezoid") else np.trapz(o_mass, t_dense))
    return area / (t_span * max(cfg.c_total_um, 1e-9))


def find_indistinguishable_pair(
    theta_ref: np.ndarray,
    times_min: np.ndarray,
    cfg: KineticObservableConfig | None = None,
    optical_rel_tol: float = 1.0,  # <= 1.0 * noise_std per protocol ("<= noise level")
    min_delta_q1: float = 0.15,
    max_iters: int = 35,
    pop_size: int = 16,
    seed: int = 42,
) -> IndistinguishablePair:
    """CMA-ES search maximizing |q1(theta_ref) - q1(theta_b)| subject to noise-weighted optical distance <= tol."""
    c = cfg or KineticObservableConfig()
    rng = np.random.default_rng(seed)
    y_ref = nominal_kinetic_forward_observables(theta_ref, times_min, c)
    sig_ref = observation_noise_std(y_ref, c)
    t_end = float(times_min[-1])
    q1_ref = _dense_nominal_q1(theta_ref, t_end, c)

    d = len(theta_ref)
    b_lo = np.array([-10.5, -3.5, -9.0, -4.2] + ([-2.0] if d >= 5 else []) + ([-1.0] if d >= 6 else []) + ([0.0] if d >= 7 else []), dtype=float)[:d]
    b_hi = np.array([-4.5, 0.5, -4.0, 0.5] + ([2.0] if d >= 5 else []) + ([1.0] if d >= 6 else []) + ([0.3] if d >= 7 else []), dtype=float)[:d]

    mean = np.clip(theta_ref.copy(), b_lo, b_hi)
    sigma = 0.25 * (b_hi - b_lo)
    cov = np.eye(d, dtype=float)
    mu = max(2, pop_size // 3)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)

    best_feasible: tuple[float, np.ndarray, float, float] | None = None
    best_any = (-float("inf"), mean.copy(), q1_ref, float("inf"))

    for _ in range(max_iters):
        candidates = []
        cov_step = np.outer(sigma, sigma) * cov
        cov_step = 0.5 * (cov_step + cov_step.T) + 1e-8 * np.eye(d)
        for _ in range(pop_size):
            cand = np.clip(rng.multivariate_normal(mean, cov_step), b_lo, b_hi)
            y_b = nominal_kinetic_forward_observables(cand, times_min, c)
            # Noise-normalized RMS distance (dimensionless)
            rel_dist = float(np.sqrt(np.mean(((y_b - y_ref) / sig_ref) ** 2)))
            q1_b = _dense_nominal_q1(cand, t_end, c)
            dq1 = abs(q1_b - q1_ref)
            if rel_dist <= optical_rel_tol:
                score = -dq1
                if best_feasible is None or dq1 > best_feasible[0]:
                    best_feasible = (dq1, cand.copy(), q1_b, rel_dist)
            else:
                score = 10.0 * (rel_dist - optical_rel_tol) - 0.1 * dq1
            if (dq1 - 10.0 * max(0.0, rel_dist - optical_rel_tol)) > best_any[0]:
                best_any = (dq1, cand.copy(), q1_b, rel_dist)
            candidates.append((score, cand))

        candidates.sort(key=lambda item: item[0])
        elites = np.stack([item[1] for item in candidates[:mu]], axis=0)
        new_mean = np.sum(weights[:, None] * elites, axis=0)
        diff = elites - mean[None, :]
        cov_emp = (weights[:, None] * diff).T @ diff
        cov = 0.7 * cov + 0.3 * (cov_emp / np.maximum(sigma[:, None] * sigma[None, :], 1e-8) + 1e-4 * np.eye(d))
        mean = new_mean

    if best_feasible is not None:
        dq1, th_b, q1_b, rel_d = best_feasible
        return IndistinguishablePair(
            theta_a=np.asarray(theta_ref, dtype=float),
            theta_b=th_b,
            q1_a=q1_ref,
            q1_b=q1_b,
            abs_delta_q1=dq1,
            rel_optical_distance=rel_d,
            feasible=bool(rel_d <= optical_rel_tol and dq1 >= min_delta_q1),
        )
    _, th_b, q1_b, rel_d = best_any
    return IndistinguishablePair(
        theta_a=np.asarray(theta_ref, dtype=float),
        theta_b=th_b,
        q1_a=q1_ref,
        q1_b=q1_b,
        abs_delta_q1=abs(q1_b - q1_ref),
        rel_optical_distance=rel_d,
        feasible=False,
    )
