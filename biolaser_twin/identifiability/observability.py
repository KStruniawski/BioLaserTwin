"""Time-series Fisher Information Matrix (FIM) and windowed kinetic observability (never imports simulator/)."""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from ..constants import NominalOpticalConstants


@dataclass(frozen=True)
class KineticObservableConfig:
    """Configuration for realistic nominal time-series kinetic-optical forward map.

    Enforces physical initialization at t0_min (independent of truncated observation windows),
    nuisance uncertainty (s_sigma, a_pump, L_eff, rho_o), and non-zero lag-phase noise floors.
    """

    observable_set: str = "full_lasing"  # {"mass_only", "e_th", "full_lasing", "lasing_plus_mass"}
    s_sigma_known: bool = False
    tht_conc_mm: float = 3.0
    nominal_rho_o: float = 0.15
    crossover_ell: float = 120.0  # nominal RGD k*L ~ 1 crossover length (monomers)
    noise_std_rel: float = 0.03
    noise_std_abs: float = 0.25  # non-zero additive floor prevents noiseless lag-phase FIM inflation
    c_total_um: float = 50.0
    t0_min: float = 0.0
    nominal_cavity_um: float = 30.0
    pump_scale_a: float = 1.0
    fluo_scale_sf: float = 1.0
    q_nom: int = 4
    k_d_nom: float = 0.01
    grating_l_mm: int = 1200  # Kymera 328i: {300: 0.44 nm, 600: 0.21 nm, 1200: 0.10 nm}


def integrate_nominal_5moment_ode(
    times_min: np.ndarray,
    k_n: float,
    k_plus: float,
    k_2: float,
    k_c: float,
    k_d: float = 0.01,
    q_nom: int = 4,
    c_total_um: float = 50.0,
    t0_min: float = 0.0,
) -> dict[str, np.ndarray]:
    """Integrate nominal 5-moment ODE (m, O, P, M, M2) from t0_min and evaluate at times_min."""
    t_obs = np.asarray(times_min, dtype=float)
    if t_obs.ndim != 1 or len(t_obs) < 1 or np.any(t_obs < t0_min) or np.any(np.diff(t_obs) <= 0.0):
        raise ValueError("times_min must be strictly increasing and >= t0_min.")

    y0 = np.array([c_total_um, 0.0, 0.0, 0.0, 0.0], dtype=float)
    t_all = t_obs if abs(t_obs[0] - t0_min) < 1e-12 else np.concatenate([[t0_min], t_obs])

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        m, o, p_num, mass, _m2 = y
        m_pos = m if m > 0.0 else 0.0
        mass_pos = mass if mass > 0.0 else 0.0
        jn = k_n * (m_pos**2)
        j2 = k_2 * mass_pos * (m_pos**2)
        dm = -q_nom * (jn + j2) + q_nom * k_d * o - 2.0 * k_plus * m_pos * p_num
        do = jn + j2 - (k_c + k_d) * o
        dp = k_c * o
        dmass = 2.0 * k_plus * m_pos * p_num + q_nom * k_c * o
        dm2 = 2.0 * k_plus * m_pos * (2.0 * mass_pos + p_num) + (q_nom**2) * k_c * o
        return np.array([dm, do, dp, dmass, dm2], dtype=float)

    sol = solve_ivp(rhs, (float(t0_min), float(t_all[-1])), y0, method="Radau", t_eval=t_all, rtol=1e-8, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"Nominal 5-moment ODE failed: {sol.message}")

    start = 0 if len(t_all) == len(t_obs) else 1
    return {
        "times_min": t_obs,
        "m": sol.y[0, start:],
        "O": sol.y[1, start:],
        "P": sol.y[2, start:],
        "M": sol.y[3, start:],
        "M2": sol.y[4, start:],
    }


def _validate_theta_dim(theta_vec: np.ndarray, cfg: KineticObservableConfig) -> None:
    if cfg.s_sigma_known and len(theta_vec) != 4:
        raise ValueError(f"s_sigma_known=True requires 4 parameters, got {len(theta_vec)}.")
    if (not cfg.s_sigma_known) and len(theta_vec) < 5:
        raise ValueError(f"s_sigma_known=False requires >= 5 parameters (5th = log10 s_sigma), got {len(theta_vec)}.")


def nominal_kinetic_forward_observables(
    theta_vec: np.ndarray,
    times_min: np.ndarray,
    cfg: KineticObservableConfig | None = None,
    constants: NominalOpticalConstants | None = None,
) -> np.ndarray:
    """Map log10 kinetic rates (+ nuisance log10 s_sigma, optional a_pump, rho_o) to realistic observables.

    Uses a polydisperse-consistent RGD Padé closure with weight-average length ell_w = M2 / M:
      sigma_eff = s_sigma * (M2 / (1 + ell_w / ell_c) + q^2 * O)
    which reduces to M2 = (M^2/P)*(1 + CV_ell^2) for short rods (ell_w << ell_c) and collapses
    purely onto fibril mass ell_c * M for mature rods (ell_w >> ell_c).
    """
    c = cfg or KineticObservableConfig()
    _validate_theta_dim(theta_vec, c)
    consts = constants or NominalOpticalConstants()

    k_n = 10.0 ** float(theta_vec[0])
    k_plus = 10.0 ** float(theta_vec[1])
    k_2 = 10.0 ** float(theta_vec[2])
    k_c = 10.0 ** float(theta_vec[3])
    s_sigma = 1.0 if c.s_sigma_known else (10.0 ** float(theta_vec[4]))
    a_pump = c.pump_scale_a if len(theta_vec) < 6 else (10.0 ** float(theta_vec[5]))
    rho_o = c.nominal_rho_o if len(theta_vec) < 7 else float(np.clip(theta_vec[6], 0.0, 0.3))

    mom = integrate_nominal_5moment_ode(
        times_min,
        k_n=k_n,
        k_plus=k_plus,
        k_2=k_2,
        k_c=k_c,
        k_d=c.k_d_nom,
        q_nom=c.q_nom,
        c_total_um=c.c_total_um,
        t0_min=c.t0_min,
    )
    m_mass = np.maximum(0.0, mom["M"])
    m2_mom = np.maximum(0.0, mom["M2"])
    o_num = np.maximum(0.0, mom["O"])
    o_mass = float(c.q_nom) * o_num

    # Polydisperse-consistent RGD crossover using weight-average length ell_w = M2 / M:
    #   short rods (ell_w << ell_c) -> M2 (exact polydisperse (M^2/P)*(1 + CV_ell^2))
    #   long rods  (ell_w >> ell_c) -> ell_c * M (collapses purely onto fibril mass M)
    ell_w = np.where(m_mass > 1e-12, m2_mom / np.maximum(m_mass, 1e-12), 0.0)
    rgd_eff = s_sigma * (m2_mom / (1.0 + ell_w / max(c.crossover_ell, 1.0)) + (c.q_nom**2) * o_num)
    scat_loss = consts.scat_scale_nominal * rgd_eff

    # Bound ThT fluorescence / gain channel
    bound_eq = c.fluo_scale_sf * (m_mass + rho_o * o_mass)
    mirror_loss = -np.log(consts.mirror_reflectivity_nominal**2) / (2.0 * max(c.nominal_cavity_um * 1e-4, 1e-5))
    gain_drive = np.maximum(0.15, 0.25 * bound_eq + 0.05 * c.tht_conc_mm)

    # Observable 1: Pump-scaled lasing threshold E_th (uJ)
    e_th = a_pump * (mirror_loss + scat_loss) / gain_drive
    # Observable 2: Differential lasing slope dI/dE above threshold (gain-to-loss ratio)
    slope_eff = gain_drive / (1.0 + 0.2 * (mirror_loss + scat_loss))
    # Observable 3: Gain-peak wavelength shift (nm relative to 478 nm free ThT)
    lambda_shift = 8.0 * (bound_eq / max(c.c_total_um, 1.0)) - 0.15 * scat_loss

    if c.observable_set == "mass_only":
        return bound_eq[:, None]
    if c.observable_set == "e_th":
        return e_th[:, None]
    if c.observable_set == "full_lasing":
        return np.column_stack([e_th, slope_eff, lambda_shift])
    if c.observable_set == "lasing_plus_mass":
        return np.column_stack([e_th, slope_eff, lambda_shift, bound_eq])
    raise ValueError(f"Unknown observable_set: {c.observable_set}")


def observation_noise_std(y_model: np.ndarray, cfg: KineticObservableConfig) -> np.ndarray:
    """Compute grating-dependent, channel-calibrated noise std (wavelength noise scales with Kymera PSF)."""
    grating_psf = {300: 0.44, 600: 0.21, 1200: 0.10}.get(cfg.grating_l_mm, 0.10)
    wl_floor = 0.25 * grating_psf
    if cfg.observable_set == "mass_only":
        abs_floor = np.array([0.5])
    elif cfg.observable_set == "e_th":
        abs_floor = np.array([0.3])
    elif cfg.observable_set == "full_lasing":
        abs_floor = np.array([0.3, 0.02, wl_floor])
    elif cfg.observable_set == "lasing_plus_mass":
        abs_floor = np.array([0.3, 0.02, wl_floor, 0.5])
    else:
        abs_floor = np.array([cfg.noise_std_abs])
    return np.sqrt((abs_floor**2) + (cfg.noise_std_rel * np.abs(y_model)) ** 2)


def compute_kinetic_timeseries_fim(
    theta_vec: np.ndarray,
    times_min: np.ndarray,
    cfg: KineticObservableConfig | None = None,
    eps: float = 1e-4,
    noise_std: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Fisher Information Matrix F = J^T Sigma^-1 J over a time-series observable trajectory."""
    c = cfg or KineticObservableConfig()
    _validate_theta_dim(theta_vec, c)
    y0 = nominal_kinetic_forward_observables(theta_vec, times_min, c)
    scales = (observation_noise_std(y0, c) if noise_std is None else np.broadcast_to(np.asarray(noise_std, dtype=float), y0.shape)).reshape(-1)
    cov_inv_diag = 1.0 / (np.maximum(scales, 1e-12) ** 2)
    if valid_mask is not None:
        cov_inv_diag = np.where(np.asarray(valid_mask, dtype=bool).reshape(-1), cov_inv_diag, 0.0)

    d = len(theta_vec)
    jac = np.zeros((y0.size, d), dtype=float)
    for j in range(d):
        step = np.zeros(d, dtype=float)
        step[j] = eps
        yp = nominal_kinetic_forward_observables(theta_vec + step, times_min, c).reshape(-1)
        ym = nominal_kinetic_forward_observables(theta_vec - step, times_min, c).reshape(-1)
        jac[:, j] = (yp - ym) / (2.0 * eps)

    return (jac.T * cov_inv_diag) @ jac


def kinetic_crb_widths(fim: np.ndarray, z_crit: float = 1.96, rel_tol: float = 1e-8) -> np.ndarray:
    """Return 95% CRB CI widths; parameters projecting onto the FIM null space get np.inf."""
    d = fim.shape[0]
    eigvals, eigvecs = np.linalg.eigh(0.5 * (fim + fim.T))
    max_eig = float(np.max(eigvals)) if len(eigvals) > 0 else 0.0
    if max_eig <= 1e-18:
        return np.full(d, np.inf, dtype=float)

    null_mask = eigvals <= (rel_tol * max_eig)
    pos_mask = ~null_mask
    # Reconstruct Moore-Penrose inverse from positive eigenvalues only
    v_pos = eigvecs[:, pos_mask]
    cov_pos = (v_pos / eigvals[pos_mask]) @ v_pos.T

    widths = np.full(d, np.inf, dtype=float)
    v_null = eigvecs[:, null_mask]
    for i in range(d):
        if np.any(null_mask) and float(np.linalg.norm(v_null[i, :])) > 1e-6:
            widths[i] = np.inf
        else:
            widths[i] = 2.0 * z_crit * float(np.sqrt(max(0.0, cov_pos[i, i])))
    return widths
