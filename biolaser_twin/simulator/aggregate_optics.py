"""Aggregate RGD scattering optics, millimolar ThT binding saturation, and crowding microviscosity."""

from dataclasses import dataclass
import numpy as np
from scipy.special import sici

from .kinetics.population_balance import PBETrajectory
from .photophysics import TruthSampleState


# Precomputed 64-node Gauss-Legendre quadrature over mu = cos(theta) in [-1, 1]
_GL_MU, _GL_W = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True)
class AggregateOpticsParams:
    """Physical parameters coupling PBE state to RGD scattering, ThT binding, and microviscosity."""

    refractive_index: float = 1.345
    reference_wavelength_nm: float = 485.0
    monomer_step_nm: float = 0.48  # cross-beta axial rise per monomer (nm)
    monomer_radius_nm: float = 1.2  # effective monomer hydrodynamic radius (nm)
    s_sigma_scale: float = 1.0  # dimensionless specific scattering cross-section multiplier
    base_rayleigh_cm2: float = 1.5e-20  # effective monomer Rayleigh cross-section at ref wavelength (cm^2)
    binding_stoich_s: float = 0.25  # ThT binding sites per fibril monomer
    oligomer_tht_contrast: float = 0.15  # rho_o in [0, 0.3]
    kd_tht_um: float = 5.0  # ThT dissociation constant (uM)
    eta_solvent_cp: float = 1.0  # baseline buffer viscosity (cP)
    crowding_beta: float = 1.8  # crowding exponent
    crowding_ref_um: float = 50.0  # reference concentration for crowding (uM)
    crowding_sat_coeff: float = 0.25  # saturation prevents unphysical blowup at 500 uM
    oligomer_visc_weight: float = 0.5  # relative crowding weight of oligomers vs fibrils
    forster_hoffmann_x: float = 0.55  # Phi_F ~ eta_mu^x, x in [0.4, 0.7]


def _rgd_rod_form_factor(q_len: np.ndarray) -> np.ndarray:
    """Orientation-averaged RGD form factor P_rod(qL) = (2/x)*Si(x) - (sin(x/2)/(x/2))^2."""
    x = np.asarray(q_len, dtype=float)
    out = np.ones_like(x)
    mask = x > 1e-5
    xm = x[mask]
    si_val, _ = sici(xm)
    sinc_half = np.sin(0.5 * xm) / (0.5 * xm)
    out[mask] = (2.0 / xm) * si_val - (sinc_half**2)
    return np.maximum(out, 0.0)


def rgd_rod_cross_section(
    lengths_ell: np.ndarray,
    wavelength_nm: float,
    params: AggregateOpticsParams | None = None,
) -> np.ndarray:
    """Compute orientation-averaged RGD total scattering cross-section (cm^2) for thin rods.

    Includes the Rayleigh (lambda_ref / lambda)^4 prefactor and naturally crosses over from
    sigma ~ ell^2 (k*L << 1) to sigma ~ ell (k*L >> 1).
    """
    p = params or AggregateOpticsParams()
    ell = np.asarray(lengths_ell, dtype=float)
    l_nm = ell * p.monomer_step_nm
    k_med = 2.0 * np.pi * p.refractive_index / wavelength_nm
    lam_prefactor = (p.reference_wavelength_nm / wavelength_nm) ** 4

    sin_half = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - _GL_MU)))
    ang_weight = 0.5 * (1.0 + _GL_MU**2) * _GL_W

    q_len = (2.0 * k_med * l_nm[:, None]) * sin_half[None, :]
    p_factor = _rgd_rod_form_factor(q_len)
    integral = p_factor @ ang_weight
    return p.s_sigma_scale * p.base_rayleigh_cm2 * lam_prefactor * (ell**2) * (0.75 * integral)


def rgd_sphere_cross_section(
    q_oligomer: int,
    wavelength_nm: float,
    params: AggregateOpticsParams | None = None,
) -> float:
    """Compute RGD total scattering cross-section (cm^2) for a spherical oligomer (sigma ~ V^2)."""
    p = params or AggregateOpticsParams()
    r_nm = p.monomer_radius_nm * (float(q_oligomer) ** (1.0 / 3.0))
    k_med = 2.0 * np.pi * p.refractive_index / wavelength_nm
    lam_prefactor = (p.reference_wavelength_nm / wavelength_nm) ** 4

    sin_half = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - _GL_MU)))
    ang_weight = 0.5 * (1.0 + _GL_MU**2) * _GL_W

    qr = 2.0 * k_med * r_nm * sin_half
    p_sph = np.ones_like(qr)
    mask = qr > 1e-4
    qrm = qr[mask]
    p_sph[mask] = (3.0 * (np.sin(qrm) - qrm * np.cos(qrm)) / (qrm**3)) ** 2
    integral = float(np.dot(p_sph, ang_weight))
    return float(p.s_sigma_scale * p.base_rayleigh_cm2 * lam_prefactor * (float(q_oligomer) ** 2) * (0.75 * integral))


def compute_aggregate_scattering_cm1(
    traj: PBETrajectory,
    wavelength_nm: float = 485.0,
    params: AggregateOpticsParams | None = None,
    include_off_pathway: bool = True,
) -> np.ndarray:
    """Compute total RGD turbidity alpha_scat(t) in cm^-1 from oligomer + fibril distributions."""
    p = params or AggregateOpticsParams()
    avogadro_per_um = 6.02214076e14
    sigma_rod = rgd_rod_cross_section(traj.pivots_ell, wavelength_nm, p)
    sigma_sph = rgd_sphere_cross_section(traj.q_oligomer, wavelength_nm, p)

    rod_part = traj.fibril_bins_um @ sigma_rod
    o_total = traj.oligomer_um + (traj.off_pathway_um if include_off_pathway else 0.0)
    oligo_part = o_total * sigma_sph
    return avogadro_per_um * (rod_part + oligo_part)


def compute_tht_binding(
    m_mass_um: np.ndarray,
    oligomer_um: np.ndarray,
    q_oligomer: int,
    tht_conc_mm: float,
    params: AggregateOpticsParams | None = None,
    off_pathway_um: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bound ThT concentration c_b (uM) and bound fraction f_b = c_b / [ThT] under mM saturation."""
    p = params or AggregateOpticsParams()
    if not (0.0 <= p.oligomer_tht_contrast <= 0.3):
        raise ValueError(f"oligomer_tht_contrast ({p.oligomer_tht_contrast}) must be in [0.0, 0.3].")

    tht_um = max(0.0, float(tht_conc_mm) * 1000.0)
    if tht_um <= 0.0:
        zeros = np.zeros_like(m_mass_um, dtype=float)
        return zeros, zeros

    o_total = oligomer_um + (off_pathway_um if off_pathway_um is not None else 0.0)
    sites_um = p.binding_stoich_s * np.maximum(0.0, m_mass_um + p.oligomer_tht_contrast * q_oligomer * o_total)
    b_term = tht_um + sites_um + p.kd_tht_um
    disc = np.sqrt(np.maximum(0.0, b_term**2 - 4.0 * tht_um * sites_um))
    # Cancellation-resistant quadratic root: 2 * a * c / (b + sqrt(b^2 - 4ac))
    c_bound_um = (2.0 * tht_um * sites_um) / np.maximum(b_term + disc, 1e-30)
    f_bound = c_bound_um / tht_um
    return c_bound_um, f_bound


def compute_crowding_microviscosity_cp(
    m_mass_um: np.ndarray,
    oligomer_um: np.ndarray,
    q_oligomer: int,
    params: AggregateOpticsParams | None = None,
    off_pathway_um: np.ndarray | None = None,
) -> np.ndarray:
    """Compute saturating crowding-induced microviscosity eta_mu(t) in cP (finite up to 500+ uM)."""
    p = params or AggregateOpticsParams()
    o_total = oligomer_um + (off_pathway_um if off_pathway_um is not None else 0.0)
    eff_conc = np.maximum(0.0, m_mass_um + p.oligomer_visc_weight * q_oligomer * o_total)
    denom = max(p.crowding_ref_um, 1e-6) + p.crowding_sat_coeff * eff_conc
    return p.eta_solvent_cp * np.exp(p.crowding_beta * eff_conc / denom)


def trajectory_to_truth_states(
    traj: PBETrajectory,
    tht_conc_mm: float = 3.0,
    cavity_length_um: float = 30.0,
    wavelength_nm: float = 485.0,
    temperature_k: float = 298.15,
    params: AggregateOpticsParams | None = None,
    spectral_grid_nm: tuple[float, ...] = (450.0, 465.0, 480.0, 495.0, 510.0, 530.0, 560.0, 600.0),
) -> list[TruthSampleState]:
    """Map a PBE trajectory into a time series of TruthSampleState frames with spectral RGD scattering."""
    p = params or AggregateOpticsParams()
    alpha_scat = compute_aggregate_scattering_cm1(traj, wavelength_nm=wavelength_nm, params=p)
    alpha_spec = np.stack(
        [compute_aggregate_scattering_cm1(traj, wavelength_nm=wl, params=p) for wl in spectral_grid_nm],
        axis=1,
    )
    c_bound, f_bound = compute_tht_binding(
        traj.m_mass_um,
        traj.oligomer_um,
        traj.q_oligomer,
        tht_conc_mm=tht_conc_mm,
        params=p,
        off_pathway_um=traj.off_pathway_um,
    )
    eta_cp = compute_crowding_microviscosity_cp(
        traj.m_mass_um,
        traj.oligomer_um,
        traj.q_oligomer,
        params=p,
        off_pathway_um=traj.off_pathway_um,
    )

    return [
        TruthSampleState(
            refractive_index=float(p.refractive_index),
            cavity_length_um=float(cavity_length_um),
            microviscosity_cp=float(eta_cp[i]),
            bound_fraction=float(np.clip(f_bound[i], 0.0, 1.0)),
            scattering_loss_cm1=float(max(1e-4, alpha_scat[i])),
            tht_conc_mm=float(tht_conc_mm),
            temperature_k=float(temperature_k),
            bound_tht_um=float(c_bound[i]),
            forster_hoffmann_x=float(p.forster_hoffmann_x),
            scattering_wavelengths_nm=spectral_grid_nm,
            scattering_loss_spectrum_cm1=tuple(float(max(1e-4, v)) for v in alpha_spec[i]),
        )
        for i in range(len(traj.times_min))
    ]
