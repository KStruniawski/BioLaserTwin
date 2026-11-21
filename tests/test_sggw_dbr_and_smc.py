"""Unit tests for SGGW DBR mirror integration, dispersion inflation, and MH-SMC smoother."""

import numpy as np

from biolaser_twin.identifiability import (
    KineticObservableConfig,
    nominal_kinetic_forward_observables,
    profile_likelihood_log10_kp,
)
from biolaser_twin.simulator import TruthSampleState
from biolaser_twin.simulator.fp_cavity import load_sggw_dbr_mirror_spectra, sggw_dbr_pump_transmittance
from biolaser_twin.simulator.photophysics import effective_gain_and_emission
from biolaser_twin.solvers import SequentialImportanceSmoother


def test_sggw_dbr_spectra_and_dispersive_reflectivity_preserved() -> None:
    """Verify sggw_dbr_mirror_spectra.json loader, pump transmittance, and zero-pump R(lambda)."""
    dbr = load_sggw_dbr_mirror_spectra()
    assert len(dbr["wavelength_nm"]) == 346
    t_433 = sggw_dbr_pump_transmittance(433.3, dbr_shift_nm=0.0, use_exp=True)
    assert 0.02 <= t_433 <= 0.05

    state = TruthSampleState(
        refractive_index=1.333,
        cavity_length_um=33.22,
        microviscosity_cp=1.0,
        bound_fraction=0.0,
        scattering_loss_cm1=0.0,
        tht_conc_mm=3.0,
    )
    wl = np.array([480.0, 505.0, 530.0])
    r_lam = np.interp(wl, dbr["wavelength_nm"], dbr["experiment_R_0deg_inferred"])
    _, net_gain, _ = effective_gain_and_emission(state, 433.3, 0.0, wl, mirror_reflectivity=r_lam)
    l_cm = state.cavity_length_um * 1e-4
    mirror_loss_lam = -np.log(r_lam**2) / (2.0 * l_cm)
    g_rt = r_lam * np.exp((net_gain + mirror_loss_lam) * l_cm)
    assert np.allclose(g_rt / r_lam, 1.0, atol=0.02)


def test_profile_likelihood_inflate_dispersion() -> None:
    """Verify inflate_dispersion=True widens CI when chi2_red > 1."""
    times = np.array([0.0, 120.0, 240.0, 360.0, 480.0, 600.0])
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, grating_l_mm=1200)
    th = np.array([-7.5, -1.3, -5.5, -1.0])
    y_obs = nominal_kinetic_forward_observables(th, times, cfg)
    y_obs[2, 0] += 2.5  # inject misfit so chi2_red > 1
    grid = np.linspace(-1.8, -0.8, 21)
    p_std = profile_likelihood_log10_kp(y_obs, times, th, cfg, kp_grid=grid, inflate_dispersion=False)
    p_inf = profile_likelihood_log10_kp(y_obs, times, th, cfg, kp_grid=grid, inflate_dispersion=True)
    assert np.isclose(p_inf.chi2_min, p_std.chi2_min, rtol=1e-5)
    assert p_inf.ci_width >= p_std.ci_width


def test_mh_smc_sequential_importance_smoother_ess() -> None:
    """Verify posterior-invariant MH-SMC smoother maintains non-collapsed ESS."""
    times = np.array([0.0, 150.0, 300.0, 450.0, 600.0])
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, grating_l_mm=1200)
    th = np.array([-7.5, -1.3, -5.5, -1.0])
    y_obs = nominal_kinetic_forward_observables(th, times, cfg)
    fit = SequentialImportanceSmoother(cfg, n_particles=24, seed=42).filter_predict(
        times, y_obs, theta_prior_mean=th + 0.1, theta_prior_std=np.full(4, 0.5)
    )
    assert fit.ess is not None and fit.ess >= 2.0
    assert fit.q1_std > 0.0

    # Smoke check on matched nominal data (k_d_eff=0.01 matching cfg.k_d_nom; does not establish full calibration)
    from biolaser_twin.kinetics_nominal import NominalKineticParams, integrate_nominal_moment_ode
    mom = integrate_nominal_moment_ode(
        times, NominalKineticParams(k_n_eff=10**th[0], k_plus_eff=10**th[1], k_2_eff=10**th[2], k_c_eff=10**th[3], k_d_eff=cfg.k_d_nom)
    )
    q1_true = float(np.trapz(4.0 * mom["O"], times) / (times[-1] * 50.0))
    fit_tight = SequentialImportanceSmoother(cfg, n_particles=64, seed=7).filter_predict(
        times, y_obs, theta_prior_mean=th, theta_prior_std=np.full(4, 0.25)
    )
    assert fit_tight.ess is not None and fit_tight.ess >= 10.0
    assert np.isfinite(fit_tight.q1_mean) and q1_true > 0.0


def test_observations_adapter_sign_preserving_lambda_scale() -> None:
    """Verify calibrate_kinetic_adapter_from_dev preserves negative lam_scale and |scale| < 0.1 fallback."""
    from types import SimpleNamespace
    from biolaser_twin.observations import calibrate_kinetic_adapter_from_dev

    # Negative slope: com rises while y_nom[:, 2] falls
    feats_neg = [SimpleNamespace(threshold_energy_uj=20.0 - 2.0 * i, slope_efficiency=0.1 * i, center_of_mass_nm=480.0 + float(i)) for i in range(5)]
    y_nom = np.column_stack([20.0 - np.arange(5), 0.1 * np.arange(5), -2.0 * np.arange(5)])
    cal_neg = calibrate_kinetic_adapter_from_dev(feats_neg, y_nom)
    assert cal_neg.lambda_scale < -0.5 and np.isclose(cal_neg.lambda_ref_nm, 480.0)

    # Near-zero slope fallback (|raw_scale| < 0.1)
    feats_flat = [SimpleNamespace(threshold_energy_uj=20.0, slope_efficiency=0.1, center_of_mass_nm=480.0 + 2.0 * i) for i in range(5)]
    y_nom_tiny = np.column_stack([np.full(5, 10.0), np.full(5, 0.2), -0.01 * np.arange(5)])
    cal_fb = calibrate_kinetic_adapter_from_dev(feats_flat, y_nom_tiny)
    assert cal_fb.lambda_scale == -0.5
