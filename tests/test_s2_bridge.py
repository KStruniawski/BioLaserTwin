"""End-to-end S2 anti-inverse-crime bridge test: simulator -> disjoint dev adapter -> profile likelihood."""

import numpy as np

from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import (
    KineticObservableConfig,
    nominal_kinetic_forward_observables,
    profile_likelihood_log10_kp,
)
from biolaser_twin.observations import (
    calibrate_kinetic_adapter_from_dev,
    estimate_kinetic_noise_std,
    extract_kinetic_observable_series,
)
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, trajectory_to_truth_states
from biolaser_twin.simulator.kinetics import MECHANISM_ZOO, solve_pbe


def test_s2_end_to_end_simulator_to_profile_likelihood_bridge() -> None:
    """Verify S2 data-generation contract: disjoint dev calibration, >=5 noise replicates, and chi2/dof <= 3."""
    times = np.array([0.0, 90.0, 150.0, 210.0, 300.0, 450.0, 600.0])
    energies = np.linspace(2.0, 45.0, 12)
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, grating_l_mm=1200)

    # Disjoint dev trajectory (M3, not M2) calibrates adapter nuisance parameters
    spec_dev = MECHANISM_ZOO["M3"]
    states_dev = trajectory_to_truth_states(solve_pbe(spec_dev, times), tht_conc_mm=3.0, cavity_length_um=30.0)
    theta_dev = np.array([np.log10(spec_dev.k_n), np.log10(spec_dev.k_plus), np.log10(spec_dev.k_2), np.log10(spec_dev.k_c)])
    y_nom_dev = nominal_kinetic_forward_observables(theta_dev, times, cfg)
    sim_dev = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=8, seed=10))
    feats_dev = [
        extract_sweep_features(
            b.frames, b.wavelengths_nm, b.pump_energies_uj, spec_psf_fwhm_nm=b.spec_psf_fwhm_nm, dark_frames=b.dark_frames
        )
        for b in (sim_dev.render_pump_sweep(s, 445.0, energies) for s in states_dev)
    ]
    adapter = calibrate_kinetic_adapter_from_dev(feats_dev, y_nom_dev)

    # Target test cell (M2): 5 independent noise replicates + 1 held-out fitted replicate (seed=16)
    spec = MECHANISM_ZOO["M2"]
    states = trajectory_to_truth_states(solve_pbe(spec, times), tht_conc_mm=3.0, cavity_length_um=30.0)
    theta_m2 = np.array([np.log10(spec.k_n), np.log10(spec.k_plus), np.log10(spec.k_2), np.log10(spec.k_c)], dtype=float)

    noise_replicates = []
    for seed in (11, 12, 13, 14, 15):
        sim = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=8, seed=seed))
        feats = [
            extract_sweep_features(
                b.frames, b.wavelengths_nm, b.pump_energies_uj, spec_psf_fwhm_nm=b.spec_psf_fwhm_nm, dark_frames=b.dark_frames
            )
            for b in (sim.render_pump_sweep(s, 445.0, energies) for s in states)
        ]
        y_mat, _ = extract_kinetic_observable_series(feats, adapter=adapter)
        noise_replicates.append(y_mat)

    emp_std = estimate_kinetic_noise_std(noise_replicates, min_floors=(1.5, 0.08, 0.4))
    assert emp_std.shape == (len(times), 3) and np.all(emp_std > 0.0)

    sim_fit = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=8, seed=16))
    feats_fit = [
        extract_sweep_features(
            b.frames, b.wavelengths_nm, b.pump_energies_uj, spec_psf_fwhm_nm=b.spec_psf_fwhm_nm, dark_frames=b.dark_frames
        )
        for b in (sim_fit.render_pump_sweep(s, 445.0, energies) for s in states)
    ]
    y_fit, v_mask_fit = extract_kinetic_observable_series(feats_fit, adapter=adapter)

    theta_start = theta_m2 + np.array([0.2, -0.3, 0.2, -0.2])
    prof = profile_likelihood_log10_kp(
        y_fit,
        times,
        theta_start,
        cfg,
        kp_grid=np.linspace(-2.5, -0.5, 41),
        noise_std=emp_std,
        valid_mask=v_mask_fit,
    )
    dof = max(1, int(np.sum(v_mask_fit)) - 4)
    chi2_red = prof.chi2_min / dof
    assert chi2_red <= 3.0, f"Adapter misfit too large: chi2/dof = {chi2_red:.2f}"
    assert prof.status != "failed"
    assert abs(prof.mle_value - theta_m2[1]) < 0.8
