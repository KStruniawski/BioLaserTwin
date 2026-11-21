"""Verification tests for RGD aggregate optics, mM ThT saturation, end-to-end lasing, and SymPy lambdify."""

from pathlib import Path
import numpy as np
import sympy as sp

from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.simulator import (
    AggregateOpticsParams,
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    compute_aggregate_scattering_cm1,
    compute_crowding_microviscosity_cp,
    compute_tht_binding,
    effective_gain_and_emission,
    rgd_rod_cross_section,
    rgd_sphere_cross_section,
    trajectory_to_truth_states,
)
from biolaser_twin.simulator.aggregate_optics import _rgd_rod_form_factor
from biolaser_twin.simulator.kinetics import MECHANISM_ZOO, solve_pbe
from biolaser_twin.simulator.kinetics.symbolic_audit import (
    export_verified_optics_latex,
    verify_rgd_rod_asymptotics,
    verify_s_sigma_scaling_symmetry,
    verify_tht_millimolar_saturation_limit,
)


def test_sympy_optics_asymptotics_lambdify_and_latex(tmp_path: Path) -> None:
    """Verify RGD limits, mM ThT expansion, s_sigma confounding, and SymPy lambdify cross-checks."""
    small_x, large_x = verify_rgd_rod_asymptotics()
    x = sp.Symbol("x", positive=True)
    assert sp.simplify(small_x - (1 - x**2 / 36)) == 0
    assert sp.simplify(large_x - sp.pi) == 0

    assert verify_tht_millimolar_saturation_limit() == 0

    # s_sigma scaling is NOT invariant in alpha_scat, only in dM/dt = 2*k_+*m*P
    res_a, res_dm = verify_s_sigma_scaling_symmetry()
    assert res_a != 0 and res_dm == 0

    # Cross-check compute_tht_binding against SymPy lambdify
    tht_s, s_s, kd_s = sp.symbols("ThT S K_d", positive=True)
    b_s = tht_s + s_s + kd_s
    cb_sym = sp.Rational(1, 2) * (b_s - sp.sqrt(b_s**2 - 4 * tht_s * s_s))
    cb_fn = sp.lambdify((tht_s, s_s, kd_s), cb_sym, "numpy")

    p = AggregateOpticsParams()
    m_arr = np.array([1.0, 25.0, 50.0, 500.0])
    o_arr = np.array([0.5, 2.0, 1.0, 5.0])
    cb_num, _ = compute_tht_binding(m_arr, o_arr, q_oligomer=4, tht_conc_mm=3.0, params=p)
    sites = p.binding_stoich_s * (m_arr + p.oligomer_tht_contrast * 4 * o_arr)
    assert np.allclose(cb_num, cb_fn(3000.0, sites, p.kd_tht_um), rtol=1e-10)

    # Cross-check _rgd_rod_form_factor against SymPy lambdify
    p_sym = (2 / x) * sp.Si(x) - (sp.sin(x / 2) / (x / 2)) ** 2
    p_fn = sp.lambdify(x, p_sym, "scipy")
    x_grid = np.array([0.01, 0.5, 2.0, 10.0, 50.0])
    assert np.allclose(_rgd_rod_form_factor(x_grid), p_fn(x_grid), rtol=1e-10)

    tex_path = export_verified_optics_latex(tmp_path / "equations_optics.tex")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\begin{align}" in tex and "c_b(t)" in tex


def test_rgd_rod_ell2_to_ell1_crossover_and_rayleigh_wavelength_prefactor() -> None:
    """RGD rod cross-section scales as ell^2 -> ell^1 and includes Rayleigh (lambda_ref/lambda)^4."""
    params = AggregateOpticsParams()
    wl = 485.0

    ell_short = np.array([4.0, 8.0, 16.0])
    sig_short = rgd_rod_cross_section(ell_short, wl, params)
    slope_short = float(np.polyfit(np.log(ell_short), np.log(sig_short), 1)[0])
    assert 1.95 <= slope_short <= 2.01

    ell_long = np.array([5.0e4, 1.0e5, 2.0e5])
    sig_long = rgd_rod_cross_section(ell_long, wl, params)
    slope_long = float(np.polyfit(np.log(ell_long), np.log(sig_long), 1)[0])
    assert 0.95 <= slope_long <= 1.05

    # Short-rod wavelength ratio obeys (600/450)^4
    sig_450 = rgd_rod_cross_section(np.array([4.0]), 450.0, params)[0]
    sig_600 = rgd_rod_cross_section(np.array([4.0]), 600.0, params)[0]
    assert abs((sig_450 / sig_600) - (600.0 / 450.0) ** 4) < 0.05


def test_rgd_sphere_volume_squared_scaling() -> None:
    """Spherical oligomer scattering scales as V^2 (q^2) for small oligomers."""
    params = AggregateOpticsParams()
    sig_4 = rgd_sphere_cross_section(4, 485.0, params)
    sig_8 = rgd_sphere_cross_section(8, 485.0, params)
    ratio = sig_8 / sig_4
    assert 3.9 <= ratio <= 4.05


def test_millimolar_tht_saturation_and_500um_finiteness() -> None:
    """Verify mM ThT binding, null-contrast rho_o=0, and finite optics/gain up to 500 uM."""
    m_mass = np.array([0.0, 10.0, 500.0])
    oligo = np.array([2.0, 2.0, 5.0])

    for tht_mm in (1.0, 3.0, 10.0):
        cb, fb = compute_tht_binding(m_mass, oligo, q_oligomer=4, tht_conc_mm=tht_mm)
        assert np.all(cb >= 0.0)
        assert np.all(fb < 0.15)  # bounded across [20, 500] uM

    # Null-contrast stress arm (rho_o = 0): oligomers produce zero bound ThT when M = 0
    params_null = AggregateOpticsParams(oligomer_tht_contrast=0.0)
    cb_null, fb_null = compute_tht_binding(np.zeros(3), oligo, q_oligomer=4, tht_conc_mm=3.0, params=params_null)
    assert np.allclose(cb_null, 0.0) and np.allclose(fb_null, 0.0)

    # Off-pathway oligomers are consistently mapped to binding and crowding viscosity
    cb_off, _ = compute_tht_binding(np.zeros(1), np.zeros(1), q_oligomer=4, tht_conc_mm=3.0, off_pathway_um=np.array([10.0]))
    eta_off = compute_crowding_microviscosity_cp(np.zeros(1), np.zeros(1), q_oligomer=4, off_pathway_um=np.array([10.0]))
    assert cb_off[0] > 0.0 and eta_off[0] > 1.0

    eta_500 = compute_crowding_microviscosity_cp(np.array([500.0]), np.array([0.0]), q_oligomer=4)
    assert np.isfinite(eta_500[0]) and eta_500[0] < 500.0


def test_end_to_end_pbe_to_fp_cavity_lases_at_445nm_mm_tht() -> None:
    """End-to-end chain (PBE -> aggregate_optics -> FabryPerotTruthSimulator) lases within 5-45 uJ."""
    times = np.linspace(0.0, 600.0, 13)
    traj = solve_pbe(MECHANISM_ZOO["M2"], times)
    states = trajectory_to_truth_states(traj, tht_conc_mm=3.0, cavity_length_um=30.0)

    # Check final fibril-rich state produces finite gain, spectral RGD, and lases in FP cavity
    final_state = states[-1]
    assert final_state.scattering_wavelengths_nm is not None
    assert final_state.scattering_loss_spectrum_cm1 is not None
    assert final_state.scattering_loss_spectrum_cm1[0] > final_state.scattering_loss_spectrum_cm1[-1]
    wl = np.linspace(465.0, 505.0, 128)
    spec, net_gain, e_th = effective_gain_and_emission(final_state, 445.0, 35.0, wl)
    assert np.all(np.isfinite(spec)) and np.all(np.isfinite(net_gain))
    assert 1.0 < e_th < 30.0

    sim = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=16, seed=7))
    energies = np.linspace(2.0, 45.0, 10)
    bundle = sim.render_pump_sweep(final_state, excitation_nm=445.0, pump_energies_uj=energies)
    feats = extract_sweep_features(
        bundle.frames,
        bundle.wavelengths_nm,
        bundle.pump_energies_uj,
        spec_psf_fwhm_nm=bundle.spec_psf_fwhm_nm,
        dark_frames=bundle.dark_frames,
        nominal_cavity_um=bundle.nominal_cavity_um,
    )
    assert feats.lasing_detected is True
    assert feats.threshold_energy_uj is not None and feats.threshold_energy_uj < 35.0
    assert np.all(np.isfinite(compute_aggregate_scattering_cm1(traj)))
