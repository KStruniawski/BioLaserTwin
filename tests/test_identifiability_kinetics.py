"""Unit tests for kinetic FIM, profile likelihood, null-space CRBs, and parameter pairs."""

import ast
from pathlib import Path
import numpy as np
import pytest
import sympy as sp

from biolaser_twin.identifiability import (
    KineticObservableConfig,
    find_indistinguishable_pair,
    integrate_nominal_5moment_ode,
    kinetic_crb_widths,
    nominal_kinetic_forward_observables,
    profile_likelihood_log10_kp,
)


def test_identifiability_never_imports_simulator() -> None:
    """Enforce strict anti-inverse-crime isolation: identifiability/ never imports simulator/."""
    pkg_dir = Path(__file__).resolve().parent.parent / "biolaser_twin" / "identifiability"
    for py_file in pkg_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("simulator" in a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert "simulator" not in (node.module or "")


def test_sympy_5moment_ode_conservation_and_pade_rgd_closure() -> None:
    """Verify 5-moment ODE conservation and SymPy Padé RGD closure limits + lambdify."""
    m, o, p, mass, m2, kn, kp, k2, kc, kd = sp.symbols("m O P M M2 kn kp k2 kc kd", positive=True)
    q = 4
    jn = kn * m**2
    j2 = k2 * mass * m**2
    dm = -q * (jn + j2) + q * kd * o - 2 * kp * m * p
    do = jn + j2 - (kc + kd) * o
    dmass = 2 * kp * m * p + q * kc * o
    assert sp.simplify(dm + q * do + dmass) == 0

    # Padé RGD closure: sigma_fib = s_sigma * M2 / (1 + ell_w / ell_c) with ell_w = M2 / M
    s_sig, ell_w, ell_c = sp.symbols("s_sigma ell_w ell_c", positive=True)
    m2_sym, m_sym = sp.symbols("M2 M", positive=True)
    # Short-rod limit (ell_w -> 0 at fixed M2): recovers s_sigma * M2
    short_lim = sp.limit(s_sig * m2_sym / (1 + ell_w / ell_c), ell_w, 0)
    assert sp.simplify(short_lim - s_sig * m2_sym) == 0
    # Long-rod limit (M2 = ell_w * M, ell_w -> oo): collapses purely onto s_sigma * ell_c * M
    long_lim = sp.limit(s_sig * (ell_w * m_sym) / (1 + ell_w / ell_c), ell_w, sp.oo)
    assert sp.simplify(long_lim - s_sig * ell_c * m_sym) == 0

    # Dimensional check + lambdify cross-check against production nominal_kinetic_forward_observables
    from sympy.physics import units as u

    conc_u = u.mole / u.liter
    assert u.systems.SI.get_dimensional_expr((conc_u / conc_u) * conc_u / conc_u) == 1
    times_t = np.array([0.0, 200.0])
    th_t = np.array([-8.5, -1.3, -7.0, -0.5])
    cfg_t = KineticObservableConfig(observable_set="e_th", s_sigma_known=True)
    mom_t = integrate_nominal_5moment_ode(times_t, *(10.0 ** th_t))
    fn_rgd = sp.lambdify((s_sig, m2_sym, m_sym, ell_c), s_sig * m2_sym / (1 + (m2_sym / m_sym) / ell_c), "numpy")
    rgd_expected = fn_rgd(1.0, mom_t["M2"][1], mom_t["M"][1], cfg_t.crossover_ell) + 16.0 * mom_t["O"][1]
    from biolaser_twin.constants import NominalOpticalConstants

    consts = NominalOpticalConstants()
    scat_exp = consts.scat_scale_nominal * rgd_expected
    mirror_l = -np.log(consts.mirror_reflectivity_nominal**2) / (2.0 * 30.0e-4)
    gain_d = max(0.15, 0.25 * (mom_t["M"][1] + 0.15 * 4.0 * mom_t["O"][1]) + 0.05 * cfg_t.tht_conc_mm)
    eth_prod = nominal_kinetic_forward_observables(th_t, times_t, cfg_t)[1, 0]
    assert np.isclose(eth_prod, (mirror_l + scat_exp) / gain_d, rtol=1e-10)


def test_singular_fim_returns_inf_and_truncated_window_preserves_t0() -> None:
    """Singular FIM [[1,1],[1,1]] returns inf CRB widths; truncated window integrates from t0_min=0."""
    fim_sing = np.array([[1.0, 1.0], [1.0, 1.0]])
    widths = kinetic_crb_widths(fim_sing)
    assert np.all(np.isinf(widths))

    mom_full = integrate_nominal_5moment_ode(np.array([0.0, 300.0, 600.0]), 1e-8, 0.05, 2e-7, 0.3, t0_min=0.0)
    mom_late = integrate_nominal_5moment_ode(np.array([300.0, 600.0]), 1e-8, 0.05, 2e-7, 0.3, t0_min=0.0)
    assert np.allclose(mom_late["M"], mom_full["M"][1:], rtol=1e-7)

    # Dimension enforcement on s_sigma_known=False
    cfg_unk = KineticObservableConfig(s_sigma_known=False)
    with pytest.raises(ValueError):
        nominal_kinetic_forward_observables(np.array([-8.5, -1.3, -7.0, -0.5]), np.array([0.0, 300.0]), cfg_unk)


def test_profile_likelihood_handles_empty_boundary_and_resolved_ci() -> None:
    """Profile likelihood never reports ci_width=0 on empty/missing region and resolves interior CIs."""
    times = np.linspace(0.0, 600.0, 25)
    theta_true = np.array([-8.5, -1.3, -7.0, -0.5], dtype=float)
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, noise_std_rel=0.02, noise_std_abs=0.1)
    y_obs = nominal_kinetic_forward_observables(theta_true, times, cfg)

    # Grid missing the MLE must report status="failed" and inf width, never 0.0
    prof_miss = profile_likelihood_log10_kp(y_obs, times, theta_true, cfg, kp_grid=np.array([-3.5, -3.0]))
    assert prof_miss.status == "failed" and np.isinf(prof_miss.ci_width) and prof_miss.bounded_within_1_decade is False

    # Well-resolved interior grid vs coarse 2-point grid (auto-subdivided <=0.05) give consistent brentq CI width
    prof_ok = profile_likelihood_log10_kp(y_obs, times, theta_true, cfg, kp_grid=np.linspace(-2.5, -0.3, 13))
    prof_coarse = profile_likelihood_log10_kp(y_obs, times, theta_true, cfg, kp_grid=np.array([-2.5, -0.3]))
    assert prof_ok.status == "resolved" and prof_ok.ci_width < 1.0 and prof_ok.bounded_within_1_decade is True
    assert abs(prof_ok.ci_width - prof_coarse.ci_width) < 0.05


def test_cmaes_indistinguishable_pair_feasibility() -> None:
    """CMA-ES find_indistinguishable_pair enforces strict feasibility flag."""
    times = np.linspace(0.0, 600.0, 17)
    theta_ref = np.array([-8.0, -1.3, -7.0, -1.5], dtype=float)
    cfg = KineticObservableConfig(observable_set="mass_only", s_sigma_known=True, nominal_rho_o=0.0)
    pair = find_indistinguishable_pair(theta_ref, times, cfg, optical_rel_tol=0.15, min_delta_q1=0.05, max_iters=15, pop_size=10, seed=7)
    assert isinstance(pair.feasible, bool)
    if pair.feasible:
        assert pair.rel_optical_distance <= 0.15 and pair.abs_delta_q1 >= 0.05
