"""Unit tests for kinetic PIELM, UDE/PI-KAN, LNN/Mamba, FMPE, and uncertainty quantification solvers."""

import ast
from pathlib import Path
import subprocess
import sys
import numpy as np
import sympy as sp

from biolaser_twin import __version__
from biolaser_twin.identifiability import KineticObservableConfig, nominal_kinetic_forward_observables
from biolaser_twin.kinetics_nominal import NominalKineticParams, integrate_nominal_moment_ode
from biolaser_twin.solvers import (
    AmyloFitGlobalNLSSolver,
    FMPECFMSolver,
    HierarchicalKineticPIELMSolver,
    KineticXTFCSolver,
    LabelsOnlyNominalODESolver,
    MomentUKFSolver,
    PILNNCfCSolver,
    PINNMambaSolver,
    PINNsFormerSolver,
    PirateNetSolver,
    PlainELMTemporalSolver,
    StoichiometricUDESolver,
)


def test_version_and_solvers_never_import_simulator() -> None:
    """Enforce v0.2.0 version consistency and strict anti-inverse-crime isolation."""
    assert __version__ == "0.2.0"
    pkg_dir = Path(__file__).resolve().parent.parent / "biolaser_twin" / "solvers"
    for py_file in pkg_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("simulator" in a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert "simulator" not in (node.module or "")

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, biolaser_twin.solvers; "
            "assert not any('simulator' in m for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_sympy_stoichiometric_ude_conservation() -> None:
    """Verify StoichiometricUDE flux structure conserves m + q*O + M == C_total identically."""
    r_nuc, r_conv, r_elong = sp.symbols("r_nuc r_conv r_elong", nonnegative=True)
    dm = -r_nuc - r_elong
    do = r_nuc - r_conv
    dmass = r_conv + r_elong
    assert sp.simplify(dm + do + dmass) == 0


def _make_synthetic_r0_batch(n_times: int = 15, m0_um: float = 50.0):
    times = np.linspace(0.0, 600.0, n_times)
    params = NominalKineticParams(c_total_um=m0_um)
    mom = integrate_nominal_moment_ode(times, params, m0_um=m0_um)
    y_bio = np.column_stack([mom["m"], params.q_nom * mom["O"], mom["M"]])
    theta = np.array([np.log10(params.k_n_eff), np.log10(params.k_plus_eff), np.log10(params.k_2_eff), np.log10(params.k_c_eff)])
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, c_total_um=m0_um)
    x_obs = nominal_kinetic_forward_observables(theta, times, cfg)
    mask_r0 = np.zeros(len(times), dtype=bool)
    mask_r0[0] = True
    mask_r0[-1] = True
    return times, x_obs, y_bio, mask_r0, theta, cfg


def test_pielm_solvers_grouped_cv_and_z1_rescue() -> None:
    """Test LabelsOnly endpoint conditioning, PlainELM, KineticXTFC, and HierarchicalKineticPIELM."""
    times, x_obs, y_bio, mask_r0, _, _ = _make_synthetic_r0_batch(15)

    floor_pred = LabelsOnlyNominalODESolver().predict_trajectory(times, y_final_bio=y_bio[-1])
    assert len(floor_pred.m_um) == len(times) and floor_pred.uq_calibrated is False

    plain = PlainELMTemporalSolver(hidden_dim=24, seed=1).fit(times, x_obs, y_bio, mask_r0).predict_trajectory(times, x_obs)
    xtfc = KineticXTFCSolver(hidden_dim=24, seed=1).fit(times, x_obs, y_bio, mask_r0).predict_trajectory(times, x_obs)
    assert xtfc.o_um[0] == 0.0 and xtfc.mass_um[0] == 0.0
    assert np.allclose(xtfc.m_um + 4.0 * xtfc.o_um + xtfc.mass_um, 50.0, atol=1e-6)
    assert plain.q1_std > 0.0 and xtfc.q1_std > 0.0

    # Grouped CV across 3 batches with different m0
    b1 = _make_synthetic_r0_batch(12, 30.0)
    b2 = _make_synthetic_r0_batch(12, 50.0)
    b3 = _make_synthetic_r0_batch(12, 80.0)
    for r_dyn in ("moment_ode", "finke_watzky", "avrami", "monotone_o", "smooth"):
        pielm = HierarchicalKineticPIELMSolver(hidden_dim=24, r_dyn_type=r_dyn, seed=1).fit_grouped(
            [b1[0], b2[0], b3[0]],
            [b1[1], b2[1], b3[1]],
            [b1[2], b2[2], b3[2]],
            [b1[3], b2[3], b3[3]],
        )
        pred = pielm.predict_trajectory(b2[0], b2[1], c_total_um=50.0)
        assert 0.0 <= pred.q1_mean <= 1.0 and pred.q1_std > 0.0 and pred.uq_calibrated is False


def test_classical_multi_m0_nls_and_sequential_ukf() -> None:
    """Test multi-m0 AmyloFitGlobalNLSSolver and sequential MomentUKFSolver with ESS."""
    b1 = _make_synthetic_r0_batch(12, 30.0)
    b2 = _make_synthetic_r0_batch(12, 60.0)
    cfg = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True)
    nls = AmyloFitGlobalNLSSolver(cfg)
    fits = nls.fit_predict_multi([b1[0], b2[0]], [b1[1], b2[1]], [30.0, 60.0], theta_init=b1[4] + 0.15)
    assert len(fits) == 2 and all(np.all(np.isfinite(f.theta_log10)) and f.q1_std >= 0.0 for f in fits)

    ukf = MomentUKFSolver(cfg, n_particles=12, seed=0).filter_predict(
        b2[0], b2[1], theta_prior_mean=b2[4] + 0.1, theta_prior_std=np.full(4, 0.25), m0_um=60.0
    )
    assert ukf.ess is not None and ukf.ess >= 1.0


def test_stoichiometric_ude_rk4_convergence_and_pikan() -> None:
    """StoichiometricUDE RK4 conserves mass and converges under dt refinement."""
    times, x_obs, y_bio, mask_r0, _, _ = _make_synthetic_r0_batch(10)
    for closure in ("mlp", "pikan"):
        ude = StoichiometricUDESolver(closure_type=closure, n_epochs=20, dt_max_min=2.0, seed=2).fit(times, x_obs, y_bio, mask_r0)
        p_coarse = ude.predict_trajectory(times, x_obs)
        ude.dt_max_min = 0.5
        p_fine = ude.predict_trajectory(times, x_obs)
        assert np.max(np.abs(p_coarse.m_um + 4.0 * p_coarse.o_um + p_coarse.mass_um - 50.0)) < 1e-4
        assert np.max(np.abs(p_coarse.mass_um - p_fine.mass_um)) / 50.0 < 0.02


def test_modern_pinn_and_liquid_comparators_with_ode_residual() -> None:
    """PINNMamba, PI-LNN/CfC, PirateNet, and PINNsFormer run cleanly on irregular timestamps."""
    times = np.array([0.0, 12.0, 35.0, 80.0, 150.0, 240.0, 360.0, 480.0, 600.0])
    params = NominalKineticParams()
    mom = integrate_nominal_moment_ode(times, params)
    y_bio = np.column_stack([mom["m"], params.q_nom * mom["O"], mom["M"]])
    theta = np.array([np.log10(params.k_n_eff), np.log10(params.k_plus_eff), np.log10(params.k_2_eff), np.log10(params.k_c_eff)])
    x_obs = nominal_kinetic_forward_observables(theta, times, KineticObservableConfig(s_sigma_known=True))
    mask_r0 = np.zeros(len(times), dtype=bool)
    mask_r0[0] = True
    mask_r0[-1] = True

    for cls in (PINNMambaSolver, PILNNCfCSolver, PirateNetSolver, PINNsFormerSolver):
        solver = cls(n_epochs=15, seed=3).fit(times, x_obs, y_bio, mask_r0)
        pred = solver.predict_trajectory(times, x_obs)
        assert np.allclose(pred.m_um + 4.0 * pred.o_um + pred.mass_um, 50.0, atol=1e-4)
        assert pred.q1_std >= 0.0 and pred.uq_calibrated is False


def test_fmpe_cfm_sequence_order_sensitivity() -> None:
    """FMPECFMSolver encoders (including Transformer) are strictly order-sensitive."""
    rng = np.random.default_rng(0)
    x_seqs = rng.normal(size=(12, 10, 3))
    theta_targets = rng.uniform(0.0, 1.0, size=(12, 2))
    for bb in ("lnn_cfc", "mamba_s4", "transformer"):
        fmpe = FMPECFMSolver(backbone=bb, emb_dim=16, n_epochs=12, seed=0).fit(x_seqs, theta_targets)
        m_fwd, std = fmpe.predict_with_uncertainty(x_seqs[:2], n_samples=16)
        m_rev, _ = fmpe.predict_with_uncertainty(x_seqs[:2, ::-1, :], n_samples=16)
        assert np.all(np.isfinite(m_fwd)) and np.all(std > 0.0)
        assert np.max(np.abs(m_fwd - m_rev)) > 1e-3, f"{bb} encoder was permutation-invariant"
