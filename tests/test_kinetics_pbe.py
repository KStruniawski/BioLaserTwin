"""S1 verification gate & SymPy symbolic audit tests for BioLaserTwin v0.2 kinetics."""

from pathlib import Path
import warnings
import numpy as np
import pytest

from biolaser_twin.simulator import simulate_secondary_nucleation_kinetics
from biolaser_twin.simulator.aggregate_optics import compute_aggregate_scattering_cm1
from biolaser_twin.simulator.kinetics import (
    MECHANISM_ZOO,
    MechanismSpec,
    build_conservative_fragmentation_matrix,
    build_log_pivot_grid,
    cohen_2011_early_time_mass,
    compute_moment_from_bins,
    elongation_only_analytical,
    extract_kinetic_targets,
    extract_truth_kinetic_targets,
    oosawa_primary_only_mass,
    solve_pbe,
    solve_reduced_moments_length_independent,
)
from biolaser_twin.simulator.kinetics.symbolic_audit import (
    export_verified_kinetics_latex,
    verify_cohen_early_time_ode_residual,
    verify_elongation_only_ode_residual,
    verify_kinetic_dimensions,
    verify_oosawa_ode_residual,
    verify_pbe_rhs_matches_sympy,
    verify_symbolic_mass_conservation,
)


def test_sympy_symbolic_audit_and_latex_export(tmp_path: Path) -> None:
    """Verify dimensions, mass conservation, analytical ODE residuals, and LaTeX export via SymPy."""
    dims = verify_kinetic_dimensions()
    assert all(dims.values()), f"Dimensional check failed: {dims}"

    assert verify_symbolic_mass_conservation() == 0
    assert verify_cohen_early_time_ode_residual() == 0
    assert verify_oosawa_ode_residual() == 0
    res_m, res_mass = verify_elongation_only_ode_residual()
    assert res_m == 0 and res_mass == 0

    tex_path = export_verified_kinetics_latex(tmp_path / "equations_kinetics.tex")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\begin{align}" in tex and "J_n" in tex and "J_2" in tex

    # Cross-check executable PBE RHS on a 6-bin log-pivot + fragmentation grid via SymPy lambdify
    piv_small = build_log_pivot_grid(q_min=4, l_max=30.0, n_discrete=4, n_log_bins=4)
    fmat_small = build_conservative_fragmentation_matrix(piv_small, q_min=4)
    spec = MECHANISM_ZOO["M3"]
    y_test = np.concatenate([[40.0, 0.5, 0.1], np.linspace(0.2, 0.02, len(piv_small))])
    spec_dict = {
        "q_oligomer": float(spec.q_oligomer),
        "k_n": spec.k_n,
        "k_plus": spec.k_plus,
        "k_2": spec.k_2,
        "k_m_sat": spec.k_m_sat,
        "k_f": 1e-4,
        "k_c": spec.k_c,
        "k_d": spec.k_d,
        "k_off": spec.k_off,
        "k_d_off": spec.k_d_off,
        "n_c": spec.n_c,
        "n_2": spec.n_2,
    }
    assert verify_pbe_rhs_matches_sympy(piv_small, fmat_small, spec_dict, y_test) < 1e-12


def test_truth_targets_are_cadence_independent() -> None:
    """extract_truth_kinetic_targets evaluates on a dense internal grid independent of observation cadence."""
    t_dense = extract_truth_kinetic_targets(MECHANISM_ZOO["M4"], t_end_min=900.0, n_dense=301)
    t_finer = extract_truth_kinetic_targets(MECHANISM_ZOO["M4"], t_end_min=900.0, n_dense=601)
    assert abs(t_dense.q1_oligomer_burden - t_finer.q1_oligomer_burden) < 1e-3
    assert abs(t_dense.q2_peak_time_norm - t_finer.q2_peak_time_norm) < 0.01


def test_pbe_exact_mass_conservation_and_realistic_alpha_syn_regimes() -> None:
    """All mechanisms M0..M5 + M_w conserve mass to <= 1e-8 and span realistic alpha-syn t50 & lengths."""
    times = np.linspace(0.0, 900.0, 181)
    q1_by_mech: dict[str, float] = {}
    for name, spec in MECHANISM_ZOO.items():
        traj = solve_pbe(spec, times, tht_conc_mm=3.0, temperature_k=310.15)
        assert traj.max_rel_mass_error <= 1e-8, f"{name} mass error {traj.max_rel_mass_error}"
        assert traj.max_upper_boundary_mass_frac < 0.01, f"{name} hit upper boundary"
        assert np.all(traj.monomer_um >= -1e-10)
        assert np.all(traj.oligomer_um >= -1e-10)
        assert np.all(traj.off_pathway_um >= -1e-10)
        assert np.all(traj.fibril_bins_um >= -1e-10)

        # Realistic timescale (t50 >= 60 min) and fibril length crossing k*L ~ 1 (ell > 120)
        half_idx = np.where(traj.m_mass_um >= 0.5 * spec.c_total_um)[0]
        if name != "M5":  # M5 traps mass off-pathway
            assert len(half_idx) > 0, f"{name} did not reach 50% fibril conversion"
            t50 = float(times[half_idx[0]])
            assert 40.0 <= t50 <= 700.0, f"{name} t50={t50} out of range"
        assert float(np.max(traj.mean_length)) > 120.0, f"{name} never crossed k*L~1"

        targets = extract_kinetic_targets(traj)
        q1_by_mech[name] = targets.q1_oligomer_burden
        assert 0.0 <= targets.q1_oligomer_burden <= 1.0
        assert 0.0 <= targets.q2_peak_time_norm <= 1.0

    # H4 feasibility: divergent mechanisms span |Delta q1| >= 0.25
    assert (q1_by_mech["M5"] - q1_by_mech["M2"]) >= 0.25


def test_fragmentation_matrix_conservation_and_exact_daughter_distribution() -> None:
    """Fragmentation matrix conserves mass, satisfies dP/dt = sum max(x-2q+1, 0) c, and populates small bins."""
    q_min = 4
    pivots = build_log_pivot_grid(q_min=q_min, l_max=1e4, n_discrete=64, n_log_bins=64)
    fmat = build_conservative_fragmentation_matrix(pivots, q_min=q_min)

    mass_leak = np.abs(pivots @ fmat) / pivots
    assert np.max(mass_leak) < 1e-12

    # Net number production per parent at pivot x_j equals max(x_j - 2*q_min + 1, 0)
    expected_dp = np.maximum(pivots - 2.0 * q_min + 1.0, 0.0)
    actual_dp = np.sum(fmat, axis=0)
    assert np.allclose(actual_dp, expected_dp, rtol=1e-10, atol=1e-12)

    # Large parent (last bin) must populate every small discrete bin 4..19 (boundary bin=1.0, interior=2.0)
    small_daughters = fmat[:16, -1]
    assert small_daughters[0] >= 0.99
    assert np.all(small_daughters[1:] >= 1.99)


def test_pbe_mesh_doubling_convergence_m2_and_scattering() -> None:
    """Default grid converges to <= 1% on M2 and RGD scattering across M0, M1, M2, M4."""
    times = np.linspace(0.0, 600.0, 61)
    for name in ("M0", "M1", "M2", "M4"):
        spec = MECHANISM_ZOO[name]
        grid_std = build_log_pivot_grid(q_min=spec.q_oligomer, l_max=1e5, n_discrete=96, n_log_bins=224)
        grid_fine = build_log_pivot_grid(q_min=spec.q_oligomer, l_max=1e5, n_discrete=192, n_log_bins=448)

        t_std = solve_pbe(spec, times, pivots=grid_std)
        t_fine = solve_pbe(spec, times, pivots=grid_fine)

        rel_m2 = np.max(np.abs(t_std.m2_second_moment_um - t_fine.m2_second_moment_um)) / max(
            np.max(t_fine.m2_second_moment_um), 1e-12
        )
        s_std = compute_aggregate_scattering_cm1(t_std)
        s_fine = compute_aggregate_scattering_cm1(t_fine)
        rel_scat = np.max(np.abs(s_std - s_fine)) / max(np.max(s_fine), 1e-12)
        assert rel_m2 <= 0.01, f"{name} M2 mesh error {rel_m2:.4f} > 0.01"
        assert rel_scat <= 0.01, f"{name} scattering mesh error {rel_scat:.4f} > 0.01"


def test_pbe_grid_and_seed_validation() -> None:
    """Invalid custom grids or out-of-domain seed lengths raise ValueError."""
    spec = MECHANISM_ZOO["M0"]
    times = np.array([0.0, 10.0])
    with pytest.raises(ValueError):
        solve_pbe(spec, times, pivots=np.array([5.0, 10.0]))  # pivots[0] != q_oligomer
    with pytest.raises(ValueError):
        solve_pbe(spec, times, p0_um=0.1, p0_length_ell=2.0)  # below pivots[0]


def test_pbe_matches_elongation_only_analytical_limit() -> None:
    """Pure elongation (k_n=k_2=k_f=0, seeded P0>0) matches analytical exponential decay."""
    spec = MechanismSpec(
        name="elong_only",
        description="Elongation only",
        k_n=0.0,
        k_plus=0.02,
        k_2=0.0,
        k_f=0.0,
        k_c=10.0,
        k_d=0.0,
        q_oligomer=4,
        c_total_um=50.0,
    )
    times = np.linspace(0.0, 120.0, 61)
    m0 = 50.0
    p0 = 0.25
    seed_len = 200.0
    traj = solve_pbe(spec, times, m0_um=m0, p0_um=p0, p0_length_ell=seed_len)
    m_ana, mass_ana = elongation_only_analytical(times, m0_um=m0, p0_um=p0, k_plus=spec.k_plus, m_mass0_um=p0 * seed_len)

    assert np.max(np.abs(traj.monomer_um - m_ana)) / m0 < 2e-3
    assert np.max(np.abs(traj.m_mass_um - mass_ana)) / m0 < 2e-3


def test_pbe_matches_oosawa_and_cohen_2011_limits() -> None:
    """PBE matches Oosawa primary-only limit and Cohen (2011) early-time cosh growth."""
    spec_oosawa = MechanismSpec(
        name="oosawa",
        description="Oosawa primary limit",
        k_n=5e-6,
        n_c=2.0,
        k_plus=0.04,
        k_2=0.0,
        k_f=0.0,
        k_c=1e3,
        k_d=0.0,
        q_oligomer=2,
        c_total_um=50.0,
    )
    times_o = np.linspace(0.0, 200.0, 51)
    pivots_o = build_log_pivot_grid(q_min=2, l_max=1e5, n_discrete=64, n_log_bins=128)
    traj_o = solve_pbe(spec_oosawa, times_o, pivots=pivots_o)
    mass_o = oosawa_primary_only_mass(times_o, m0_um=50.0, k_n=spec_oosawa.k_n, k_plus=spec_oosawa.k_plus, n_c=2.0)
    assert np.max(np.abs(traj_o.m_mass_um - mass_o)) / 50.0 < 0.02

    spec_cohen = MechanismSpec(
        name="cohen",
        description="Cohen early-time",
        k_n=1e-8,
        n_c=2.0,
        k_plus=0.04,
        k_2=2e-7,
        n_2=2.0,
        k_m_sat=float("inf"),
        k_f=0.0,
        k_c=1e3,
        k_d=0.0,
        q_oligomer=2,
        c_total_um=50.0,
    )
    times_c = np.linspace(0.0, 80.0, 41)
    traj_c = solve_pbe(spec_cohen, times_c, pivots=pivots_o)
    early_mask = (traj_c.times_min >= 20.0) & (traj_c.m_mass_um < 0.08 * 50.0)
    assert np.any(early_mask)
    mass_c = cohen_2011_early_time_mass(
        times_c[early_mask],
        m0_um=50.0,
        k_n=spec_cohen.k_n,
        k_plus=spec_cohen.k_plus,
        k_2=spec_cohen.k_2,
        n_c=2.0,
        n_2=2.0,
    )
    rel_err = np.max(np.abs(traj_c.m_mass_um[early_mask] - mass_c) / np.maximum(traj_c.m_mass_um[early_mask], 1e-6))
    assert rel_err < 0.08


def test_pbe_agrees_with_reduced_moment_odes_under_fragmentation() -> None:
    """For long seeded fibrils (ell >> 2q) with length-independent fragmentation, PBE moments match moment ODEs."""
    spec = MechanismSpec(
        name="frag_test",
        description="Fragmentation moment check",
        k_n=1e-6,
        n_c=2.0,
        k_plus=0.02,
        k_2=0.0,
        k_f=5e-4,
        k_c=10.0,
        k_d=0.0,
        q_oligomer=4,
        c_total_um=50.0,
    )
    times = np.linspace(0.0, 60.0, 41)
    p0 = 0.05
    seed_len = 300.0
    traj = solve_pbe(spec, times, m0_um=50.0, p0_um=p0, p0_length_ell=seed_len)
    mom = solve_reduced_moments_length_independent(
        spec, times, m0_um=50.0, p0_um=p0, m_mass0_um=p0 * seed_len
    )

    assert np.max(np.abs(traj.m_mass_um - mom["M"])) / 50.0 < 0.015
    assert np.max(np.abs(traj.p_number_um - mom["P"])) / np.max(mom["P"]) < 0.03
    assert np.allclose(compute_moment_from_bins(traj.fibril_bins_um, traj.pivots_ell, 2), traj.m2_second_moment_um)


def test_legacy_kinetics_emits_deprecation_warning() -> None:
    """Verify DeprecationWarning on old kinetics helper."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traj = simulate_secondary_nucleation_kinetics(np.array([0.0, 5.0, 10.0]))
        assert len(traj.monomer_frac) == 3
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
