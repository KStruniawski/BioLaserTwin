"""Standalone numerical and analytical verification script for BioLaserTwin.

Reproduces the S1 verification checks directly from `biolaser_twin` without
requiring external benchmark scripts or cached JSON outputs:
1. Mass conservation across all 7 PBE mechanisms (M0-M5, M_w).
2. 2x pivot mesh-doubling convergence (M2, M4).
3. Cohen et al. (2011) early-time and Oosawa (1975) analytical limits.
4. RGD thin-rod scattering log-log slopes ([5, 20] and [400, 1600] monomers).
5. Bitwise equivalence for history=None across fidelity levels 0-2.
"""

from __future__ import annotations

import numpy as np

from biolaser_twin.simulator import (
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    rgd_rod_cross_section,
    trajectory_to_truth_states,
)
from biolaser_twin.simulator.kinetics import (
    MECHANISM_ZOO,
    build_log_pivot_grid,
    solve_pbe,
)
from biolaser_twin.simulator.kinetics.analytic import (
    cohen_2011_early_time_mass,
    oosawa_primary_only_mass,
)


def main() -> None:
    times = np.linspace(0.0, 600.0, 25)

    # 1. Mass conservation across all 7 mechanisms
    cons_errs: dict[str, float] = {}
    for name, spec in MECHANISM_ZOO.items():
        traj = solve_pbe(spec, times)
        cons_errs[name] = float(traj.max_rel_mass_error)
        assert cons_errs[name] < 1e-8, f"Mass conservation failed for {name}: {cons_errs[name]}"

    # 2. 2x mesh doubling on M2 and M4
    mesh_mass_errs: dict[str, float] = {}
    mesh_m2_errs: dict[str, float] = {}
    p_coarse = build_log_pivot_grid(n_discrete=96, n_log_bins=224)
    p_fine = build_log_pivot_grid(n_discrete=192, n_log_bins=448)
    for name in ("M2", "M4"):
        spec = MECHANISM_ZOO[name]
        coarse = solve_pbe(spec, times, pivots=p_coarse)
        fine = solve_pbe(spec, times, pivots=p_fine)
        mesh_mass_errs[name] = float(np.max(np.abs(coarse.m_mass_um - fine.m_mass_um)) / coarse.c_total_um)
        mesh_m2_errs[name] = float(
            np.max(np.abs(coarse.m2_second_moment_um - fine.m2_second_moment_um))
            / max(1.0, float(np.max(fine.m2_second_moment_um)))
        )
        assert mesh_m2_errs[name] <= 0.01, f"Mesh doubling M2 failed for {name}: {mesh_m2_errs[name]}"

    # 3. Cohen (2011) early-time and Oosawa (1975) limits
    t_an = np.linspace(0.0, 300.0, 16)
    spec_m2 = MECHANISM_ZOO["M2"]
    traj_cohen = solve_pbe(spec_m2, t_an)
    m_cohen = cohen_2011_early_time_mass(
        t_an, traj_cohen.c_total_um, spec_m2.k_n, spec_m2.k_plus, spec_m2.k_2
    )
    cohen_rel = float(np.max(np.abs(traj_cohen.m_mass_um[:6] - m_cohen[:6])) / traj_cohen.c_total_um)

    spec_m0 = MECHANISM_ZOO["M0"]
    traj_oos = solve_pbe(spec_m0, t_an)
    m_oos = oosawa_primary_only_mass(t_an, traj_oos.c_total_um, spec_m0.k_n, spec_m0.k_plus)
    oosawa_rel = float(np.max(np.abs(traj_oos.m_mass_um - m_oos)) / traj_oos.c_total_um)

    # 4. RGD l^2 -> l crossover slopes
    ells = np.array([5.0, 10.0, 20.0, 400.0, 800.0, 1600.0])
    sigmas = rgd_rod_cross_section(ells, 480.0)
    short_slope = float(np.polyfit(np.log10(ells[:3]), np.log10(sigmas[:3]), 1)[0])
    long_slope = float(np.polyfit(np.log10(ells[3:]), np.log10(sigmas[3:]), 1)[0])

    # 5. history=None bitwise check across levels 0, 1, 2
    state0 = trajectory_to_truth_states(solve_pbe(spec_m2, np.array([0.0, 300.0])), tht_conc_mm=3.0)[1]
    energies = np.linspace(5.0, 40.0, 6)
    bitwise_ok = True
    for lvl in (0, 1, 2):
        s1 = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=lvl, n_spatial_px=8, seed=7))
        s2 = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=lvl, n_spatial_px=8, seed=7))
        b1 = s1.render_pump_sweep(state0, 445.0, energies)
        b2, _ = s2.render_pump_sweep_with_history(state0, 445.0, energies, history=None)
        bitwise_ok = bitwise_ok and bool(np.array_equal(b1.frames, b2.frames))

    print(f"Max mass conservation rel err : {max(cons_errs.values()):.3e}")
    print(f"Max mesh-2x mass rel err      : {max(mesh_mass_errs.values()):.3e}")
    print(f"Mesh-2x M2 rel err (M2, M4)   : {mesh_m2_errs['M2']:.4f}, {mesh_m2_errs['M4']:.4f}")
    print(f"Cohen early rel err           : {cohen_rel:.3e}")
    print(f"Oosawa limit rel err          : {oosawa_rel:.3e}")
    print(f"RGD slopes (short, long)      : {short_slope:.3f}, {long_slope:.3f}")
    print(f"history=None bitwise (0..2)   : {bitwise_ok}")


if __name__ == "__main__":
    main()
