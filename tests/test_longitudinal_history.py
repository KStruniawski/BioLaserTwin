"""Verification tests for persistent longitudinal SampleHistory and history=None regression."""

import hashlib
from pathlib import Path
import numpy as np

from biolaser_twin.simulator import (
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    LatentProxyVector,
    LongitudinalParams,
    SampleHistory,
    diffusive_refill_time_min,
    make_thermal_history_temperature_fn,
    map_latents_to_truth_state,
    relax_sample_history,
    update_sample_history,
)
from biolaser_twin.simulator.kinetics import MECHANISM_ZOO, solve_pbe
from biolaser_twin.simulator.kinetics.symbolic_audit import (
    export_verified_longitudinal_latex,
    verify_longitudinal_relaxation_limits,
)


def test_sympy_longitudinal_limits_and_latex(tmp_path: Path) -> None:
    """Verify longitudinal dt->oo relaxation limits and LaTeX export in SymPy."""
    d_inf, dt_inf = verify_longitudinal_relaxation_limits()
    assert d_inf == 0 and dt_inf == 0

    tex_path = export_verified_longitudinal_latex(tmp_path / "equations_longitudinal.tex")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\begin{align}" in tex and "D_k" in tex and "b_k" in tex


def test_history_none_matches_zero_history_and_v01_golden_hashes() -> None:
    """history=None matches zero SampleHistory at t=0 and preserves exact v0.1 golden frame hashes."""
    state = map_latents_to_truth_state(LatentProxyVector(0.3, 0.7, 1.345, 0.5, 30.0, 10.0))
    energies = np.linspace(5.0, 35.0, 6)

    expected_hashes = {
        0: "df2bfc402dadde2884f4785589c154d8ac538ea97f042ca3b463730f2439e1d5",
        1: "ed12682cc30bea3f4ceee32fdf64e500fa121d2451ae899bd42cb8d26a6c6b70",
        2: "3792b1bfeb879f78bdeb8b4a8fa79415732072a96365c7e071b9dae3b22c9198",
    }
    for level, golden in expected_hashes.items():
        cfg = FabryPerotConfig(cavity_length_um=30.0, level=level, n_spatial_px=8, seed=11)
        sim_a = FabryPerotTruthSimulator(cfg)
        sim_b = FabryPerotTruthSimulator(cfg)
        b_none = sim_a.render_pump_sweep(state, 445.0, energies, history=None)
        b_zero = sim_b.render_pump_sweep(state, 445.0, energies, history=SampleHistory(), current_time_min=0.0)
        assert np.array_equal(b_none.frames, b_zero.frames)
        digest = hashlib.sha256(np.ascontiguousarray(b_none.frames).tobytes()).hexdigest()
        assert digest == golden, f"v0.1 regression at level={level}: {digest} != {golden}"


def test_multi_sweep_longitudinal_history_persistence() -> None:
    """render_pump_sweep_with_history propagates actual jittered dose and viscosity-dependent refill."""
    cfg = FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=8, seed=19)
    state = map_latents_to_truth_state(LatentProxyVector(0.3, 0.7, 1.345, 0.5, 30.0, 10.0))
    energies = np.linspace(5.0, 35.0, 6)  # nominal sum = 120 uJ

    sim = FabryPerotTruthSimulator(cfg)
    b1, h1 = sim.render_pump_sweep_with_history(state, 445.0, energies, history=None, current_time_min=0.0)
    assert h1.last_time_min == 0.0
    assert 100.0 < h1.dose_uj < 140.0  # actual jittered dose ~120 uJ
    assert h1.bleach_factor == float(np.exp(-LongitudinalParams().bleach_beta_per_uj * h1.dose_uj))

    b2, h2 = sim.render_pump_sweep_with_history(state, 445.0, energies, history=h1, current_time_min=10.0)
    assert h2.last_time_min == 10.0
    assert h2.dose_uj > h1.dose_uj * float(np.exp(-10.0 / diffusive_refill_time_min(state.microviscosity_cp)))
    assert float(np.sum(b2.frames)) < float(np.sum(b1.frames))

    # Higher microviscosity slows diffusive refill
    params = LongitudinalParams(tau_rec_base_min=10.0, eta_ref_cp=1.0)
    r_low = relax_sample_history(h1, current_time_min=20.0, microviscosity_cp=1.0, params=params)
    r_high = relax_sample_history(h1, current_time_min=20.0, microviscosity_cp=20.0, params=params)
    assert r_high.dose_uj > r_low.dose_uj
    h_next = update_sample_history(h1, sweep_dose_uj=120.0, current_time_min=20.0, microviscosity_cp=5.0, params=params)
    assert h_next.dose_uj > 120.0

    # Verify Delta_T_k -> M_w Arrhenius thermal feedback alters M_w kinetics
    temp_fn = make_thermal_history_temperature_fn([SampleHistory(delta_temp_k=12.0, last_time_min=100.0)])
    t_grid = np.linspace(0.0, 400.0, 21)
    traj_iso = solve_pbe(MECHANISM_ZOO["M_w"], t_grid, tht_conc_mm=3.0, temperature_k=298.15)
    traj_heated = solve_pbe(MECHANISM_ZOO["M_w"], t_grid, tht_conc_mm=3.0, temperature_k=temp_fn)
    assert traj_heated.m_mass_um[-1] > traj_iso.m_mass_um[-1]
