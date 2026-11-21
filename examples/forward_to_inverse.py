"""Three-stage API demonstration for BioLaserTwin v0.2.0.

Demonstrates three core package components:
1. Forward PBE simulation (M2) + 2D spatial-spectral cavity rendering & feature extraction.
2. Fisher Information Matrix (FIM) / Cramer-Rao Bound (CRB) check on optical latents.
3. Noisy multi-concentration nominal kinetic inversion with AmyloFitGlobalNLSSolver.
"""

from __future__ import annotations

import time
import numpy as np

from biolaser_twin.constants import NominalOpticalConstants
from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.identifiability import (
    DesignConfig,
    KineticObservableConfig,
    compute_fim,
    cramer_rao_sd,
    nominal_kinetic_forward_observables,
    numerical_jacobian,
)
from biolaser_twin.simulator import FabryPerotConfig, FabryPerotTruthSimulator, trajectory_to_truth_states
from biolaser_twin.simulator.kinetics import MECHANISM_ZOO, solve_pbe
from biolaser_twin.solvers import AmyloFitGlobalNLSSolver


def main() -> None:
    t0 = time.perf_counter()

    # 1. Forward PBE + FP cavity rendering + feature extraction
    traj = solve_pbe(MECHANISM_ZOO["M2"], np.array([0.0, 240.0]))
    state_late = trajectory_to_truth_states(traj, tht_conc_mm=3.0, cavity_length_um=30.0)[1]
    sim = FabryPerotTruthSimulator(FabryPerotConfig(cavity_length_um=30.0, level=1, n_spatial_px=8, seed=7))
    bundle = sim.render_pump_sweep(state_late, 445.0, np.geomspace(10.0, 300.0, 8))
    feats = extract_sweep_features(bundle.frames, bundle.wavelengths_nm, bundle.pump_energies_uj)

    # 2. Fisher Information Matrix & CRB check on optical latents
    design = DesignConfig(nominal_cavity_um=30.0, excitations_nm=(405.0, 440.0))
    cov_inv = np.diag(1.0 / (np.array([0.05, 0.25, 0.25, 0.15, 0.15, 0.02]) ** 2))
    jac = numerical_jacobian(np.array([40.0, 0.6, 0.3, 4.0]), design, NominalOpticalConstants())
    fim = compute_fim(jac, cov_inv)
    crb = cramer_rao_sd(fim)

    # 3. Noisy nominal kinetic inversion with AmyloFitGlobalNLSSolver (multi-m0)
    rng = np.random.default_rng(42)
    times = np.linspace(0.0, 360.0, 12)
    theta_true = np.array([-8.0, -3.5, -5.5, -2.5])  # log10(k_n, k_+, k_2, k_c)
    cfg1 = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, c_total_um=30.0)
    cfg2 = KineticObservableConfig(observable_set="full_lasing", s_sigma_known=True, c_total_um=60.0)
    y1_clean = nominal_kinetic_forward_observables(theta_true, times, cfg1)
    y2_clean = nominal_kinetic_forward_observables(theta_true, times, cfg2)
    y1 = y1_clean + rng.normal(0.0, 0.02 * np.maximum(1.0, np.abs(y1_clean)))
    y2 = y2_clean + rng.normal(0.0, 0.02 * np.maximum(1.0, np.abs(y2_clean)))

    theta_init = theta_true + np.array([0.25, -0.20, 0.30, -0.15])
    fits = AmyloFitGlobalNLSSolver(cfg1).fit_predict_multi([times, times], [y1, y2], [30.0, 60.0], theta_init=theta_init)
    theta_hat = fits[0].theta_log10
    err_l2 = float(np.linalg.norm(theta_hat - theta_true))

    print(f"Extracted sweep FSR     : {feats.mean_fsr_nm:.4f} nm (lasing={feats.lasing_detected})")
    print(f"Optical-latent FIM rank : {np.linalg.matrix_rank(fim)}/4 | max CRB = {np.max(crb):.4f}")
    print(f"NLS log10(k_n,k_+,k_2,k_c) true : {np.round(theta_true, 3).tolist()}")
    print(f"NLS log10(k_n,k_+,k_2,k_c) fit  : {np.round(theta_hat, 3).tolist()}")
    print(f"Noisy nominal L2 err    : {err_l2:.4e} (runtime {time.perf_counter() - t0:.2f} s)")


if __name__ == "__main__":
    main()
