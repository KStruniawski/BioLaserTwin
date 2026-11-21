"""Illustrative Example: Simulating `spectra-96` cavity modes with UW DBR mirrors.

Reproduces the SGGW Fabry-Perot microcavity (`d = 33.22 um`, `n_g = 1.3524`)
coupled to late-stage M2 aggregation kinetics and digitized UW DBR mirror
reflectance/transmittance, then compares the extracted FSR (`485 nm` convention)
against the experimental `spectra-96` reference modes.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

from biolaser_twin.extractor import extract_sweep_features
from biolaser_twin.simulator import (
    AggregateOpticsParams,
    FabryPerotConfig,
    FabryPerotTruthSimulator,
    trajectory_to_truth_states,
)
from biolaser_twin.simulator.kinetics import MECHANISM_ZOO, solve_pbe
from biolaser_twin.simulator.fp_cavity import sggw_dbr_pump_transmittance


def main() -> None:
    exp_text = (
        resources.files("biolaser_twin.simulator")
        .joinpath("sggw_exp_lasing_spectrum_96.json")
        .read_text(encoding="utf-8")
    )
    exp_96 = json.loads(exp_text)
    exp_modes = np.array([m["wavelength_nm"] for m in exp_96["resolved_modes"]], dtype=float)
    exp_fsr_485 = float((485.0**2) * np.median(np.abs(np.diff(1.0 / exp_modes))))

    # Effective optical cavity length from experimental FSR at 505 nm (n_g * d = 44.922 um)
    n0 = 1.333
    n_g = n0 + 0.0194
    d_um = 44.922 / n_g

    # Late-stage aggregation state under mechanism M2
    traj = solve_pbe(MECHANISM_ZOO["M2"], np.array([0.0, 450.0]))
    state_late = trajectory_to_truth_states(
        traj,
        tht_conc_mm=3.0,
        cavity_length_um=d_um,
        params=AggregateOpticsParams(refractive_index=n0),
    )[1]

    # Digitized UW DBR pump transmittance at 433.3 nm
    eta_dbr_433 = sggw_dbr_pump_transmittance(433.3, dbr_shift_nm=0.0, use_exp=True)

    sim = FabryPerotTruthSimulator(
        FabryPerotConfig(
            cavity_length_um=d_um,
            refractive_index=n0,
            lambda_min_nm=495.5,
            lambda_max_nm=533.0,
            spec_psf_fwhm_nm=0.36,
            n_spatial_px=8,
            level=1,
            use_sggw_dbr=True,
            pump_jitter_rel=0.015,
            pump_coupling_eta=eta_dbr_433,
            seed=96,
        )
    )

    pump_ladder_uj = np.geomspace(20.0, 5000.0, 12)
    bundle = sim.render_pump_sweep(state_late, 433.3, pump_ladder_uj)
    feats = extract_sweep_features(
        bundle.frames,
        bundle.wavelengths_nm,
        bundle.pump_energies_uj,
        spec_psf_fwhm_nm=0.36,
        dark_frames=bundle.dark_frames,
    )

    rel_diff_pct = 100.0 * (feats.mean_fsr_nm - exp_fsr_485) / exp_fsr_485

    # Full 465-535 nm window comparison to report uncropped spectrum peak and CoM
    sim_full = FabryPerotTruthSimulator(
        FabryPerotConfig(
            cavity_length_um=d_um,
            refractive_index=n0,
            lambda_min_nm=465.0,
            lambda_max_nm=535.0,
            n_spectral_px=1918,
            spec_psf_fwhm_nm=0.36,
            n_spatial_px=8,
            level=1,
            use_sggw_dbr=True,
            pump_jitter_rel=0.015,
            pump_coupling_eta=eta_dbr_433,
            seed=96,
        )
    )
    b_full = sim_full.render_pump_sweep(state_late, 433.3, pump_ladder_uj)
    f_full = extract_sweep_features(
        b_full.frames,
        b_full.wavelengths_nm,
        b_full.pump_energies_uj,
        spec_psf_fwhm_nm=0.36,
        dark_frames=b_full.dark_frames,
    )
    high_full = np.mean(b_full.frames[-1], axis=0)
    full_peak_nm = float(b_full.wavelengths_nm[int(np.argmax(high_full))])

    print(f"Cavity length d       : {d_um:.4f} um (n_g = {n_g:.4f})")
    print(f"DBR pump T(433.3 nm)  : {eta_dbr_433:.4f}")
    print(f"Lasing detected       : {feats.lasing_detected}")
    print(f"Sim extracted FSR     : {feats.mean_fsr_nm:.4f} nm")
    print(f"Exp 485nm-conv FSR    : {exp_fsr_485:.4f} nm")
    print(f"Relative FSR diff     : {rel_diff_pct:+.2f}%")
    print(f"Full 465-535 nm peak  : {full_peak_nm:.1f} nm (CoM = {f_full.center_of_mass_nm:.1f} nm)")


if __name__ == "__main__":
    main()
