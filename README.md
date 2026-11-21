# BioLaserTwin (`v0.2.0`)

**BioLaserTwin** is an open-source Python software package for physics-based simulation of biophotonic Fabry–Pérot (FP) microcavity lasers coupled to amyloid-like protein aggregation kinetics, 2D spatial–spectral feature extraction, Fisher/profile-likelihood identifiability analysis, and multi-paradigm inverse parameter estimation.

- **Author (Software):** Karol Struniawski, PhD (Warsaw University of Life Sciences — WULS, `karol_struniawski@sggw.edu.pl`)
- **Experimental Reference Data (`simulator/sggw_*.json`):** Measured and provided by the study co-authors (K. Struniawski, A. Konopka, P. Hańczyc, K. Rusakov), redistributed under the MIT License ([`LICENSE`](LICENSE)).
- **Funding:** National Science Centre (NCN), Poland (SONATINA grant No. `UMO-2026/60/C/ST7/00025`)
- **License:** MIT ([`LICENSE`](LICENSE))

---

## Architecture

The package is organized into five decoupled subpackages under `biolaser_twin/`:

1. **`biolaser_twin.simulator`** — Independent forward physics engine (isolated from solvers and reduced nominal models):
   - `kinetics/`: Kumar–Ramkrishna discrete-plus-geometric-pivot Population Balance Equation (PBE) solver (`96` discrete + `224` log pivots) with stiff `Radau` integration, analytical Jacobian, and SymPy symbolic audit across 7 mechanisms (`M0`–`M5`, `M_w`).
   - `aggregate_optics.py`: Rayleigh–Gans–Debye (RGD) thin-rod scattering crossover ($\ell^2 \to \ell$), Thioflavin-T (ThT) binding, and crowding microviscosity.
   - `fp_cavity.py` & `photophysics.py`: Dispersive Fabry–Pérot Airy cavity, pump-threshold lasing dynamics, and digitized University of Warsaw (UW) dielectric Bragg reflector (DBR) spectra (`simulator/sggw_dbr_mirror_spectra.json`, `simulator/sggw_exp_lasing_spectrum_96.json`).
   - `longitudinal.py`: Multi-shot thermal and photobleaching sample history.
2. **`biolaser_twin.kinetics_nominal`** — Reduced 4-moment ODE models, priors, and SymPy symbolic checks (intentionally isolated from `simulator/`).
3. **`biolaser_twin.extractor`** — 2D spatial–spectral frame feature extractor (`frame_features.py`): peak detection, threshold estimation, center-of-mass wavelength, and Free Spectral Range (FSR) referenced to `485 nm`.
4. **`biolaser_twin.identifiability`** — Fisher Information Matrix (FIM), optimal Design of Experiments (DoE), and quasi-likelihood profile-likelihood observability diagnostics.
5. **`biolaser_twin.solvers`** — Inverse solvers spanning Physics-Informed Extreme Learning Machines (`kinetic_pielm.py`, `photonic_pielm.py`), classical NLS / tempered SMC (`kinetic_classical.py`), Neural/KAN UDEs (`kinetic_ude.py`), Simulation-Based Inference (`kinetic_sbi.py`), and modern SciML architectures (`kinetic_modern.py`).

---

## Installation

Python `>=3.10` (tested on `3.11.0`) is required. Note that `numpy<2.4.0` is pinned for compatibility with legacy `np.trapz` calls in benchmark scripts (core `biolaser_twin` modules support both `np.trapezoid` and `np.trapz`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Verification & Testing

Run the full unit and symbolic verification test suite (`59` tests; ~28 min, `1708.39 s` in reference log):

```bash
pytest
```

For a fast smoke check (~1 s; `0.73 s` in reference log) of analytical cavity limits, `DesignConfig` masks, and API regression tests:

```bash
pytest tests/test_verification.py
```

For the standalone numerical/analytical verification suite (~5 min, checking mass conservation across all 7 PBE mechanisms, $2\times$ mesh-doubling, Cohen/Oosawa limits, RGD slopes, and `history=None` equivalence):

```bash
python3 examples/verify_s1.py
```

Representative output (floating-point values and runtimes are machine-dependent):
`Max mass conservation rel err : ~2e-15`, `Max mesh-2x mass rel err : 2.997e-10`, `Mesh-2x M2 rel err : 0.0086, 0.0085`, `Cohen : 5.909e-04`, `Oosawa : 1.527e-03`, `RGD slopes : 1.999, 1.236`, `history=None bitwise : True`.

---

## Illustrative Examples

1. **`spectra-96` mode-spacing consistency check (`examples/illustrative_spectra96.py`):** Simulates a $d = 33.22\,\mu\text{m}$ Fabry–Pérot microcavity with digitized UW DBR mirror spectra and compares extracted FSR (`2.6430 nm`) against the `spectra-96` reference spectrum (`2.6115 nm` in the `485 nm` convention, `+1.21%`; full-window `465–535 nm` peak `478.8 nm`, CoM `482.0 nm`):

```bash
python3 examples/illustrative_spectra96.py
```

2. **Three-stage API demonstration (`examples/forward_to_inverse.py`):** Demonstrates forward cavity rendering + feature extraction (`FSR = 2.8313 nm`), optical-latent FIM/CRB rank (`4/4`), and `AmyloFitGlobalNLSSolver.fit_predict_multi` on noisy multi-concentration nominal kinetics (`2%` seeded noise; representative output: `fit ≈ [-8.496, -3.5, -6.046, -1.757]`, `L2 ≈ 1.05`, runtime `< 10 s`):

```bash
python3 examples/forward_to_inverse.py
```

---

## Citation

If you use **BioLaserTwin** in academic work, please cite it using [`CITATION.cff`](CITATION.cff).
