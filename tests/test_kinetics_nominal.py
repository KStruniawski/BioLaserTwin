"""Verification & anti-inverse-crime isolation tests for biolaser_twin.kinetics_nominal."""

import ast
from pathlib import Path
import subprocess
import sys
import numpy as np

from biolaser_twin.kinetics_nominal import (
    NominalKineticParams,
    avrami_trajectory,
    finke_watzky_residual,
    finke_watzky_trajectory,
    integrate_nominal_moment_ode,
    monotone_oligomer_penalty,
    nominal_moment_ode_residual,
    smoothness_residual,
)
from biolaser_twin.kinetics_nominal.symbolic_check import (
    verify_avrami_ode_residual,
    verify_finke_watzky_ode_residual,
    verify_nominal_moment_conservation,
)


def test_kinetics_nominal_never_imports_simulator() -> None:
    """Enforce strict import isolation: kinetics_nominal must never import biolaser_twin.simulator."""
    pkg_dir = Path(__file__).resolve().parent.parent / "biolaser_twin" / "kinetics_nominal"
    for py_file in pkg_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "simulator" not in alias.name, f"Forbidden import {alias.name} in {py_file.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "simulator" not in mod, f"Forbidden import from {mod} in {py_file.name}"

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, biolaser_twin.kinetics_nominal; "
            "assert not any('simulator' in m for m in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_sympy_nominal_priors_and_conservation() -> None:
    """Verify nominal 4-moment conservation, Finke-Watzky, and Avrami ODE residuals via SymPy."""
    assert verify_nominal_moment_conservation() == 0
    assert verify_finke_watzky_ode_residual() == 0
    assert verify_avrami_ode_residual() == 0


def test_nominal_moment_ode_and_priors_numerics() -> None:
    """Verify nominal moment ODE conservation, active-window residuals, and Finke-Watzky k2=0 limit."""
    params = NominalKineticParams()
    times = np.linspace(0.0, 600.0, 201)
    out = integrate_nominal_moment_ode(times, params)
    total = out["m"] + params.q_nom * out["O"] + out["M"]
    assert np.max(np.abs(total - params.c_total_um)) / params.c_total_um < 1e-8

    res = nominal_moment_ode_residual(times, out["m"], out["O"], out["P"], out["M"], params)
    assert res.shape == (len(times), 4)
    # Check max residual over interior active window (excluding boundary 1-sided finite diff)
    assert np.max(np.abs(res[2:-2])) < 0.05

    b_fw = finke_watzky_trajectory(times, c_total_um=50.0, k1_per_min=1e-4, k2_per_um_min=1e-3)
    fw_res = finke_watzky_residual(times, b_fw, 50.0, 1e-4, 1e-3)
    assert np.max(np.abs(fw_res[2:-2])) < 0.05

    # Finke-Watzky k2=0 limit must start at 0 and follow 50*(1 - exp(-k1*t))
    b_k2_zero = finke_watzky_trajectory(times, c_total_um=50.0, k1_per_min=1e-3, k2_per_um_min=0.0)
    assert b_k2_zero[0] == 0.0
    assert np.allclose(b_k2_zero, 50.0 * (1.0 - np.exp(-1e-3 * times)), rtol=1e-10)

    m_av = avrami_trajectory(times, c_total_um=50.0, k_avrami=0.01, n_avrami=3.0)
    assert m_av[0] == 0.0 and m_av[-1] > 49.0

    assert np.sum(monotone_oligomer_penalty(out["O"])) > 0.0
    assert smoothness_residual(times, out["M"]).shape == times.shape
