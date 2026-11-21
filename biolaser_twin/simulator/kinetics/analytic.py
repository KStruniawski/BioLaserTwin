"""Exact and asymptotic analytical limits for PBE numerical verification."""

import numpy as np


def oosawa_primary_only_mass(
    times_min: np.ndarray,
    m0_um: float,
    k_n: float,
    k_plus: float,
    n_c: float = 2.0,
) -> np.ndarray:
    """Oosawa & Asakura (1975) analytical fibril mass M(t) for primary nucleation + elongation (m_0 >> M_nuc).

    lambda = sqrt(2 * k_plus * k_n * m0**n_c).
    M(t) = m0 * (1 - sech(sqrt(n_c / 2) * lambda * t)**(2 / n_c)).
    """
    t = np.asarray(times_min, dtype=float)
    lam = np.sqrt(2.0 * k_plus * k_n * (m0_um**n_c))
    arg = np.sqrt(n_c / 2.0) * lam * t
    sech = 1.0 / np.cosh(np.minimum(arg, 500.0))
    return m0_um * (1.0 - (sech ** (2.0 / n_c)))


def cohen_2011_early_time_mass(
    times_min: np.ndarray,
    m0_um: float,
    k_n: float,
    k_plus: float,
    k_2: float = 0.0,
    k_f: float = 0.0,
    n_c: float = 2.0,
    n_2: float = 2.0,
) -> np.ndarray:
    """Cohen et al. (2011) early-time linearized secondary-nucleation / fragmentation limit:

    M(t) ~ (lambda^2 / kappa^2) * m0 * (cosh(kappa * t) - 1),
    where:
      kappa = sqrt(2 * k_plus * m0 * (k_2 * m0^n_2 + k_f))   [min^-1],
      lambda = sqrt(2 * k_plus * k_n * m0^n_c)               [min^-1].
    """
    t = np.asarray(times_min, dtype=float)
    kappa = np.sqrt(2.0 * k_plus * m0_um * (k_2 * (m0_um**n_2) + k_f))
    lam = np.sqrt(2.0 * k_plus * k_n * (m0_um**n_c))
    if kappa < 1e-15:
        return 0.5 * (lam**2) * m0_um * (t**2)
    return ((lam**2) / (kappa**2)) * m0_um * (np.cosh(np.minimum(kappa * t, 500.0)) - 1.0)


def elongation_only_analytical(
    times_min: np.ndarray,
    m0_um: float,
    p0_um: float,
    k_plus: float,
    m_mass0_um: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact analytical monomer m(t) and fibril mass M(t) for pure elongation (k_n = k_2 = k_f = 0)."""
    t = np.asarray(times_min, dtype=float)
    rate = 2.0 * k_plus * p0_um
    m_t = m0_um * np.exp(-rate * t)
    mass_t = m_mass0_um + m0_um * (1.0 - np.exp(-rate * t))
    return m_t, mass_t
