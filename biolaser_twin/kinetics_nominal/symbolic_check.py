"""Independent SymPy verification for nominal inverse kinetic priors (isolated from simulator/)."""

import sympy as sp


def verify_nominal_moment_conservation() -> sp.Expr:
    """Verify d/dt(m + q*O + M) == 0 for the nominal 4-moment ODE system."""
    m, o, p, mass = sp.symbols("m O P M", nonnegative=True)
    k_n, k_plus, k_2, k_c, k_d, q = sp.symbols("k_n_eff k_plus_eff k_2_eff k_c_eff k_d_eff q_nom", positive=True)

    j_n = k_n * m**2
    j_2 = k_2 * mass * m**2
    dm = -q * (j_n + j_2) + q * k_d * o - 2 * k_plus * m * p
    do = j_n + j_2 - (k_c + k_d) * o
    dmass = 2 * k_plus * m * p + q * k_c * o
    return sp.simplify(dm + q * do + dmass)


def verify_finke_watzky_ode_residual() -> sp.Expr:
    """Verify closed-form Finke-Watzky B(t) satisfies dB/dt = (k1 + k2*B)*(A0 - B)."""
    t, a0, k1, k2 = sp.symbols("t A0 k1 k2", positive=True)
    num = k1 / k2 + a0
    b_t = a0 - num / (1 + (k1 / (k2 * a0)) * sp.exp((k1 + k2 * a0) * t))
    res = sp.diff(b_t, t) - (k1 + k2 * b_t) * (a0 - b_t)
    return sp.simplify(res)


def verify_avrami_ode_residual() -> sp.Expr:
    """Verify Avrami M(t) satisfies dM/dt = n * k^n * t^(n-1) * (C_total - M)."""
    t, c_tot, k, n = sp.symbols("t C_total k n", positive=True)
    m_t = c_tot * (1 - sp.exp(-((k * t) ** n)))
    res = sp.diff(m_t, t) - n * (k**n) * (t ** (n - 1)) * (c_tot - m_t)
    return sp.simplify(res)
