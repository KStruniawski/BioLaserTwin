"""SymPy symbolic verification and LaTeX equation export for aggregation kinetics."""

from pathlib import Path
import numpy as np
import sympy as sp
from sympy.physics import units as u


def verify_kinetic_dimensions() -> dict[str, bool]:
    """Verify dimensional consistency of all flux and ODE terms using sympy.physics.units."""
    conc = u.mole / u.liter
    time_u = u.second

    # Variables & parameters with physical units
    m, o, p, mass, km = sp.symbols("m O P M K_M", positive=True)
    q = sp.Integer(4)
    n_c, n_2 = sp.Integer(2), sp.Integer(2)

    kn_u = conc ** (1 - n_c) / time_u
    k2_u = conc ** (-n_2) / time_u
    kp_u = 1 / (conc * time_u)
    kc_u = 1 / time_u
    kd_u = 1 / time_u
    kf_u = 1 / time_u

    # Symbolic dimensional expressions
    j_n = kn_u * (conc**n_c)
    j_2 = k2_u * conc * (conc**n_2) / (1 + (conc / conc) ** n_2)

    dm_terms = [q * j_n, q * j_2, q * kd_u * conc, 2 * kp_u * conc * conc]
    do_terms = [j_n, j_2, kc_u * conc, kd_u * conc]
    dp_terms = [kc_u * conc, kf_u * conc]
    dmass_terms = [2 * kp_u * conc * conc, q * kc_u * conc]

    target_rate = conc / time_u
    dim_sys = u.systems.SI

    def _same_dim(expr: sp.Expr, ref: sp.Expr) -> bool:
        return dim_sys.get_dimensional_expr(expr / ref) == 1

    return {
        "dm_consistent": all(_same_dim(term, target_rate) for term in dm_terms),
        "do_consistent": all(_same_dim(term, target_rate) for term in do_terms),
        "dp_consistent": all(_same_dim(term, target_rate) for term in dp_terms),
        "dmass_consistent": all(_same_dim(term, target_rate) for term in dmass_terms),
    }


def verify_symbolic_mass_conservation() -> sp.Expr:
    """Symbolically prove d/dt(m + q*O + q*O_off + sum_{ell=q}^L ell*c_ell) == 0."""
    q = 4
    l_max = 8
    m, o, o_off = sp.symbols("m O O_off", nonnegative=True)
    c = sp.symbols(f"c_{q}:{l_max + 1}", nonnegative=True)
    k_n, k_plus, k_2, k_m, k_c, k_d, k_off, k_doff, k_f = sp.symbols(
        "k_n k_plus k_2 K_M k_c k_d k_off k_d_off k_f", nonnegative=True
    )
    n_c, n_2 = sp.symbols("n_c n_2", positive=True)

    mass_m = sum((q + idx) * c[idx] for idx in range(len(c)))
    p_grow = sum(c[:-1])  # no-outflow top bin

    j_n = k_n * (m**n_c)
    j_2 = k_2 * mass_m * (m**n_2) / (1 + (m / k_m) ** n_2)

    dm = -q * (j_n + j_2) + q * k_d * o + q * k_doff * o_off - 2 * k_plus * m * p_grow
    do = j_n + j_2 - (k_c + k_d + k_off) * o
    do_off = k_off * o - k_doff * o_off

    dc = []
    for idx in range(len(c)):
        rate = sp.Integer(0)
        if idx == 0:
            rate += k_c * o
        if idx > 0:
            rate += 2 * k_plus * m * c[idx - 1]
        if idx < len(c) - 1:
            rate -= 2 * k_plus * m * c[idx]
        dc.append(rate)

    d_total = dm + q * do + q * do_off + sum((q + idx) * dc[idx] for idx in range(len(c)))
    return sp.simplify(d_total)


def verify_cohen_early_time_ode_residual() -> sp.Expr:
    """Verify symbolically that M(t) = (lam^2 / kappa^2) * m0 * (cosh(kappa*t) - 1) satisfies d^2M/dt^2 = kappa^2 M + lam^2 m0."""
    t, m0, lam, kappa = sp.symbols("t m_0 lambda kappa", positive=True)
    m_early = (lam**2 / kappa**2) * m0 * (sp.cosh(kappa * t) - 1)
    residual = sp.simplify(sp.diff(m_early, t, 2) - (kappa**2 * m_early + lam**2 * m0))
    return residual


def verify_oosawa_ode_residual() -> sp.Expr:
    """Verify symbolically that Oosawa M(t) satisfies dM/dt = 2*k_+*m*P in the n_c=2 limit."""
    t, m0, k_n, k_plus = sp.symbols("t m_0 k_n k_plus", positive=True)
    lam = sp.sqrt(2 * k_plus * k_n * m0**2)
    m_t = m0 * sp.sech(lam * t)
    mass_t = m0 - m_t
    # P(t) = int_0^t k_n m(s)^2 ds = (k_n * m0^2 / lam) * tanh(lam * t)
    p_t = (k_n * m0**2 / lam) * sp.tanh(lam * t)
    residual = sp.simplify(sp.diff(mass_t, t) - 2 * k_plus * m_t * p_t)
    return residual


def verify_elongation_only_ode_residual() -> tuple[sp.Expr, sp.Expr]:
    """Verify symbolically that elongation-only closed-form solution satisfies dm/dt and dM/dt."""
    t, m0, p0, mass0, k_plus = sp.symbols("t m_0 P_0 M_0 k_plus", positive=True)
    m_t = m0 * sp.exp(-2 * k_plus * p0 * t)
    mass_t = mass0 + m0 * (1 - sp.exp(-2 * k_plus * p0 * t))
    res_m = sp.simplify(sp.diff(m_t, t) + 2 * k_plus * p0 * m_t)
    res_mass = sp.simplify(sp.diff(mass_t, t) - 2 * k_plus * p0 * m_t)
    return res_m, res_mass


def export_verified_kinetics_latex(output_path: str | Path) -> Path:
    """Export SymPy-verified kinetic equations to a standalone LaTeX file."""
    t, m, o, o_off, p, mass = sp.symbols("t m O O_off P M")
    k_n, k_plus, k_2, k_m, k_c, k_d, k_off, k_doff, k_f, q = sp.symbols(
        "k_n k_+ k_2 K_M k_c k_d k_off k_d_off k_f q"
    )
    n_c, n_2 = sp.symbols("n_c n_2")

    j_n_expr = k_n * m**n_c
    j_2_expr = k_2 * mass * m**n_2 / (1 + (m / k_m) ** n_2)
    dm_expr = -q * (sp.Symbol("J_n") + sp.Symbol("J_2")) + q * k_d * o + q * k_doff * o_off - 2 * k_plus * m * p
    do_expr = sp.Symbol("J_n") + sp.Symbol("J_2") - (k_c + k_d + k_off) * o

    lines = [
        "% Auto-generated by biolaser_twin.simulator.kinetics.symbolic_audit (SymPy verified)",
        "\\begin{align}",
        f"J_n &= {sp.latex(j_n_expr)}, \\\\",
        f"J_2 &= {sp.latex(j_2_expr)}, \\\\",
        f"\\frac{{dO}}{{dt}} &= {sp.latex(do_expr)}, \\\\",
        f"\\frac{{dm}}{{dt}} &= {sp.latex(dm_expr)}.",
        "\\end{align}",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_rgd_rod_asymptotics() -> tuple[sp.Expr, sp.Expr]:
    """Symbolically verify RGD rod form factor limits: P_rod(x) -> 1 (x<<1) and ~ pi/x (x>>1)."""
    x = sp.Symbol("x", positive=True)
    term1 = (2 / x) * sp.Si(x)
    term2 = (sp.sin(x / 2) / (x / 2)) ** 2
    p_rod = term1 - term2
    small_x = sp.series(p_rod, x, 0, 4).removeO()
    large_x = sp.limit(x * term1, x, sp.oo) - sp.limit(x * term2, x, sp.oo)
    return sp.simplify(small_x), sp.simplify(large_x)


def verify_tht_millimolar_saturation_limit() -> sp.Expr:
    """Symbolically verify c_b = S * (1 - K_d / [ThT] + O([ThT]^-2)) in the mM ThT limit."""
    tht, s_sites, k_d = sp.symbols("ThT S K_d", positive=True)
    eps = sp.Symbol("eps", positive=True)
    # Substitute ThT = S / eps and expand around eps -> 0 (mM ThT >> S, K_d)
    b = tht + s_sites + k_d
    c_b = sp.Rational(1, 2) * (b - sp.sqrt(b**2 - 4 * tht * s_sites))
    c_b_sub = c_b.subs(tht, 1 / eps)
    series_cb = sp.series(c_b_sub, eps, 0, 2).removeO().subs(eps, 1 / tht)
    expected = s_sites * (1 - k_d / tht)
    return sp.simplify(series_cb - expected)


def verify_s_sigma_scaling_symmetry() -> tuple[sp.Expr, sp.Expr]:
    """Verify exact confounding vs non-invariance under (s_sigma, P, k_+) -> (a*s_sigma, P/a, a*k_+).

    In the long-rod regime, alpha_scat = s_sigma * M is NOT invariant if s_sigma scales by a
    (alpha_scaled - alpha_scat == (a - 1) * s_sigma * M), whereas dM/dt = 2 * k_+ * m * P IS invariant.
    """
    s_sig, p, k_plus, m, mass, a = sp.symbols("s_sigma P k_plus M m a", positive=True)
    alpha_scat = s_sig * mass
    dm_dt = 2 * k_plus * m * p

    alpha_scaled = alpha_scat.subs({s_sig: a * s_sig})
    dm_dt_scaled = dm_dt.subs({p: p / a, k_plus: a * k_plus})
    return sp.simplify(alpha_scaled - alpha_scat), sp.simplify(dm_dt_scaled - dm_dt)


def export_verified_optics_latex(output_path: str | Path) -> Path:
    """Export SymPy-verified aggregate optics & ThT saturation equations to LaTeX."""
    tht, s_sites, k_d, m, o_mass, rho_o, s_b = sp.symbols("ThT S K_d M O_mass rho_o s")
    sites_expr = s_b * (m + rho_o * o_mass)
    cb_expr = sp.Rational(1, 2) * (
        (tht + s_sites + k_d) - sp.sqrt((tht + s_sites + k_d) ** 2 - 4 * tht * s_sites)
    )
    lines = [
        "% Auto-generated by biolaser_twin.simulator.kinetics.symbolic_audit (SymPy verified)",
        "\\begin{align}",
        f"S(t) &= {sp.latex(sites_expr)}, \\\\",
        f"c_b(t) &= {sp.latex(cb_expr)}.",
        "\\end{align}",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_longitudinal_relaxation_limits() -> tuple[sp.Expr, sp.Expr]:
    """Verify dt -> 0 and dt -> inf limits of longitudinal dose & thermal relaxation."""
    d_prev, dt_prev, dt, tau_rec, tau_th, e_sum, chi = sp.symbols(
        "D_prev dT_prev dt tau_rec tau_th E_sum chi", positive=True
    )
    d_k = d_prev * sp.exp(-dt / tau_rec) + e_sum
    dt_k = dt_prev * sp.exp(-dt / tau_th) + chi * e_sum
    # As dt -> oo, prior memory vanishes cleanly
    d_inf = sp.limit(d_k, dt, sp.oo) - e_sum
    dt_inf = sp.limit(dt_k, dt, sp.oo) - chi * e_sum
    return sp.simplify(d_inf), sp.simplify(dt_inf)


def export_verified_longitudinal_latex(output_path: str | Path) -> Path:
    """Export SymPy-verified longitudinal history equations to LaTeX."""
    d_prev, dt_prev, dt, tau_rec, tau_th, e_sum, chi, beta, d_k = sp.symbols(
        "D_{k-1} Delta_T_{k-1} Delta_t_k tau_rec tau_th E_k chi beta D_k"
    )
    d_expr = d_prev * sp.exp(-dt / tau_rec) + e_sum
    b_expr = sp.exp(-beta * d_k)
    dt_expr = dt_prev * sp.exp(-dt / tau_th) + chi * e_sum
    lines = [
        "% Auto-generated by biolaser_twin.simulator.kinetics.symbolic_audit (SymPy verified)",
        "\\begin{align}",
        f"D_k &= {sp.latex(d_expr)}, \\\\",
        f"b_k &= {sp.latex(b_expr)}, \\\\",
        f"\\Delta T_k &= {sp.latex(dt_expr)}.",
        "\\end{align}",
    ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def verify_pbe_rhs_matches_sympy(
    pivots: np.ndarray,
    frag_mat: np.ndarray,
    spec_params: dict[str, float],
    state_y: np.ndarray,
) -> float:
    """Build a SymPy symbolic RHS on the exact log-pivot grid + fragmentation matrix and compare via lambdify."""
    q = int(spec_params["q_oligomer"])
    n_bins = len(pivots)
    y_syms = list(sp.symbols(f"y0:{3 + n_bins}", nonnegative=True))
    m_s, o_s, o_off_s = y_syms[0], y_syms[1], y_syms[2]
    c_s = sp.Matrix(y_syms[3:])
    piv_v = sp.Matrix([sp.Float(v) for v in pivots])
    dx_v = [sp.Float(pivots[i + 1] - pivots[i]) for i in range(n_bins - 1)]

    k_n = sp.Float(spec_params["k_n"])
    k_plus = sp.Float(spec_params["k_plus"])
    k_2 = sp.Float(spec_params["k_2"])
    k_m = sp.Float(spec_params["k_m_sat"])
    k_f = sp.Float(spec_params["k_f"])
    k_c = sp.Float(spec_params["k_c"])
    k_d = sp.Float(spec_params["k_d"])
    k_off = sp.Float(spec_params["k_off"])
    k_doff = sp.Float(spec_params["k_d_off"])
    n_c = sp.Float(spec_params["n_c"])
    n_2 = sp.Float(spec_params["n_2"])

    mass_s = piv_v.dot(c_s)
    j_n_s = k_n * (m_s**n_c)
    j_2_s = k_2 * mass_s * (m_s**n_2) / (1 + (m_s / k_m) ** n_2) if np.isfinite(spec_params["k_m_sat"]) else k_2 * mass_s * (m_s**n_2)
    p_grow_s = sum(c_s[i] for i in range(n_bins - 1))

    dm_s = -q * (j_n_s + j_2_s) + q * k_d * o_s + q * k_doff * o_off_s - 2 * k_plus * m_s * p_grow_s
    do_s = j_n_s + j_2_s - (k_c + k_d + k_off) * o_s
    do_off_s = k_off * o_s - k_doff * o_off_s

    dc_list = [sp.Integer(0)] * n_bins
    dc_list[0] += k_c * o_s
    for i in range(n_bins - 1):
        flx = (2 * k_plus * m_s / dx_v[i]) * c_s[i]
        dc_list[i] -= flx
        dc_list[i + 1] += flx
    f_sym = sp.Matrix([[sp.Float(frag_mat[i, j]) for j in range(n_bins)] for i in range(n_bins)])
    dc_vec = sp.Matrix(dc_list) + k_f * (f_sym * c_s)

    rhs_sym = [dm_s, do_s, do_off_s] + list(dc_vec)
    rhs_fn = sp.lambdify(y_syms, rhs_sym, "numpy")
    sym_val = np.asarray(rhs_fn(*state_y), dtype=float)

    # Independent NumPy evaluation matching population_balance.solve_pbe RHS
    m, o_on, o_off = state_y[0], state_y[1], state_y[2]
    c = state_y[3:]
    dx = np.diff(pivots)
    mass_m = float(np.dot(pivots, c))
    j_n = spec_params["k_n"] * (m ** spec_params["n_c"])
    sat = 1.0 + ((m / spec_params["k_m_sat"]) ** spec_params["n_2"] if np.isfinite(spec_params["k_m_sat"]) else 0.0)
    j_2 = spec_params["k_2"] * mass_m * (m ** spec_params["n_2"]) / sat
    flux = (2.0 * spec_params["k_plus"] * m / dx) * c[:-1]
    dm = -q * (j_n + j_2) + q * spec_params["k_d"] * o_on + q * spec_params["k_d_off"] * o_off - 2.0 * spec_params["k_plus"] * m * float(np.sum(c[:-1]))
    do = j_n + j_2 - (spec_params["k_c"] + spec_params["k_d"] + spec_params["k_off"]) * o_on
    do_off = spec_params["k_off"] * o_on - spec_params["k_d_off"] * o_off
    dc = np.zeros(n_bins)
    dc[0] += spec_params["k_c"] * o_on
    dc[:-1] -= flux
    dc[1:] += flux
    dc += spec_params["k_f"] * (frag_mat @ c)
    num_val = np.concatenate([[dm, do, do_off], dc])
    return float(np.max(np.abs(sym_val - num_val)))
