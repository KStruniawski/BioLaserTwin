"""Physics-informed ELM, XTFC, and hierarchical kinetic inversion solvers (isolated from simulator/)."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

from ..kinetics_nominal import (
    NominalKineticParams,
    integrate_nominal_moment_ode,
)


def _rff_features(t_norm: np.ndarray, x_obs: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    inp = np.column_stack([t_norm, x_obs])
    return np.tanh(inp @ w + b)


@dataclass
class KineticPrediction:
    """Predicted biological state trajectories, microviscosity z1(t), and uncertainty-propagated (q1, q2)."""

    m_um: np.ndarray
    o_um: np.ndarray
    mass_um: np.ndarray
    q1_mean: float
    q1_std: float
    q2_mean: float
    q2_std: float
    selected_lambda_dyn: float = 0.0
    z1_pred: np.ndarray | None = None
    uq_calibrated: bool = False


def _targets_from_o(times_min: np.ndarray, o_um: np.ndarray, c_total_um: float, q_nom: int = 4) -> tuple[float, float]:
    t = np.asarray(times_min, dtype=float)
    t_span = max(float(t[-1] - t[0]), 1e-9)
    o_pos = np.maximum(0.0, o_um)
    area = float(np.trapezoid(q_nom * o_pos, t) if hasattr(np, "trapezoid") else np.trapz(q_nom * o_pos, t))
    q1 = float(np.clip(area / (t_span * max(c_total_um, 1e-9)), 0.0, 1.0))
    q2 = float((t[int(np.argmax(o_pos))] - t[0]) / t_span)
    return q1, q2


def _z1_from_states(mass_um: np.ndarray, o_um: np.ndarray, c_total_um: float, q_nom: int = 4) -> np.ndarray:
    """Infer normalized microviscosity proxy z1(t) in [0, 1] from kinetic crowding."""
    eff = np.maximum(0.0, mass_um + 0.5 * q_nom * o_um) / max(c_total_um, 1e-9)
    return np.clip(eff, 0.0, 1.0)


class LabelsOnlyNominalODESolver:
    """R0 endpoint-conditioned floor: fits nominal ODE to (t=0, t_final) biological labels without optics."""

    def __init__(self, params: NominalKineticParams | None = None) -> None:
        self.params = params or NominalKineticParams()

    def predict_trajectory(
        self,
        times_min: np.ndarray,
        m0_um: float | None = None,
        y_final_bio: np.ndarray | None = None,
    ) -> KineticPrediction:
        m_init = self.params.c_total_um if m0_um is None else float(m0_um)
        best_params = self.params
        if y_final_bio is not None and len(y_final_bio) == 3:
            target_mass = float(y_final_bio[2])

            def _obj(log_scale: float) -> float:
                sc = 10.0**log_scale
                p_try = NominalKineticParams(
                    k_n_eff=self.params.k_n_eff * sc,
                    k_plus_eff=self.params.k_plus_eff,
                    k_2_eff=self.params.k_2_eff * sc,
                    k_c_eff=self.params.k_c_eff,
                    k_d_eff=self.params.k_d_eff,
                    q_nom=self.params.q_nom,
                    c_total_um=m_init,
                )
                out = integrate_nominal_moment_ode(times_min, p_try, m0_um=m_init)
                return float((out["M"][-1] - target_mass) ** 2)

            res = minimize_scalar(_obj, bounds=(-2.0, 2.0), method="bounded")
            sc_opt = 10.0 ** float(res.x)
            best_params = NominalKineticParams(
                k_n_eff=self.params.k_n_eff * sc_opt,
                k_plus_eff=self.params.k_plus_eff,
                k_2_eff=self.params.k_2_eff * sc_opt,
                k_c_eff=self.params.k_c_eff,
                k_d_eff=self.params.k_d_eff,
                q_nom=self.params.q_nom,
                c_total_um=m_init,
            )

        out = integrate_nominal_moment_ode(times_min, best_params, m0_um=m_init)
        q1, q2 = _targets_from_o(times_min, out["O"], m_init, best_params.q_nom)
        return KineticPrediction(
            m_um=out["m"],
            o_um=out["O"],
            mass_um=out["M"],
            q1_mean=q1,
            q1_std=0.20,
            q2_mean=q2,
            q2_std=0.20,
            z1_pred=None,
            uq_calibrated=False,
        )


class PlainELMTemporalSolver:
    """Unconstrained temporal random-feature ELM with Bayesian weight-posterior trajectory UQ."""

    def __init__(self, hidden_dim: int = 48, reg_lambda: float = 1e-2, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.seed = seed
        self._w: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._cov_post: np.ndarray | None = None
        self._sigma2: float = 1.0
        self._t0: float = 0.0
        self._t_scale: float = 1.0
        self._c_total: float = 50.0

    def fit(self, times_min: np.ndarray, x_obs: np.ndarray, y_bio: np.ndarray, mask_labeled: np.ndarray) -> "PlainELMTemporalSolver":
        rng = np.random.default_rng(self.seed)
        self._t0 = float(times_min[0])
        self._t_scale = max(float(times_min[-1] - self._t0), 1.0)
        t_norm = (times_min - self._t0) / self._t_scale
        in_dim = 1 + x_obs.shape[1]
        self._w = rng.normal(0.0, 0.8, size=(in_dim, self.hidden_dim))
        self._b = rng.uniform(-1.0, 1.0, size=self.hidden_dim)

        h_all = _rff_features(t_norm, x_obs, self._w, self._b)
        h_lab = h_all[mask_labeled]
        y_lab = y_bio[mask_labeled]
        self._c_total = float(np.sum(y_lab[0])) if len(y_lab) > 0 else 50.0

        precision = h_lab.T @ h_lab + self.reg_lambda * np.eye(self.hidden_dim)
        self._w_out = np.linalg.solve(precision, h_lab.T @ y_lab)
        res = y_lab - h_lab @ self._w_out
        self._sigma2 = max(0.25, float(np.mean(res**2)))
        self._cov_post = self._sigma2 * np.linalg.inv(precision)
        return self

    def predict_trajectory(self, times_min: np.ndarray, x_obs: np.ndarray, q_nom: int = 4, n_mc: int = 64) -> KineticPrediction:
        assert self._w is not None and self._b is not None and self._w_out is not None and self._cov_post is not None
        t_norm = (times_min - self._t0) / self._t_scale
        h = _rff_features(t_norm, x_obs, self._w, self._b)
        pred = np.maximum(0.0, h @ self._w_out)
        m_um = pred[:, 0]
        o_um = pred[:, 1] / max(q_nom, 1)
        mass_um = pred[:, 2]
        q1, q2 = _targets_from_o(times_min, o_um, self._c_total, q_nom)

        rng = np.random.default_rng(self.seed + 77)
        q1s, q2s = [], []
        chol = np.linalg.cholesky(self._cov_post + 1e-8 * np.eye(self.hidden_dim))
        for _ in range(n_mc):
            w_s = self._w_out + chol @ rng.normal(size=(self.hidden_dim, 3))
            p_s = np.maximum(0.0, h @ w_s)
            q1_i, q2_i = _targets_from_o(times_min, p_s[:, 1] / max(q_nom, 1), self._c_total, q_nom)
            q1s.append(q1_i)
            q2s.append(q2_i)

        return KineticPrediction(
            m_um=m_um,
            o_um=o_um,
            mass_um=mass_um,
            q1_mean=q1,
            q1_std=float(np.std(q1s)),
            q2_mean=q2,
            q2_std=float(np.std(q2s)),
            z1_pred=None,
            uq_calibrated=False,
        )


class KineticXTFCSolver:
    """Theory of Functional Connections (XTFC) ELM: exact ICs with non-zero initial slope + mass conservation."""

    def __init__(
        self,
        hidden_dim: int = 48,
        reg_lambda: float = 1e-2,
        lambda_dyn: float = 0.2,
        params: NominalKineticParams | None = None,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.reg_lambda = reg_lambda
        self.lambda_dyn = lambda_dyn
        self.params = params or NominalKineticParams()
        self.seed = seed
        self._w: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._cov_post: np.ndarray | None = None
        self._t0: float = 0.0
        self._t_scale: float = 1.0
        self._c_total: float = self.params.c_total_um

    def _xtfc_basis(self, t_norm: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        assert self._w is not None and self._b is not None
        h_t = _rff_features(t_norm, x_obs, self._w, self._b)
        # Single switching factor (t_norm) gives O(0)=0, M(0)=0 while allowing non-zero dO/dt(0) > 0
        return t_norm[:, None] * h_t

    def fit(self, times_min: np.ndarray, x_obs: np.ndarray, y_bio: np.ndarray, mask_labeled: np.ndarray) -> "KineticXTFCSolver":
        rng = np.random.default_rng(self.seed)
        self._t0 = float(times_min[0])
        self._t_scale = max(float(times_min[-1] - self._t0), 1.0)
        t_norm = (times_min - self._t0) / self._t_scale
        in_dim = 1 + x_obs.shape[1]
        self._w = rng.normal(0.0, 0.8, size=(in_dim, self.hidden_dim))
        self._b = rng.uniform(-1.0, 1.0, size=self.hidden_dim)

        h_xtfc = self._xtfc_basis(t_norm, x_obs)
        h_lab = h_xtfc[mask_labeled]
        y_om_lab = y_bio[mask_labeled][:, 1:3]
        self._c_total = float(np.sum(y_bio[0])) if mask_labeled[0] else self.params.c_total_um

        dh_dt = np.gradient(h_xtfc, times_min, axis=0)
        # Initial fit for Picard state-evaluated ODE linearization
        p0 = h_lab.T @ h_lab + self.reg_lambda * np.eye(self.hidden_dim)
        w0 = np.linalg.solve(p0, h_lab.T @ y_om_lab)
        om_est = np.maximum(0.0, h_xtfc @ w0)
        q_o_est = np.minimum(om_est[:, 0], self._c_total)
        mass_est = np.minimum(om_est[:, 1], np.maximum(0.0, self._c_total - q_o_est))
        m_est = np.maximum(0.0, self._c_total - q_o_est - mass_est)
        o_est = q_o_est / max(self.params.q_nom, 1)

        jn = self.params.k_n_eff * (m_est**2)
        j2 = self.params.k_2_eff * mass_est * (m_est**2)
        do_rhs = self.params.q_nom * (jn + j2 - (self.params.k_c_eff + self.params.k_d_eff) * o_est)
        # Exact trapezoid cumulative integral for P(t) on irregular grids
        dt_arr = np.diff(times_min)
        o_rate = self.params.k_c_eff * o_est
        p_est = np.concatenate([[0.0], np.cumsum(0.5 * (o_rate[:-1] + o_rate[1:]) * dt_arr)])
        dmass_rhs = 2.0 * self.params.k_plus_eff * m_est * p_est + self.params.q_nom * self.params.k_c_eff * o_est
        f_om = np.column_stack([do_rhs, dmass_rhs])

        precision = h_lab.T @ h_lab + self.lambda_dyn * (dh_dt.T @ dh_dt) + self.reg_lambda * np.eye(self.hidden_dim)
        rhs = h_lab.T @ y_om_lab + self.lambda_dyn * (dh_dt.T @ f_om)
        self._w_out = np.linalg.solve(precision, rhs)
        res = y_om_lab - h_lab @ self._w_out
        sigma2 = max(0.25, float(np.mean(res**2)))
        self._cov_post = sigma2 * np.linalg.inv(precision)
        return self

    def predict_trajectory(self, times_min: np.ndarray, x_obs: np.ndarray, n_mc: int = 64) -> KineticPrediction:
        assert self._w_out is not None and self._cov_post is not None
        t_norm = (times_min - self._t0) / self._t_scale
        h_xtfc = self._xtfc_basis(t_norm, x_obs)
        om = np.maximum(0.0, h_xtfc @ self._w_out)
        q_o = np.minimum(om[:, 0], self._c_total)
        mass_um = np.minimum(om[:, 1], np.maximum(0.0, self._c_total - q_o))
        m_um = np.maximum(0.0, self._c_total - q_o - mass_um)
        o_um = q_o / max(self.params.q_nom, 1)
        q1, q2 = _targets_from_o(times_min, o_um, self._c_total, self.params.q_nom)

        rng = np.random.default_rng(self.seed + 88)
        chol = np.linalg.cholesky(self._cov_post + 1e-8 * np.eye(self.hidden_dim))
        q1s, q2s = [], []
        for _ in range(n_mc):
            w_s = self._w_out + chol @ rng.normal(size=(self.hidden_dim, 2))
            om_s = np.maximum(0.0, h_xtfc @ w_s)
            qo_s = np.minimum(om_s[:, 0], self._c_total)
            q1_i, q2_i = _targets_from_o(times_min, qo_s / max(self.params.q_nom, 1), self._c_total, self.params.q_nom)
            q1s.append(q1_i)
            q2s.append(q2_i)

        return KineticPrediction(
            m_um=m_um,
            o_um=o_um,
            mass_um=mass_um,
            q1_mean=q1,
            q1_std=float(np.std(q1s)),
            q2_mean=q2,
            q2_std=float(np.std(q2s)),
            selected_lambda_dyn=self.lambda_dyn,
            z1_pred=None,
            uq_calibrated=False,
        )


class HierarchicalKineticPIELMSolver:
    """Hierarchical Kinetic PIELM with state-evaluated R_dyn priors, grouped hold-out CV, and propagated UQ."""

    def __init__(
        self,
        hidden_dim: int = 48,
        r_dyn_type: str = "moment_ode",  # {"moment_ode", "finke_watzky", "avrami", "monotone_o", "smooth"}
        lambda_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0),
        reg_lambda: float = 1e-2,
        params: NominalKineticParams | None = None,
        seed: int = 0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.r_dyn_type = r_dyn_type
        self.lambda_grid = lambda_grid
        self.reg_lambda = reg_lambda
        self.params = params or NominalKineticParams()
        self.seed = seed
        self._w: np.ndarray | None = None
        self._b: np.ndarray | None = None
        self._w_out: np.ndarray | None = None
        self._cov_post: np.ndarray | None = None
        self._t0: float = 0.0
        self._t_scale: float = 1.0
        self._c_total: float = self.params.c_total_um
        self.selected_lambda_dyn: float = 0.0

    def _state_evaluated_r_dyn(
        self, times_min: np.ndarray, h_all: np.ndarray, w_curr: np.ndarray, c_total_um: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate per-channel R_dyn residuals [(R_mat_ch, R_rhs_ch) for ch in (m, qO, M)]."""
        dh_dt = np.gradient(h_all, times_min, axis=0)
        y_hat = np.maximum(0.0, h_all @ w_curr)
        m_hat = y_hat[:, 0]
        qo_hat = y_hat[:, 1]
        o_hat = qo_hat / max(self.params.q_nom, 1)
        mass_hat = y_hat[:, 2]
        zeros_r = (np.zeros((1, self.hidden_dim)), np.zeros(1, dtype=float))

        if self.r_dyn_type == "moment_ode":
            jn = self.params.k_n_eff * (m_hat**2)
            j2 = self.params.k_2_eff * mass_hat * (m_hat**2)
            dt_arr = np.diff(times_min)
            o_rate = self.params.k_c_eff * o_hat
            p_hat = np.concatenate([[0.0], np.cumsum(0.5 * (o_rate[:-1] + o_rate[1:]) * dt_arr)])
            dm = -self.params.q_nom * (jn + j2) + self.params.q_nom * self.params.k_d_eff * o_hat - 2.0 * self.params.k_plus_eff * m_hat * p_hat
            dqo = self.params.q_nom * (jn + j2 - (self.params.k_c_eff + self.params.k_d_eff) * o_hat)
            dmass = 2.0 * self.params.k_plus_eff * m_hat * p_hat + self.params.q_nom * self.params.k_c_eff * o_hat
            return [(dh_dt, dm), (dh_dt, dqo), (dh_dt, dmass)]
        if self.r_dyn_type == "finke_watzky":
            k1, k2 = 1e-4, 5e-4
            dmass = (k1 + k2 * mass_hat) * np.maximum(0.0, c_total_um - mass_hat)
            return [(dh_dt, -dmass), zeros_r, (dh_dt, dmass)]
        if self.r_dyn_type == "avrami":
            k_av, n_av = 0.005, 3.0
            t_rel = np.maximum(1e-3, times_min - times_min[0])
            dmass = n_av * (k_av**n_av) * (t_rel ** (n_av - 1.0)) * np.maximum(0.0, c_total_um - mass_hat)
            return [(dh_dt, -dmass), zeros_r, (dh_dt, dmass)]
        if self.r_dyn_type == "monotone_o":
            dqo_dt = dh_dt @ w_curr[:, 1]
            neg_mask = dqo_dt < 0.0
            r_o = (dh_dt[neg_mask], np.zeros(int(np.sum(neg_mask)), dtype=float)) if np.any(neg_mask) else zeros_r
            return [zeros_r, r_o, zeros_r]
        if self.r_dyn_type == "smooth":
            d2h_dt2 = np.gradient(dh_dt, times_min, axis=0)
            z = np.zeros(len(times_min), dtype=float)
            return [(d2h_dt2, z), (d2h_dt2, z), (d2h_dt2, z)]
        raise ValueError(f"Unknown r_dyn_type: {self.r_dyn_type}")

    def _solve_per_channel(
        self, h_lab: np.ndarray, y_lab: np.ndarray, r_channels: list[tuple[np.ndarray, np.ndarray]], lam_dyn: float
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        w_out = np.empty((self.hidden_dim, 3), dtype=float)
        precs = []
        base_gram = h_lab.T @ h_lab + self.reg_lambda * np.eye(self.hidden_dim)
        for ch in range(3):
            r_mat, r_rhs = r_channels[ch]
            prec = base_gram + lam_dyn * (r_mat.T @ r_mat)
            rhs = h_lab.T @ y_lab[:, ch] + lam_dyn * (r_mat.T @ r_rhs)
            w_out[:, ch] = np.linalg.solve(prec, rhs)
            precs.append(prec)
        return w_out, precs

    def _fit_with_lambda(
        self,
        times_min: np.ndarray,
        h_all: np.ndarray,
        y_bio: np.ndarray,
        mask_labeled: np.ndarray,
        lam_dyn: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        h_lab = h_all[mask_labeled]
        y_lab = y_bio[mask_labeled]
        p0 = h_lab.T @ h_lab + self.reg_lambda * np.eye(self.hidden_dim)
        w_curr = np.linalg.solve(p0, h_lab.T @ y_lab)

        precs = [p0, p0, p0]
        for _ in range(3):
            r_ch = self._state_evaluated_r_dyn(times_min, h_all, w_curr, self._c_total)
            w_curr, precs = self._solve_per_channel(h_lab, y_lab, r_ch, lam_dyn)

        res = y_lab - h_lab @ w_curr
        sigma2 = max(0.25, float(np.mean(res**2)))
        cov_post = sigma2 * np.linalg.inv(precs[1])
        return w_curr, cov_post, sigma2

    def fit_grouped(
        self,
        times_list: list[np.ndarray],
        x_list: list[np.ndarray],
        y_list: list[np.ndarray],
        mask_list: list[np.ndarray],
    ) -> "HierarchicalKineticPIELMSolver":
        """True leave-one-batch-out grouped CV over multiple batches for lambda_dyn selection."""
        rng = np.random.default_rng(self.seed)
        self._t0 = float(times_list[0][0])
        self._t_scale = max(float(times_list[0][-1] - self._t0), 1.0)
        in_dim = 1 + x_list[0].shape[1]
        self._w = rng.normal(0.0, 0.8, size=(in_dim, self.hidden_dim))
        self._b = rng.uniform(-1.0, 1.0, size=self.hidden_dim)
        c_list = [float(np.sum(y[0])) for y in y_list]
        self._c_total = float(np.mean(c_list))

        h_list = [_rff_features((t - self._t0) / self._t_scale, x, self._w, self._b) for t, x in zip(times_list, x_list)]
        n_groups = len(times_list)

        best_lam = self.lambda_grid[0]
        best_cv = float("inf")
        for lam in self.lambda_grid:
            fold_errs = []
            for hold in range(n_groups):
                tr_h = np.concatenate([h_list[g][mask_list[g]] for g in range(n_groups) if g != hold], axis=0)
                tr_y = np.concatenate([y_list[g][mask_list[g]] for g in range(n_groups) if g != hold], axis=0)
                p0 = tr_h.T @ tr_h + self.reg_lambda * np.eye(self.hidden_dim)
                w_g = np.linalg.solve(p0, tr_h.T @ tr_y)
                for _ in range(3):
                    r_per_g = [self._state_evaluated_r_dyn(times_list[g], h_list[g], w_g, c_list[g]) for g in range(n_groups) if g != hold]
                    r_ch = [
                        (np.concatenate([rg[ch][0] for rg in r_per_g], axis=0), np.concatenate([rg[ch][1] for rg in r_per_g], axis=0))
                        for ch in range(3)
                    ]
                    w_g, _ = self._solve_per_channel(tr_h, tr_y, r_ch, lam)
                val_pred = np.maximum(0.0, h_list[hold][mask_list[hold]] @ w_g)
                fold_errs.append(float(np.mean((val_pred - y_list[hold][mask_list[hold]]) ** 2)))
            mean_cv = float(np.mean(fold_errs))
            if mean_cv < best_cv:
                best_cv = mean_cv
                best_lam = lam

        self.selected_lambda_dyn = float(best_lam)
        all_h = np.concatenate([h_list[g][mask_list[g]] for g in range(n_groups)], axis=0)
        all_y = np.concatenate([y_list[g][mask_list[g]] for g in range(n_groups)], axis=0)
        p0 = all_h.T @ all_h + self.reg_lambda * np.eye(self.hidden_dim)
        w_curr = np.linalg.solve(p0, all_h.T @ all_y)
        precs = [p0, p0, p0]
        for _ in range(3):
            r_per_g = [self._state_evaluated_r_dyn(times_list[g], h_list[g], w_curr, c_list[g]) for g in range(n_groups)]
            r_ch = [
                (np.concatenate([rg[ch][0] for rg in r_per_g], axis=0), np.concatenate([rg[ch][1] for rg in r_per_g], axis=0))
                for ch in range(3)
            ]
            w_curr, precs = self._solve_per_channel(all_h, all_y, r_ch, self.selected_lambda_dyn)
        self._w_out = w_curr
        sigma2 = max(0.25, float(np.mean((all_y - all_h @ w_curr) ** 2)))
        self._cov_post = sigma2 * np.linalg.inv(precs[1])
        return self

    def fit(self, times_min: np.ndarray, x_obs: np.ndarray, y_bio: np.ndarray, mask_labeled: np.ndarray) -> "HierarchicalKineticPIELMSolver":
        """Single-batch fit using held-out labeled fold when >= 4 labels exist, else endpoint-only."""
        rng = np.random.default_rng(self.seed)
        self._t0 = float(times_min[0])
        self._t_scale = max(float(times_min[-1] - self._t0), 1.0)
        t_norm = (times_min - self._t0) / self._t_scale
        in_dim = 1 + x_obs.shape[1]
        self._w = rng.normal(0.0, 0.8, size=(in_dim, self.hidden_dim))
        self._b = rng.uniform(-1.0, 1.0, size=self.hidden_dim)

        h_all = _rff_features(t_norm, x_obs, self._w, self._b)
        self._c_total = float(np.sum(y_bio[0])) if mask_labeled[0] else self.params.c_total_um
        lab_idx = np.where(mask_labeled)[0]

        best_lam = self.lambda_grid[0]
        if len(lab_idx) >= 3:
            # Pure held-out label CV (no nominal-trajectory leakage into CV score)
            best_score = float("inf")
            for lam in self.lambda_grid:
                errs = []
                for hold_i in lab_idx:
                    m_sub = mask_labeled.copy()
                    m_sub[hold_i] = False
                    w_sub, _, _ = self._fit_with_lambda(times_min, h_all, y_bio, m_sub, lam)
                    pred_i = np.maximum(0.0, h_all[hold_i] @ w_sub)
                    errs.append(float(np.mean((pred_i - y_bio[hold_i]) ** 2)))
                score = float(np.mean(errs))
                if score < best_score:
                    best_score = score
                    best_lam = lam
        else:
            # Under strict 2-point endpoint-only R0, select smallest non-zero physics weight that preserves mass
            best_score = float("inf")
            for lam in self.lambda_grid:
                w_tmp, _, _ = self._fit_with_lambda(times_min, h_all, y_bio, mask_labeled, lam)
                pred_all = np.maximum(0.0, h_all @ w_tmp)
                mass_err = float(np.mean((np.sum(pred_all, axis=1) - self._c_total) ** 2))
                if mass_err < best_score:
                    best_score = mass_err
                    best_lam = lam

        self.selected_lambda_dyn = float(best_lam)
        self._w_out, self._cov_post, _ = self._fit_with_lambda(times_min, h_all, y_bio, mask_labeled, self.selected_lambda_dyn)
        return self

    def predict_trajectory(
        self, times_min: np.ndarray, x_obs: np.ndarray, c_total_um: float | None = None, n_mc: int = 64
    ) -> KineticPrediction:
        assert self._w is not None and self._b is not None and self._w_out is not None and self._cov_post is not None
        c_tot = self._c_total if c_total_um is None else float(c_total_um)
        t_norm = (times_min - self._t0) / self._t_scale
        h = _rff_features(t_norm, x_obs, self._w, self._b)
        pred = np.maximum(0.0, h @ self._w_out)
        m_um = pred[:, 0]
        o_um = pred[:, 1] / max(self.params.q_nom, 1)
        mass_um = pred[:, 2]
        q1, q2 = _targets_from_o(times_min, o_um, c_tot, self.params.q_nom)

        rng = np.random.default_rng(self.seed + 99)
        chol = np.linalg.cholesky(self._cov_post + 1e-8 * np.eye(self.hidden_dim))
        q1s, q2s = [], []
        for _ in range(n_mc):
            w_s = self._w_out + chol @ rng.normal(size=(self.hidden_dim, 3))
            p_s = np.maximum(0.0, h @ w_s)
            q1_i, q2_i = _targets_from_o(times_min, p_s[:, 1] / max(self.params.q_nom, 1), c_tot, self.params.q_nom)
            q1s.append(q1_i)
            q2s.append(q2_i)

        return KineticPrediction(
            m_um=m_um,
            o_um=o_um,
            mass_um=mass_um,
            q1_mean=q1,
            q1_std=float(np.std(q1s)),
            q2_mean=q2,
            q2_std=float(np.std(q2s)),
            selected_lambda_dyn=self.selected_lambda_dyn,
            z1_pred=None,
            uq_calibrated=False,
        )
