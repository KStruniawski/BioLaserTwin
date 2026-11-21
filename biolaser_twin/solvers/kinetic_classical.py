"""Classical biophysical baselines: multi-m0 AmyloFit global NLS and sequential Moment UKF."""

from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import least_squares

from ..identifiability.observability import (
    KineticObservableConfig,
    integrate_nominal_5moment_ode,
    nominal_kinetic_forward_observables,
    observation_noise_std,
)


@dataclass
class ClassicalKineticFit:
    """Output of global NLS or sequential UKF kinetic inversion with propagated UQ."""

    theta_log10: np.ndarray
    theta_std: np.ndarray
    m_um: np.ndarray
    o_um: np.ndarray
    mass_um: np.ndarray
    q1_mean: float
    q1_std: float
    q2_mean: float
    q2_std: float
    ess: float | None = None


def _q1_q2_from_mom(times_min: np.ndarray, o_um: np.ndarray, c_total_um: float, q_nom: int = 4) -> tuple[float, float]:
    t = np.asarray(times_min, dtype=float)
    t_span = max(float(t[-1] - t[0]), 1e-9)
    o_pos = np.maximum(0.0, o_um)
    area = float(np.trapezoid(q_nom * o_pos, t) if hasattr(np, "trapezoid") else np.trapz(q_nom * o_pos, t))
    q1 = float(np.clip(area / (t_span * max(c_total_um, 1e-9)), 0.0, 1.0))
    q2 = float((t[int(np.argmax(o_pos))] - t[0]) / t_span)
    return q1, q2


class AmyloFitGlobalNLSSolver:
    """AmyloFit-style global nonlinear least squares supporting multi-m0 batches and Laplace UQ."""

    def __init__(self, cfg: KineticObservableConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or KineticObservableConfig(observable_set="full_lasing")
        self.seed = seed

    def fit_predict_multi(
        self,
        times_list: list[np.ndarray],
        y_obs_list: list[np.ndarray],
        m0_list: list[float],
        theta_init: np.ndarray | None = None,
        n_laplace: int = 24,
        noise_std_list: list[np.ndarray] | None = None,
    ) -> list[ClassicalKineticFit]:
        """Global shared-rate fit across multiple initial monomer concentrations m0."""
        d = 4 if self.cfg.s_sigma_known else 5
        x0 = np.array([-8.5, -1.3, -7.0, -0.5] + ([] if d == 4 else [0.0]), dtype=float) if theta_init is None else theta_init
        b_lo = np.array([-11.0, -3.5, -9.0, -4.0] + ([] if d == 4 else [-2.0]), dtype=float)
        b_hi = np.array([-4.5, 0.5, -4.0, 1.0] + ([] if d == 4 else [2.0]), dtype=float)

        scales_list = (
            [observation_noise_std(y_i, replace(self.cfg, c_total_um=float(m0_i))) for y_i, m0_i in zip(y_obs_list, m0_list)]
            if noise_std_list is None
            else [np.maximum(np.asarray(s, dtype=float), 1e-12) for s in noise_std_list]
        )

        def _res(th: np.ndarray) -> np.ndarray:
            chunks = []
            for t_i, y_i, m0_i, sc_i in zip(times_list, y_obs_list, m0_list, scales_list):
                cfg_i = replace(self.cfg, c_total_um=float(m0_i))
                pred_i = nominal_kinetic_forward_observables(th, t_i, cfg_i)
                chunks.append(((pred_i - y_i) / sc_i).reshape(-1))
            return np.concatenate(chunks)

        fit = least_squares(_res, np.clip(x0, b_lo, b_hi), bounds=(b_lo, b_hi), max_nfev=100)
        jtj = fit.jac.T @ fit.jac + 1e-6 * np.eye(d)
        cov = np.linalg.inv(jtj)
        std = np.sqrt(np.maximum(1e-10, np.diag(cov)))

        rng = np.random.default_rng(self.seed)
        th_samples = np.clip(rng.multivariate_normal(fit.x, cov, size=n_laplace), b_lo, b_hi)

        results: list[ClassicalKineticFit] = []
        for t_i, m0_i in zip(times_list, m0_list):
            mom = integrate_nominal_5moment_ode(
                t_i,
                k_n=10.0 ** float(fit.x[0]),
                k_plus=10.0 ** float(fit.x[1]),
                k_2=10.0 ** float(fit.x[2]),
                k_c=10.0 ** float(fit.x[3]),
                k_d=self.cfg.k_d_nom,
                q_nom=self.cfg.q_nom,
                c_total_um=float(m0_i),
                t0_min=self.cfg.t0_min,
            )
            q1, q2 = _q1_q2_from_mom(t_i, mom["O"], float(m0_i), self.cfg.q_nom)
            q1s, q2s = [], []
            for th_s in th_samples:
                m_s = integrate_nominal_5moment_ode(
                    t_i,
                    k_n=10.0 ** float(th_s[0]),
                    k_plus=10.0 ** float(th_s[1]),
                    k_2=10.0 ** float(th_s[2]),
                    k_c=10.0 ** float(th_s[3]),
                    k_d=self.cfg.k_d_nom,
                    q_nom=self.cfg.q_nom,
                    c_total_um=float(m0_i),
                    t0_min=self.cfg.t0_min,
                )
                q1_i, q2_i = _q1_q2_from_mom(t_i, m_s["O"], float(m0_i), self.cfg.q_nom)
                q1s.append(q1_i)
                q2s.append(q2_i)

            results.append(
                ClassicalKineticFit(
                    theta_log10=fit.x,
                    theta_std=std,
                    m_um=mom["m"],
                    o_um=mom["O"],
                    mass_um=mom["M"],
                    q1_mean=q1,
                    q1_std=float(np.std(q1s)),
                    q2_mean=q2,
                    q2_std=float(np.std(q2s)),
                )
            )
        return results

    def fit_predict(
        self,
        times_min: np.ndarray,
        y_obs: np.ndarray,
        m0_um: float | None = None,
        theta_init: np.ndarray | None = None,
        noise_std: np.ndarray | None = None,
    ) -> ClassicalKineticFit:
        m0_val = self.cfg.c_total_um if m0_um is None else float(m0_um)
        return self.fit_predict_multi(
            [times_min],
            [y_obs],
            [m0_val],
            theta_init=theta_init,
            noise_std_list=None if noise_std is None else [noise_std],
        )[0]


class SequentialImportanceSmoother:
    """Sequential importance weighting smoother over (m, O, P, M) with ESS diagnostic."""

    def __init__(
        self,
        cfg: KineticObservableConfig | None = None,
        n_particles: int = 24,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or KineticObservableConfig(observable_set="full_lasing")
        self.n_particles = n_particles
        self.seed = seed

    def filter_predict(
        self,
        times_min: np.ndarray,
        y_obs: np.ndarray,
        theta_prior_mean: np.ndarray,
        theta_prior_std: np.ndarray,
        m0_um: float | None = None,
        noise_std: np.ndarray | None = None,
    ) -> ClassicalKineticFit:
        rng = np.random.default_rng(self.seed)
        m0_val = self.cfg.c_total_um if m0_um is None else float(m0_um)
        cfg_use = replace(self.cfg, c_total_um=m0_val)
        thetas = rng.normal(theta_prior_mean, theta_prior_std, size=(self.n_particles, len(theta_prior_mean)))

        scales_obs = (
            observation_noise_std(y_obs, cfg_use)
            if noise_std is None
            else np.maximum(np.asarray(noise_std, dtype=float), 1e-12)
        )

        def _eval_particles(th_arr: np.ndarray) -> tuple[list, list, np.ndarray]:
            p_list, t_list, ll_arr = [], [], np.zeros(len(th_arr), dtype=float)
            for idx, th in enumerate(th_arr):
                pred = nominal_kinetic_forward_observables(th, times_min, cfg_use)
                p_list.append(pred)
                t_list.append(
                    integrate_nominal_5moment_ode(
                        times_min,
                        k_n=10.0 ** float(th[0]),
                        k_plus=10.0 ** float(th[1]),
                        k_2=10.0 ** float(th[2]),
                        k_c=10.0 ** float(th[3]),
                        k_d=cfg_use.k_d_nom,
                        q_nom=cfg_use.q_nom,
                        c_total_um=m0_val,
                        t0_min=cfg_use.t0_min,
                    )
                )
                ll_arr[idx] = -0.5 * float(np.sum(((pred - y_obs) / scales_obs) ** 2))
            return p_list, t_list, ll_arr

        # Tempered SMC with systematic resampling & random-walk rejuvenation prevents weight collapse
        preds, trajs, ll_total = _eval_particles(thetas)
        betas = (0.001, 0.005, 0.02, 0.08, 0.25, 0.60, 1.0)
        prev_beta = 0.0
        weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        for beta in betas:
            db = beta - prev_beta
            log_w = np.log(np.maximum(weights, 1e-300)) + db * ll_total
            log_w -= np.max(log_w)
            weights = np.exp(log_w)
            weights /= np.sum(weights)
            ess = float(1.0 / np.sum(weights**2))
            if ess < 0.5 * self.n_particles and beta < 1.0:
                # Systematic resampling + posterior-invariant Metropolis-Hastings rejuvenation
                positions = (rng.random() + np.arange(self.n_particles)) / self.n_particles
                cumsum = np.cumsum(weights)
                cumsum[-1] = 1.0
                ancestors = np.searchsorted(cumsum, positions)
                thetas_cur = thetas[ancestors].copy()
                preds_cur = [preds[a] for a in ancestors]
                trajs_cur = [trajs[a] for a in ancestors]
                ll_cur = ll_total[ancestors].copy()

                # Deterministic prior/temperature scale (independent of current particle population -> strictly symmetric MH)
                th_scale = 0.25 * np.asarray(theta_prior_std, dtype=float) / np.sqrt(max(beta, 1e-2))
                thetas_prop = thetas_cur + rng.normal(0.0, th_scale, size=thetas_cur.shape)
                preds_prop, trajs_prop, ll_prop = _eval_particles(thetas_prop)

                # Exact symmetric-proposal MH acceptance ratio under tempered posterior beta * ll + log_prior
                prior_cur = -0.5 * np.sum(((thetas_cur - theta_prior_mean) / np.maximum(theta_prior_std, 1e-6)) ** 2, axis=1)
                prior_prop = -0.5 * np.sum(((thetas_prop - theta_prior_mean) / np.maximum(theta_prior_std, 1e-6)) ** 2, axis=1)
                log_acc = beta * (ll_prop - ll_cur) + (prior_prop - prior_cur)
                accept = np.log(rng.random(self.n_particles)) < log_acc
                for i in range(self.n_particles):
                    if accept[i]:
                        thetas_cur[i] = thetas_prop[i]
                        preds_cur[i] = preds_prop[i]
                        trajs_cur[i] = trajs_prop[i]
                        ll_cur[i] = ll_prop[i]
                thetas, preds, trajs, ll_total = thetas_cur, preds_cur, trajs_cur, ll_cur
                weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
            prev_beta = beta

        ess = float(1.0 / np.sum(weights**2))

        th_mean = np.sum(weights[:, None] * thetas, axis=0)
        th_std = np.sqrt(np.sum(weights[:, None] * ((thetas - th_mean) ** 2), axis=0) + 1e-8)

        m_mean = sum(w * tr["m"] for w, tr in zip(weights, trajs))
        o_mean = sum(w * tr["O"] for w, tr in zip(weights, trajs))
        mass_mean = sum(w * tr["M"] for w, tr in zip(weights, trajs))

        q1s = np.array([_q1_q2_from_mom(times_min, tr["O"], m0_val, cfg_use.q_nom)[0] for tr in trajs])
        q2s = np.array([_q1_q2_from_mom(times_min, tr["O"], m0_val, cfg_use.q_nom)[1] for tr in trajs])
        q1_m = float(np.sum(weights * q1s))
        q2_m = float(np.sum(weights * q2s))
        q1_s = float(np.sqrt(np.sum(weights * ((q1s - q1_m) ** 2)) + 1e-8))
        q2_s = float(np.sqrt(np.sum(weights * ((q2s - q2_m) ** 2)) + 1e-8))

        return ClassicalKineticFit(
            theta_log10=th_mean,
            theta_std=th_std,
            m_um=m_mean,
            o_um=o_mean,
            mass_um=mass_mean,
            q1_mean=q1_m,
            q1_std=q1_s,
            q2_mean=q2_m,
            q2_std=q2_s,
            ess=ess,
        )


MomentUKFSolver = SequentialImportanceSmoother
