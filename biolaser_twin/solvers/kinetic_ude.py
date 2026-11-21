"""Stoichiometric Universal Differential Equation (UDE) with MLP or PI-KAN (RBF-KAN) closure."""

import numpy as np
import torch
import torch.nn as nn

from .kinetic_pielm import KineticPrediction, _targets_from_o


class _PIKANLayer(nn.Module):
    """Gaussian-RBF Kolmogorov-Arnold layer with inputs in [0, 1] matching knot domain [0, 1]."""

    def __init__(self, in_dim: int, out_dim: int, num_knots: int = 8) -> None:
        super().__init__()
        self.register_buffer("knots", torch.linspace(0.0, 1.0, num_knots))
        self.spline_w = nn.Parameter(torch.randn(in_dim, num_knots, out_dim) * 0.05)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, 0.0, 1.0)
        diff = x_clamped.unsqueeze(-1) - self.knots.view(1, 1, -1)
        rbf = torch.exp(-12.0 * (diff**2))
        kan_term = torch.einsum("bik,iko->bo", rbf, self.spline_w)
        return self.linear(x_clamped) + kan_term


class _StoichiometricClosure(nn.Module):
    """Predicts [r_nuc, r_conv, r_elong] fluxes with m*P elongation & exact m + q*O + M == C_total."""

    def __init__(self, closure_type: str = "mlp", hidden_dim: int = 24) -> None:
        super().__init__()
        if closure_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 3),
            )
        elif closure_type == "pikan":
            self.net = nn.Sequential(
                _PIKANLayer(4, hidden_dim, num_knots=8),
                nn.Sigmoid(),  # keeps hidden activations in [0, 1] for next PI-KAN layer
                _PIKANLayer(hidden_dim, 3, num_knots=8),
            )
        else:
            raise ValueError(f"Unknown closure_type: {closure_type}")
        self.softplus = nn.Softplus()

    def forward(self, state_norm: torch.Tensor, rate_scale: torch.Tensor) -> torch.Tensor:
        # state_norm = [m/C, q*O/C, P/C, M/C]
        raw = self.softplus(self.net(state_norm)) * 0.02 * rate_scale
        m_f = torch.clamp(state_norm[..., 0:1], min=0.0)
        o_f = torch.clamp(state_norm[..., 1:2], min=0.0)
        p_f = torch.clamp(state_norm[..., 2:3], min=0.0)
        r_nuc = raw[..., 0:1] * (m_f**2)
        r_conv = raw[..., 1:2] * o_f
        r_elong = raw[..., 2:3] * m_f * p_f  # true m*P elongation preserves lag phase
        return torch.cat([r_nuc, r_conv, r_elong], dim=-1)


class StoichiometricUDESolver:
    """Stoichiometric UDE (MLP or RBF-PIKAN) with RK4 integrator, x_obs conditioning, and MC UQ."""

    def __init__(
        self,
        closure_type: str = "mlp",
        q_nom: int = 4,
        n_epochs: int = 60,
        lr: float = 2e-2,
        dt_max_min: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.closure_type = closure_type
        self.q_nom = q_nom
        self.n_epochs = n_epochs
        self.lr = lr
        self.dt_max_min = dt_max_min
        self.seed = seed
        self._closure: _StoichiometricClosure | None = None
        self._obs_head: nn.Linear | None = None
        self._cond_net: nn.Linear | None = None
        self._c_total: float = 50.0
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def _rhs(self, s: torch.Tensor, rate_scale: torch.Tensor) -> torch.Tensor:
        assert self._closure is not None
        flux = self._closure(s, rate_scale)
        r_nuc, r_conv, r_elong = flux[:, 0:1], flux[:, 1:2], flux[:, 2:3]
        ds_m = -r_nuc - r_elong
        ds_o = r_nuc - r_conv
        ds_p = r_conv / float(max(self.q_nom, 1))
        ds_mass = r_conv + r_elong
        return torch.cat([ds_m, ds_o, ds_p, ds_mass], dim=-1)

    def _integrate_rk4(self, times_min: torch.Tensor, rate_scale: torch.Tensor, c_total: float) -> torch.Tensor:
        s = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        states = [s]
        for i in range(len(times_min) - 1):
            dt = float(times_min[i + 1].item() - times_min[i].item())
            if dt < 0.0:
                raise ValueError("times_min must be non-decreasing.")
            n_sub = max(1, int(np.ceil(dt / self.dt_max_min)))
            h = dt / n_sub
            for _ in range(n_sub):
                k1 = self._rhs(s, rate_scale)
                k2 = self._rhs(torch.clamp(s + 0.5 * h * k1, min=0.0), rate_scale)
                k3 = self._rhs(torch.clamp(s + 0.5 * h * k2, min=0.0), rate_scale)
                k4 = self._rhs(torch.clamp(s + h * k3, min=0.0), rate_scale)
                s = torch.clamp(s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), min=0.0)
                # Project conserved mass channels [m, q*O, M] onto simplex sum == 1
                mass_sum = torch.clamp(s[:, 0:1] + s[:, 1:2] + s[:, 3:4], min=1e-8)
                s = torch.cat([s[:, 0:1] / mass_sum, s[:, 1:2] / mass_sum, s[:, 2:3], s[:, 3:4] / mass_sum], dim=-1)
            states.append(s)
        out = torch.cat(states, dim=0) * c_total
        return torch.stack([out[:, 0], out[:, 1], out[:, 3]], dim=-1)

    def fit(
        self,
        times_min: np.ndarray,
        x_obs: np.ndarray,
        y_bio: np.ndarray | None = None,
        mask_labeled: np.ndarray | None = None,
    ) -> "StoichiometricUDESolver":
        torch.manual_seed(self.seed)
        self._closure = _StoichiometricClosure(self.closure_type)
        self._obs_head = nn.Linear(3, x_obs.shape[1])
        self._cond_net = nn.Linear(x_obs.shape[1] * 2, 3)
        self._c_total = float(np.sum(y_bio[0])) if (y_bio is not None and len(y_bio) > 0) else 50.0
        self._x_mean = np.mean(x_obs, axis=0)
        self._x_std = np.maximum(np.std(x_obs, axis=0), 1e-3)

        t_t = torch.tensor(times_min, dtype=torch.float32)
        x_norm = torch.tensor((x_obs - self._x_mean) / self._x_std, dtype=torch.float32)
        x_summary = torch.cat([x_norm[0:1], x_norm[-1:]], dim=-1)

        params = list(self._closure.parameters()) + list(self._obs_head.parameters()) + list(self._cond_net.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.n_epochs):
            opt.zero_grad()
            rate_scale = torch.exp(torch.clamp(self._cond_net(x_summary), -2.0, 2.0))
            traj = self._integrate_rk4(t_t, rate_scale, self._c_total)
            pred_x_norm = self._obs_head(traj / self._c_total)
            loss = torch.mean((pred_x_norm - x_norm) ** 2)
            if y_bio is not None and mask_labeled is not None and np.any(mask_labeled):
                y_t = torch.tensor(y_bio[mask_labeled], dtype=torch.float32)
                loss = loss + 2.0 * torch.mean(((traj[mask_labeled] - y_t) / self._c_total) ** 2)
            loss.backward()
            opt.step()
        return self

    def predict_trajectory(self, times_min: np.ndarray, x_obs: np.ndarray, n_mc: int = 16) -> KineticPrediction:
        assert self._closure is not None and self._cond_net is not None and self._x_mean is not None and self._x_std is not None
        x_norm = torch.tensor((x_obs - self._x_mean) / self._x_std, dtype=torch.float32)
        x_summary = torch.cat([x_norm[0:1], x_norm[-1:]], dim=-1)
        t_t = torch.tensor(times_min, dtype=torch.float32)
        with torch.no_grad():
            rate_scale = torch.exp(torch.clamp(self._cond_net(x_summary), -2.0, 2.0))
            traj = self._integrate_rk4(t_t, rate_scale, self._c_total).numpy()

            rng = np.random.default_rng(self.seed + 55)
            q1s, q2s = [], []
            for _ in range(n_mc):
                jitter = torch.tensor(rng.normal(1.0, 0.08, size=(1, 3)), dtype=torch.float32)
                tr_s = self._integrate_rk4(t_t, rate_scale * torch.clamp(jitter, min=0.5), self._c_total).numpy()
                q1_i, q2_i = _targets_from_o(times_min, tr_s[:, 1] / max(self.q_nom, 1), self._c_total, self.q_nom)
                q1s.append(q1_i)
                q2s.append(q2_i)

        m_um = traj[:, 0]
        o_um = traj[:, 1] / max(self.q_nom, 1)
        mass_um = traj[:, 2]
        q1, q2 = _targets_from_o(times_min, o_um, self._c_total, self.q_nom)
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
