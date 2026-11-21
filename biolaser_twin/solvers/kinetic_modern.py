"""2024-2026 PINN & Liquid/SSM comparators (PINNMamba, PI-LNN/CfC, PirateNets, PINNsFormer)."""

import numpy as np
import torch
import torch.nn as nn

from ..kinetics_nominal import NominalKineticParams
from .kinetic_pielm import KineticPrediction, _targets_from_o


class _PINNMambaBackbone(nn.Module):
    """Selective input-dependent SSM (Mamba-style: Delta(u), B(u), C(u)) over temporal tokens."""

    def __init__(self, in_dim: int, d_model: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.a_log = nn.Parameter(torch.zeros(d_model))
        self.delta_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Softplus())
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 3)

    def forward(self, inp: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.proj(inp))
        h = torch.zeros(u.shape[1], dtype=inp.dtype, device=inp.device)
        ys = []
        a_base = -torch.exp(self.a_log)
        for i in range(u.shape[0]):
            delta_i = self.delta_proj(u[i]) * torch.clamp(dt[i], min=0.05, max=5.0)
            a_bar = torch.exp(a_base * delta_i)
            b_bar = delta_i * self.b_proj(u[i])
            h = a_bar * h + b_bar * u[i]
            y_i = (self.c_proj(u[i]) * h) * torch.sigmoid(self.gate(u[i]))
            ys.append(y_i)
        return torch.softmax(self.out(torch.stack(ys, dim=0)), dim=-1)


class _PILNNCfCBackbone(nn.Module):
    """Physics-Informed Liquid Neural Network (CfC) with explicit time-constant tau(x, h)."""

    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim), nn.Tanh())
        self.ff2 = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim), nn.Tanh())
        self.time_a = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.time_b = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)
        self.hidden_dim = hidden_dim

    def forward(self, inp: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(self.hidden_dim, dtype=inp.dtype, device=inp.device)
        hs = []
        for i in range(inp.shape[0]):
            cat = torch.cat([inp[i], h], dim=-1)
            t_interp = torch.sigmoid(self.time_a(cat) * torch.clamp(dt[i], min=0.01) + self.time_b(cat))
            h = self.ff1(cat) * (1.0 - t_interp) + self.ff2(cat) * t_interp
            hs.append(h)
        return torch.softmax(self.head(torch.stack(hs, dim=0)), dim=-1)


class _PirateNetBackbone(nn.Module):
    """PirateNet (Wang et al., JMLR 2024): Fourier-embedded gated blocks with alpha initialized to 0."""

    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, hidden_dim // 2) * 2.0)
        self.u_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.0))  # zero-init per JMLR 2024 paper
        self.alpha2 = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, inp: torch.Tensor, _dt: torch.Tensor) -> torch.Tensor:
        proj = inp @ self.B
        phi = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        u = torch.tanh(self.u_proj(phi))
        v = torch.tanh(self.v_proj(phi))
        h = phi
        for lin, alp in ((self.l1, self.alpha1), (self.l2, self.alpha2)):
            z = torch.tanh(lin(h))
            g = z * u + (1.0 - z) * v
            h = alp * torch.tanh(self.l3(g)) + (1.0 - alp) * h
        return torch.softmax(self.head(h), dim=-1)


class _WaveletAct(nn.Module):
    """PINNsFormer Wavelet activation: w1*sin(x) + w2*cos(x)."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class _PINNsFormerBackbone(nn.Module):
    """PINNsFormer (ICLR 2024): pseudo-sequence windowing + Wavelet-activated Transformer."""

    def __init__(self, in_dim: int, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.wavelet = _WaveletAct()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=64, dropout=0.1, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 3)

    def forward(self, inp: torch.Tensor, _dt: torch.Tensor) -> torch.Tensor:
        emb = self.wavelet(self.embed(inp))
        out = self.encoder(emb.unsqueeze(0)).squeeze(0)
        return torch.softmax(self.head(out), dim=-1)


class _BaseModernPINNSolver:
    """Shared physics-informed training harness with kinetic ODE residual + MC UQ."""

    def __init__(
        self,
        arch: str,
        q_nom: int = 4,
        n_epochs: int = 60,
        lr: float = 1e-2,
        lambda_dyn: float = 0.05,
        params: NominalKineticParams | None = None,
        seed: int = 0,
    ) -> None:
        self.arch = arch
        self.q_nom = q_nom
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_dyn = lambda_dyn
        self.params = params or NominalKineticParams()
        self.seed = seed
        self._net: nn.Module | None = None
        self._c_total: float = self.params.c_total_um
        self._t0: float = 0.0
        self._t_scale: float = 1.0
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def _build_net(self, in_dim: int) -> nn.Module:
        if self.arch == "pinnmamba":
            return _PINNMambaBackbone(in_dim)
        if self.arch == "pilnn_cfc":
            return _PILNNCfCBackbone(in_dim)
        if self.arch == "piratenet":
            return _PirateNetBackbone(in_dim)
        if self.arch == "pinnsformer":
            return _PINNsFormerBackbone(in_dim)
        raise ValueError(f"Unknown architecture: {self.arch}")

    def fit(
        self,
        times_min: np.ndarray,
        x_obs: np.ndarray,
        y_bio: np.ndarray,
        mask_labeled: np.ndarray,
    ) -> "_BaseModernPINNSolver":
        torch.manual_seed(self.seed)
        self._t0 = float(times_min[0])
        self._t_scale = max(float(times_min[-1] - self._t0), 1.0)
        self._c_total = float(np.sum(y_bio[0])) if len(y_bio) > 0 else self.params.c_total_um
        self._x_mean = np.mean(x_obs, axis=0)
        self._x_std = np.maximum(np.std(x_obs, axis=0), 1e-3)

        t_norm = (times_min - self._t0) / self._t_scale
        x_norm = (x_obs - self._x_mean) / self._x_std
        inp = torch.tensor(np.column_stack([t_norm, x_norm]), dtype=torch.float32)
        dt_raw = np.concatenate([[0.0], np.diff(times_min)])
        dt = torch.tensor(dt_raw / 30.0, dtype=torch.float32)
        y_norm = torch.tensor(y_bio / self._c_total, dtype=torch.float32)
        dt_step = torch.clamp(torch.tensor(np.diff(times_min), dtype=torch.float32), min=1e-3)

        self._net = self._build_net(inp.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = self._net(inp, dt)
            loss_data = torch.mean((pred[mask_labeled] - y_norm[mask_labeled]) ** 2)

            # Kinetic ODE collocation residual on predicted [m, q*O, M] states
            m_hat = pred[:, 0] * self._c_total
            qo_hat = pred[:, 1] * self._c_total
            o_hat = qo_hat / float(max(self.q_nom, 1))
            mass_hat = pred[:, 2] * self._c_total
            jn = self.params.k_n_eff * (m_hat**2)
            j2 = self.params.k_2_eff * mass_hat * (m_hat**2)
            do_rhs = self.q_nom * (jn + j2 - (self.params.k_c_eff + self.params.k_d_eff) * o_hat)
            do_fd = (qo_hat[1:] - qo_hat[:-1]) / dt_step
            ode_res = (do_fd - 0.5 * (do_rhs[1:] + do_rhs[:-1])) / self._c_total
            loss_phys = torch.mean(ode_res**2)
            loss = loss_data + self.lambda_dyn * loss_phys
            loss.backward()
            opt.step()
        return self

    def predict_trajectory(self, times_min: np.ndarray, x_obs: np.ndarray, n_mc: int = 16) -> KineticPrediction:
        assert self._net is not None and self._x_mean is not None and self._x_std is not None
        t_norm = (times_min - self._t0) / self._t_scale
        x_norm = (x_obs - self._x_mean) / self._x_std
        inp = torch.tensor(np.column_stack([t_norm, x_norm]), dtype=torch.float32)
        dt = torch.tensor(np.concatenate([[0.0], np.diff(times_min)]) / 30.0, dtype=torch.float32)

        self._net.eval()
        with torch.no_grad():
            pred = self._net(inp, dt).numpy() * self._c_total
            rng = np.random.default_rng(self.seed + 33)
            q1s, q2s = [], []
            for _ in range(n_mc):
                noisy_inp = inp + torch.tensor(rng.normal(0.0, 0.05, size=inp.shape), dtype=torch.float32)
                p_s = self._net(noisy_inp, dt).numpy() * self._c_total
                q1_i, q2_i = _targets_from_o(times_min, p_s[:, 1] / max(self.q_nom, 1), self._c_total, self.q_nom)
                q1s.append(q1_i)
                q2s.append(q2_i)

        m_um = pred[:, 0]
        o_um = pred[:, 1] / max(self.q_nom, 1)
        mass_um = pred[:, 2]
        q1, q2 = _targets_from_o(times_min, o_um, self._c_total, self.q_nom)
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


class PINNMambaSolver(_BaseModernPINNSolver):
    """Inverse selective-SSM PINNMamba (ICML 2025) solver."""

    def __init__(self, **kwargs) -> None:
        super().__init__(arch="pinnmamba", **kwargs)


class PILNNCfCSolver(_BaseModernPINNSolver):
    """Physics-Informed Liquid Neural Network (PI-LNN / CfC) solver for irregular timestamps."""

    def __init__(self, **kwargs) -> None:
        super().__init__(arch="pilnn_cfc", **kwargs)


class PirateNetSolver(_BaseModernPINNSolver):
    """PirateNets (JMLR 2024) solver with zero-initialized residual gates."""

    def __init__(self, **kwargs) -> None:
        super().__init__(arch="piratenet", **kwargs)


class PINNsFormerSolver(_BaseModernPINNSolver):
    """PINNsFormer (ICLR 2024) solver with Wavelet activation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(arch="pinnsformer", **kwargs)
