"""Amortized Sequence SBI: FMPE / Conditional Flow Matching (CFM) with distinct Mamba/LNN/Transformer encoders."""

import numpy as np
import torch
import torch.nn as nn


class _MambaS4Encoder(nn.Module):
    """Causal diagonal state-space encoder (distinct from GRU)."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.a_log = nn.Parameter(torch.zeros(emb_dim))
        self.b_proj = nn.Linear(emb_dim, emb_dim)
        self.gate = nn.Linear(emb_dim, emb_dim)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        # h_seq: (B, T, emb_dim)
        b_size, t_len, d = h_seq.shape
        state = torch.zeros(b_size, d, dtype=h_seq.dtype, device=h_seq.device)
        a_decay = torch.exp(-torch.exp(self.a_log) * 0.2)
        for t in range(t_len):
            state = a_decay * state + self.b_proj(h_seq[:, t, :]) * torch.sigmoid(self.gate(h_seq[:, t, :]))
        return state


class _LNNCfCEncoder(nn.Module):
    """Continuous-time CfC encoder with liquid time-constant gating."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Tanh())
        self.ff2 = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.Tanh())
        self.tau_gate = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        b_size, t_len, d = h_seq.shape
        h = torch.zeros(b_size, d, dtype=h_seq.dtype, device=h_seq.device)
        for t in range(t_len):
            cat = torch.cat([h_seq[:, t, :], h], dim=-1)
            interp = torch.sigmoid(self.tau_gate(cat))
            h = self.ff1(cat) * (1.0 - interp) + self.ff2(cat) * interp
        return h


class _SequenceEncoder(nn.Module):
    """Order-sensitive trajectory encoder with sinusoidal/temporal positional encoding."""

    def __init__(self, in_dim: int, emb_dim: int = 32, backbone: str = "lnn_cfc") -> None:
        super().__init__()
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.in_proj = nn.Linear(in_dim + 1, emb_dim)  # +1 for explicit normalized timestamp
        if backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=1)
        elif backbone == "mamba_s4":
            self.seq = _MambaS4Encoder(emb_dim)
        elif backbone == "lnn_cfc":
            self.seq = _LNNCfCEncoder(emb_dim)
        else:
            raise ValueError(f"Unknown sequence encoder backbone: {backbone}")

    def forward(self, x_seq: torch.Tensor, t_norm: torch.Tensor | None = None) -> torch.Tensor:
        b_size, t_len, _ = x_seq.shape
        if t_norm is None:
            t_axis = torch.linspace(0.0, 1.0, t_len, device=x_seq.device).view(1, t_len, 1).expand(b_size, t_len, 1)
        else:
            t_axis = t_norm.unsqueeze(-1)
        # Sinusoidal positional embedding added to projected features (breaks permutation invariance)
        pos = torch.arange(t_len, device=x_seq.device, dtype=x_seq.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.emb_dim, 2, device=x_seq.device, dtype=x_seq.dtype) * (-np.log(100.0) / self.emb_dim))
        pe = torch.zeros(1, t_len, self.emb_dim, device=x_seq.device, dtype=x_seq.dtype)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)

        h = torch.tanh(self.in_proj(torch.cat([x_seq, t_axis], dim=-1))) + 0.25 * pe
        if self.backbone == "transformer":
            out = self.seq(h)
            return out[:, -1, :]  # causal-end readout preserves sequence directionality
        return self.seq(h)


class FMPECFMSolver:
    """R1 Amortized Flow-Matching Posterior Estimation (FMPE / CFM) over kinetic targets."""

    def __init__(
        self,
        backbone: str = "lnn_cfc",  # {"transformer", "mamba_s4", "lnn_cfc"}
        emb_dim: int = 32,
        n_epochs: int = 60,
        lr: float = 5e-3,
        seed: int = 0,
    ) -> None:
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.seed = seed
        self._encoder: _SequenceEncoder | None = None
        self._vf: nn.Module | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._theta_mean: np.ndarray | None = None
        self._theta_std: np.ndarray | None = None

    def fit(self, x_seqs: np.ndarray, theta_targets: np.ndarray) -> "FMPECFMSolver":
        """Train conditional flow matching vector field v_theta(z_t, t | enc(x_seq))."""
        torch.manual_seed(self.seed)
        self._x_mean = np.mean(x_seqs, axis=(0, 1), keepdims=True)
        self._x_std = np.maximum(np.std(x_seqs, axis=(0, 1), keepdims=True), 1e-3)
        self._theta_mean = np.mean(theta_targets, axis=0, keepdims=True)
        self._theta_std = np.maximum(np.std(theta_targets, axis=0, keepdims=True), 1e-3)

        x_t = torch.tensor((x_seqs - self._x_mean) / self._x_std, dtype=torch.float32)
        th_t = torch.tensor((theta_targets - self._theta_mean) / self._theta_std, dtype=torch.float32)
        d_th = th_t.shape[1]

        self._encoder = _SequenceEncoder(x_seqs.shape[2], self.emb_dim, self.backbone)
        self._vf = nn.Sequential(
            nn.Linear(d_th + 1 + self.emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, d_th),
        )
        opt = torch.optim.Adam(list(self._encoder.parameters()) + list(self._vf.parameters()), lr=self.lr)

        for _ in range(self.n_epochs):
            opt.zero_grad()
            t_flow = torch.rand(x_t.shape[0], 1)
            z0 = torch.randn_like(th_t)
            zt = (1.0 - t_flow) * z0 + t_flow * th_t
            target_v = th_t - z0
            cond = self._encoder(x_t)
            pred_v = self._vf(torch.cat([zt, t_flow, cond], dim=-1))
            loss = torch.mean((pred_v - target_v) ** 2)
            loss.backward()
            opt.step()
        return self

    def sample_posterior(
        self,
        x_seqs: np.ndarray,
        n_samples: int = 32,
        n_steps: int = 20,
    ) -> np.ndarray:
        """Integrate learned CFM ODE from t=0 to t=1 to draw posterior samples (B, n_samples, d_theta)."""
        assert self._encoder is not None and self._vf is not None
        assert self._x_mean is not None and self._x_std is not None
        assert self._theta_mean is not None and self._theta_std is not None

        torch.manual_seed(self.seed + 100)
        x_t = torch.tensor((x_seqs - self._x_mean) / self._x_std, dtype=torch.float32)
        b_size = x_t.shape[0]
        d_th = self._theta_mean.shape[1]

        with torch.no_grad():
            cond = self._encoder(x_t).unsqueeze(1).expand(b_size, n_samples, self.emb_dim)
            z = torch.randn(b_size, n_samples, d_th)
            dt = 1.0 / n_steps
            for step in range(n_steps):
                t_val = torch.full((b_size, n_samples, 1), step * dt)
                v = self._vf(torch.cat([z, t_val, cond], dim=-1))
                z = z + dt * v
            samples = z.numpy() * self._theta_std[None, :, :] + self._theta_mean[None, :, :]
        return samples

    def predict_with_uncertainty(self, x_seqs: np.ndarray, n_samples: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and std across drawn CFM samples."""
        samples = self.sample_posterior(x_seqs, n_samples=n_samples)
        return np.mean(samples, axis=1), np.std(samples, axis=1)
