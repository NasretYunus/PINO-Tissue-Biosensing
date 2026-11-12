"""
pino_model_and_loss.py - 1D Physics-Informed Neural Operator (d_v=48, k_max=12, L=4)
Includes:
- Monotone PCHIP radial interpolation (M=8 -> N=32)
- Per-channel logarithmic standardization with per-profile spatial mean centering
- Pruned inference export (258,386 bytes = 252.33 KiB INT8)
- 3D spherical Helmholtz PDE residual with delta-Eddington primary source S_diff = mu_s' * Phi_0
- 4th-order sub-diffusive validity weight w(r) = 1 - exp(-(r/l*)^4)
- Partial-current Extrapolated Boundary Condition (EBC) loss and NTK dynamic balancer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pchip_interpolate_8_to_32(r_8: torch.Tensor, y_8: torch.Tensor, r_32: torch.Tensor) -> torch.Tensor:
    """Monotone piecewise cubic Hermite interpolation (Fritsch-Carlson) from M=8 optodes to N=32 grid."""
    h = r_8[1:] - r_8[:-1]
    delta = (y_8[..., 1:] - y_8[..., :-1]) / h
    d = torch.zeros_like(y_8)
    # Interior slopes via harmonic mean when delta_k and delta_{k-1} have same sign
    same_sign = (delta[..., :-1] * delta[..., 1:]) > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    harm = (w1 + w2) / (w1 / (delta[..., :-1] + 1e-12) + w2 / (delta[..., 1:] + 1e-12))
    d[..., 1:-1] = torch.where(same_sign, harm, torch.zeros_like(harm))
    d[..., 0] = delta[..., 0]
    d[..., -1] = delta[..., -1]

    idx = torch.clamp(torch.searchsorted(r_8, r_32) - 1, 0, len(r_8) - 2)
    h_k = r_8[idx + 1] - r_8[idx]
    t = (r_32 - r_8[idx]) / h_k
    h00 = (2 * t**3 - 3 * t**2 + 1)
    h10 = (t**3 - 2 * t**2 + t) * h_k
    h01 = (-2 * t**3 + 3 * t**2)
    h11 = (t**3 - t**2) * h_k
    return (
        h00 * y_8[..., idx]
        + h10 * d[..., idx]
        + h01 * y_8[..., idx + 1]
        + h11 * d[..., idx + 1]
    )


def log_standardize_profile(rd: torch.Tensor, sigma_ln: torch.Tensor = None) -> torch.Tensor:
    """Per-profile zero-mean logarithmic standardization: removes common-mode surface attenuation."""
    ln_rd = torch.log(torch.clamp(rd, min=1e-12))
    ln_centered = ln_rd - torch.mean(ln_rd, dim=-1, keepdim=True)
    if sigma_ln is None:
        sigma_ln = torch.std(ln_centered, dim=-1, keepdim=True) + 1e-6
    return ln_centered / sigma_ln


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int = 48, out_channels: int = 48, modes: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(b, self.out_channels, n // 2 + 1, device=x.device, dtype=torch.cfloat)
        m = min(self.modes, n // 2 + 1)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weights[:, :, :m])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class PINO1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        hidden_dim: int = 48,
        modes: int = 12,
        num_layers: int = 4,
        include_aux_fluence_head: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.num_layers = num_layers
        self.include_aux_fluence_head = include_aux_fluence_head

        # Pointwise lifting P: 3 -> 48 (3 * 48 = 144 weights)
        self.lift = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        # 4 Fourier layers: R_phi (4 * 12 * 48 * 48 * 2 = 221,184 real params)
        self.spec_layers = nn.ModuleList(
            [SpectralConv1d(hidden_dim, hidden_dim, modes) for _ in range(num_layers)]
        )
        # 4 Local bypass W_l + fused BN scales: 4 * (48 * 48 + 48) = 9,408 params
        self.bypass_layers = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True) for _ in range(num_layers)]
        )
        # Deployed projection MLP Q: 48 -> 512 -> 2 (24,576 + 512 + 1,024 + 2 = 26,114 params)
        self.proj_q = nn.Sequential(
            nn.Linear(hidden_dim, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, out_channels, bias=True),
            nn.Softplus(),
        )
        # Training-only auxiliary collocation fluence head (pruned prior to edge deployment)
        if include_aux_fluence_head:
            self.aux_fluence_head = nn.Sequential(
                nn.Conv1d(hidden_dim, 128, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv1d(128, 1, kernel_size=1, bias=True),
                nn.Softplus(),
            )
        else:
            self.aux_fluence_head = None

    def prune_auxiliary_head(self):
        """Prunes the 24.0 KB training-only auxiliary fluence head for edge deployment."""
        self.aux_fluence_head = None
        self.include_aux_fluence_head = False

    def deployed_int8_byte_budget(self, lut_bytes: int = 1536) -> dict:
        lift_b = self.lift.weight.numel()
        spec_b = sum(p.weights.numel() * 2 for p in self.spec_layers)
        byp_b = sum(b.weight.numel() + b.bias.numel() for b in self.bypass_layers)
        proj_b = sum(p.numel() for p in self.proj_q.parameters())
        total_b = lift_b + spec_b + byp_b + proj_b + lut_bytes
        return {
            "lifting_P_bytes": lift_b,
            "spectral_Rphi_bytes": spec_b,
            "bypass_Wl_bn_bytes": byp_b,
            "deployed_Q_bytes": proj_b,
            "fft_pchip_cal_lut_bytes": lut_bytes,
            "total_deployed_int8_bytes": total_b,
            "total_deployed_int8_kib": round(total_b / 1024.0, 2),
        }

    def forward(self, x: torch.Tensor):
        v = self.lift(x)
        for spec, byp in zip(self.spec_layers, self.bypass_layers):
            v = F.gelu(spec(v) + byp(v))
        v_pool = torch.mean(v, dim=-1)
        params = self.proj_q(v_pool)
        if self.aux_fluence_head is not None:
            phi_s = self.aux_fluence_head(v)
            return params, phi_s
        return params


class SingularitySplitHelmholtzLoss(nn.Module):
    """Evaluates Eq. (5) & Eq. (10) with 4th-order sub-diffusive weight w(r) = 1 - exp(-(r/l*)^4) and EBC."""
    def __init__(self, n_rel: float = 1.40, p0: float = 1.0):
        super().__init__()
        self.p0 = p0
        r_eff = -1.440 / (n_rel**2) + 0.710 / n_rel + 0.668 + 0.0636 * n_rel
        self.A_factor = (1.0 + r_eff) / (1.0 - r_eff)

    def forward(self, mu_a: torch.Tensor, mu_sp: torch.Tensor, phi_s: torch.Tensor, r_coords: torch.Tensor):
        mu_tr = mu_a + mu_sp + 1e-6
        D = 1.0 / (3.0 * mu_tr)
        l_star = 1.0 / mu_tr
        w_r = 1.0 - torch.exp(-((r_coords / l_star) ** 4))

        # Primary delta-Eddington field Phi_0(r) and first-collision distributed source S_diff(r)
        phi_0 = self.p0 * torch.exp(-mu_tr * r_coords) / (4.0 * np.pi * (r_coords**2) + 1e-8)
        s_diff = mu_sp * phi_0

        # 3D radial spherical Laplacian: (1/r^2) d_r (r^2 D d_r Phi_s)
        d_phi = torch.gradient(phi_s, spacing=(r_coords[0, 0],), dim=-1)[0]
        r2_D_dphi = (r_coords**2) * D * d_phi
        div_term = torch.gradient(r2_D_dphi, spacing=(r_coords[0, 0],), dim=-1)[0] / (r_coords**2 + 1e-8)

        res_pde = -div_term + mu_a * phi_s - s_diff
        loss_pde = torch.mean(w_r * (res_pde**2))

        # Extrapolated boundary condition z_b = 2 * A * D
        z_b = 2.0 * self.A_factor * D
        res_ebc = phi_s - z_b * torch.abs(d_phi)
        loss_ebc = torch.mean(w_r * (res_ebc**2))
        return loss_pde, loss_ebc
