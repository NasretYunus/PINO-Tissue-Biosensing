"""
test_pino_gradients.py - Pytest Verification of PINO v7 Parameter Budget (258,386 B) & Gradients
"""
import torch
from models.pino_model_and_loss import PINO1d, SingularitySplitHelmholtzLoss, pchip_interpolate_8_to_32, log_standardize_profile

def test_forward_gradients_and_pruned_budget():
    model = PINO1d(in_channels=3, out_channels=2, hidden_dim=48, modes=12, num_layers=4, include_aux_fluence_head=True)
    x = torch.randn(2, 3, 32, requires_grad=True)
    r_grid = torch.linspace(0.25, 15.0, 32).unsqueeze(0).unsqueeze(0)
    params, phi_s = model(x)
    assert params.shape == (2, 2)
    assert phi_s.shape == (2, 1, 32)
    loss_pde, loss_ebc = SingularitySplitHelmholtzLoss()(params[:, 0:1].unsqueeze(-1), params[:, 1:2].unsqueeze(-1), phi_s, r_grid)
    total_loss = params.sum() + loss_pde + loss_ebc
    total_loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()

    # Verify pruned inference parameter budget is exactly 258,386 bytes (252.33 KiB)
    model.prune_auxiliary_head()
    budget = model.deployed_int8_byte_budget()
    assert budget["lifting_P_bytes"] == 144
    assert budget["spectral_Rphi_bytes"] == 221184
    assert budget["bypass_Wl_bn_bytes"] == 9408
    assert budget["deployed_Q_bytes"] == 26114
    assert budget["fft_pchip_cal_lut_bytes"] == 1536
    assert budget["total_deployed_int8_bytes"] == 258386

    # Verify scale invariance of per-profile zero-mean log standardization under 0.37x (-8.6 dB) attenuation
    r_8 = torch.tensor([0.7, 1.4, 2.3, 6.8, 9.5, 12.0, 13.8, 14.8])
    r_32 = torch.linspace(0.7, 14.8, 32)
    rd_base = torch.exp(-0.45 * r_8) / (r_8 ** 2)
    rd_scaled = 0.37 * rd_base
    z_base = pchip_interpolate_8_to_32(r_8, log_standardize_profile(rd_base), r_32)
    z_scaled = pchip_interpolate_8_to_32(r_8, log_standardize_profile(rd_scaled), r_32)
    assert torch.allclose(z_base, z_scaled, atol=1e-6)
