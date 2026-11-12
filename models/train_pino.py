"""
train_pino.py - End-to-End PyTorch Training & Pruning Pipeline for PINO (v7)
"""
import torch
import torch.optim as optim
from models.pino_model_and_loss import PINO1d, SingularitySplitHelmholtzLoss

def run_smoke_training():
    torch.manual_seed(42)
    model = PINO1d(in_channels=3, out_channels=2, hidden_dim=48, modes=12, num_layers=4, include_aux_fluence_head=True)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    phys_loss_fn = SingularitySplitHelmholtzLoss(n_rel=1.40)

    r_grid = torch.linspace(0.25, 15.0, 32).unsqueeze(0).unsqueeze(0)
    x = torch.randn(8, 3, 32, requires_grad=True)
    y = torch.tensor([[0.05, 1.50]] * 8)

    params, phi_s = model(x)
    loss_data = torch.mean((params - y) ** 2)
    loss_pde, loss_ebc = phys_loss_fn(params[:, 0:1].unsqueeze(-1), params[:, 1:2].unsqueeze(-1), phi_s, r_grid)
    loss = loss_data + 0.1 * loss_pde + 0.05 * loss_ebc
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.prune_auxiliary_head()
    budget = model.deployed_int8_byte_budget()
    print(f"[TRAIN v7] Smoke step complete. Loss={loss.item():.6f} | Pruned INT8 Budget={budget['total_deployed_int8_bytes']} B ({budget['total_deployed_int8_kib']} KiB)")

if __name__ == "__main__":
    run_smoke_training()
