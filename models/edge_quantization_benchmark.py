"""
edge_quantization_benchmark.py - Per-Mode Affine INT8 PTQ & D-Optimal Optode Selector
"""
import numpy as np

def select_d_optimal_optodes(r_grid, mu_a=0.05, mu_sp=1.50, n_select=8):
    mu_eff = np.sqrt(3.0 * mu_a * (mu_a + mu_sp))
    j_a = -r_grid * (3.0 * (2.0 * mu_a + mu_sp)) / (2.0 * mu_eff) + 1.0 / (mu_a + mu_sp)
    j_sp = -r_grid * (3.0 * mu_a) / (2.0 * mu_eff) + 1.0 / (mu_a + mu_sp)
    J = np.stack([j_a, j_sp], axis=-1)
    selected, rem = [], list(range(len(r_grid)))
    for _ in range(n_select):
        best_i = max(rem, key=lambda idx: np.linalg.det(J[selected + [idx]].T @ J[selected + [idx]] + 1e-6 * np.eye(2)))
        selected.append(best_i)
        rem.remove(best_i)
    return np.sort(r_grid[selected])
