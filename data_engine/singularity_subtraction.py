"""
singularity_subtraction.py - delta-Eddington Primary-Scattered Field Decomposition (Phi = Phi_0 + Phi_s)
"""
import numpy as np

def decompose_fluence_field(r_mm, total_fluence, mu_a, mu_sp, p0=1.0):
    mu_tr = mu_a + mu_sp
    phi_0 = (p0 * np.exp(-mu_tr * r_mm)) / (4.0 * np.pi * (r_mm ** 2) + 1e-12)
    phi_s = np.clip(total_fluence - phi_0, 0.0, None)
    return phi_0, phi_s
