"""
mcx_data_generator.py - GPU Monte Carlo (MCX) Latin Hypercube Synthetic Engine
"""
import os, json, h5py
import numpy as np

def generate_synthetic_mcx_dataset(output_h5="mcx_synthetic_dataset.h5", n_samples=1000, n_optodes=32, seed=42):
    rng = np.random.default_rng(seed)
    mu_a = rng.uniform(0.005, 0.200, n_samples).astype(np.float32)
    mu_sp = rng.uniform(0.50, 3.00, n_samples).astype(np.float32)
    g = rng.uniform(0.80, 0.95, n_samples).astype(np.float32)
    r_coords = np.linspace(0.5, 15.0, n_optodes).astype(np.float32)

    reflectance = np.zeros((n_samples, n_optodes), dtype=np.float32)
    for i in range(n_samples):
        D = 1.0 / (3.0 * (mu_a[i] + mu_sp[i]))
        mu_eff = np.sqrt(3.0 * mu_a[i] * (mu_a[i] + mu_sp[i]))
        phi_0 = np.exp(-(mu_a[i] + mu_sp[i] / (1.0 - g[i])) * r_coords) / (4.0 * np.pi * r_coords**2)
        phi_s = np.exp(-mu_eff * r_coords) / (4.0 * np.pi * D * r_coords)
        reflectance[i] = phi_0 + phi_s

    os.makedirs(os.path.dirname(output_h5) if os.path.dirname(output_h5) else ".", exist_ok=True)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("reflectance", data=reflectance, compression="gzip")
        f.create_dataset("mu_a", data=mu_a)
        f.create_dataset("mu_sp", data=mu_sp)
        f.create_dataset("g", data=g)
        f.create_dataset("r_coords", data=r_coords)
    print(f"[MCX ENGINE] Saved {n_samples} profiles to {output_h5}")

if __name__ == "__main__":
    generate_synthetic_mcx_dataset("data_engine/mcx_synthetic_dataset.h5", n_samples=1000)
