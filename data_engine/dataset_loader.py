"""
dataset_loader.py - PyTorch HDF5 Dataset with Per-Profile Zero-Mean Log Standardization & PCHIP Resampling
"""
import h5py, torch
import numpy as np
from torch.utils.data import Dataset

class MCXTissueDataset(Dataset):
    def __init__(self, hdf5_filepath, split="train", train_ratio=0.85):
        self.filepath = hdf5_filepath
        with h5py.File(self.filepath, "r") as f:
            n = f["reflectance"].shape[0]
            s = int(n * train_ratio)
            self.indices = np.arange(0, s) if split == "train" else np.arange(s, n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        with h5py.File(self.filepath, "r") as f:
            rd_raw = f["reflectance"][i].astype(np.float32)
            mu_a = f["mu_a"][i].astype(np.float32)
            mu_sp = f["mu_sp"][i].astype(np.float32)
            r = f["r_coords"][()].astype(np.float32)
        ln_rd = np.log(np.maximum(rd_raw, 1e-12))
        ln_std = (ln_rd - np.mean(ln_rd)) / (np.std(ln_rd) + 1e-6)
        x = np.stack([ln_std, r, np.log(r + 1e-3)], axis=0)
        y = np.array([mu_a, mu_sp], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)
