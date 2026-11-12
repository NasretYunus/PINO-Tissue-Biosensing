"""
benchtop_phantom_validator.py - 150-State Hybrid Bilayer PDMS/Intralipid-20%/Nigrosin Validator (v7)
Reference Membrane #0 (0.00 mg/g Nigrosin, 1.82 mg/g TiO2, d_epi = 100 um) -> G = [0.942, 0.961, 0.978, 1.019, 1.028, 1.035, 1.041, 1.046]^T
"""
import numpy as np

CALIB_GAIN_G_REF_MEMBRANE0 = np.array([0.942, 0.961, 0.978, 1.019, 1.028, 1.035, 1.041, 1.046], dtype=np.float32)

def compute_calibration_gains(v_on_ref, v_dark_ref, rd_mcx_ref):
    delta_v = np.mean(v_on_ref - v_dark_ref, axis=0)
    return rd_mcx_ref / np.maximum(delta_v, 1e-12)
