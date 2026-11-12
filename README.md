# PINO-Tissue-Biosensing (v10 Final — Double-Blind Archival Package)

**Physics-Informed Neural Operators for Real-Time, Discretization-Invariant Optical Tissue Parameterization and Melanin-Decoupled Edge Biosensing**  
*Archived DOI: `10.5281/zenodo.14058924` | Anonymous Repository: `https://anonymous.4open.science/r/PINO-Tissue-Biosensing`*

## Quick Clean-Room Verification
Run the following two commands from a fresh clone of `github_repo/` to verify byte-for-byte file integrity and reproduce the key benchmark summaries:
```bash
sha256sum -c checksums.sha256
python3 reproduce_clean_room.py
```

## Architecture & Exact Memory Accounting
- **Spectral-Convolution Encoder + Global Spatial Mean Pooling + Two-Layer MLP `Q` (`models/pino_model_and_loss.py`)**:
  - Pointwise Lifting `P` (`3 x 48`): `144 B`
  - 4 Spectral Convolution Layers `R_phi` (`4 x 12 x 48 x 48 x 2`): `221,184 B` (`18.0 KiB/mode`)
  - 4 Local Bypass Layers `W_l` + Fused BatchNorm Scales (`4 x (48 x 48 + 48)`): `9,408 B`
  - Global Spatial Mean Pooling across the `N` radial nodes -> Pooled Two-Layer MLP `Q` (`48 -> 512 -> 2`: `24,576 + 512 + 1,024 + 2 = 26,114 B`, `25,600 MACs`, `66.1k` cycles)
  - 32-pt RFFT Twiddle Factors, PCHIP Knots & Calibration LUT: `1,536 B`
  - **Total Deployed Inference Footprint**: **`258,386 bytes` (`252.33 KiB` INT8; `1,009.32 KiB` FP32)** in `.axi_sram` (`0x24000000`, `240 MHz` AXI bus).
  - **Pruned Training-Only Auxiliary Fluence Head**: `48 -> 128 -> 1` pointwise Conv1d with bias (`48x128 + 128 + 128x1 + 1 = 6,401` parameters = `25,604 B` [`25.0 KiB`] in FP32 during offline training, or `6.25 KiB` at INT8), pruned prior to deployment (`0 B` overhead at inference).
  - **Active `DTCM-RAM` Working Memory (`0x20000000`, Zero-Wait-State @ `480 MHz`)**: **`29,488 bytes` (`28.80 KiB` INT8; `76.00 KiB` FP32)** = `8.00 KiB` double-buffered ping-pong ADC DMA pair (`2 x 8 ch x 256 samples x 2 B`) + `2.40 KiB` 32-pt RFFT/PCHIP scratchpad + `18.40 KiB` ping-pong INT8 neural activations/accumulators (`65.60 KiB` in FP32, `3.57x` reduction).
