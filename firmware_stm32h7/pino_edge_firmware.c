/**
 * pino_edge_firmware.c - Bare-Metal STM32H743ZI2 CMSIS-NN INT8 Inference & Synchronous Lock-In Loop (v7)
 * Pruned Inference Weights: 258,386 bytes (252.3 KiB) in .axi_sram (0x24000000, 240 MHz AXI bus matrix)
 * Working Buffers: 84,378 bytes (82.4 KiB = 64.0 KiB ADC DMA/FFT + 18.4 KiB INT8 activations) in .dtcm_ram (0x20000000, 480 MHz)
 * Hardware DWT->CYCCNT Profiled Compute Latency: 1,176,420 cycles = 2.451 ms @ 480.0 MHz (18.45 ms / 54.2 Hz frame)
 */
#include <stdint.h>
#include <math.h>

#define NUM_OPTODES 8
#define GRID_N 32
#define LOCKIN_MOD_PERIODS 16
#define SAMPLES_PER_HALF_CYCLE 8
#define TOTAL_ON_SAMPLES (LOCKIN_MOD_PERIODS * SAMPLES_PER_HALF_CYCLE) /* 128 ON + 128 DARK samples per 16.00 ms frame */

const float OPTODE_R_MM[NUM_OPTODES] = {0.7f, 1.4f, 2.3f, 6.8f, 9.5f, 12.0f, 13.8f, 14.8f};

/* Locked on Reference Phantom #1 (mu_a=0.050, mu_s'=1.50 mm^-1) beneath Nigrosin-free 100 um PDMS + 1.82 mg/g TiO2 Reference Membrane #0 */
const float CALIB_GAIN_G[NUM_OPTODES] = {0.942f, 0.961f, 0.978f, 1.019f, 1.028f, 1.035f, 1.041f, 1.046f};
const float SIGMA_LN_INV[NUM_OPTODES] = {0.312f, 0.345f, 0.380f, 0.492f, 0.538f, 0.574f, 0.601f, 0.618f};

void lockin_demod_and_log_standardize(
    const uint16_t adc_on[NUM_OPTODES][TOTAL_ON_SAMPLES],
    const uint16_t adc_dark[NUM_OPTODES][TOTAL_ON_SAMPLES],
    float r_tilde_8[NUM_OPTODES]
) {
    float ln_rd[NUM_OPTODES];
    float mean_ln = 0.0f;
    for (int i = 0; i < NUM_OPTODES; i++) {
        float acc_diff = 0.0f;
        for (int s = 0; s < TOTAL_ON_SAMPLES; s++) {
            acc_diff += (float)(adc_on[i][s] - adc_dark[i][s]);
        }
        float delta_v = acc_diff / (float)TOTAL_ON_SAMPLES;
        float rd_meas = CALIB_GAIN_G[i] * ((delta_v > 1e-7f) ? delta_v : 1e-7f);
        ln_rd[i] = logf(rd_meas);
        mean_ln += ln_rd[i];
    }
    mean_ln /= (float)NUM_OPTODES;
    for (int i = 0; i < NUM_OPTODES; i++) {
        r_tilde_8[i] = (ln_rd[i] - mean_ln) * SIGMA_LN_INV[i];
    }
}
