# Noise-Path A/B Matrix

> **Date:** 2026-05-13

Four silence-capture conditions profiled on the BL-2021USB path:

| Condition | RMS dB | Median dB | Flatness | Top peak Hz | Top peak dB |
|-----------|--------|-----------|----------|-------------|-------------|
| 1. BL-2021USB alone | -45.98 | -46.07 | 0.0788 | 1004.2 | 27.13 |
| 2. Device 4 connected, idle | -45.75 | -45.78 | 0.0797 | 59.0 | 26.63 |
| 3. Device 4 plugged in, powered on | -45.60 | -45.70 | 0.0724 | 59.0 | 30.91 |
| 4. Device 4 silent stream open | -48.58 | -48.91 | 0.0484 | 59.0 | 29.33 |

## Key Finding

The 59 Hz contamination is not present when the BL-2021USB runs alone (condition 1, dominant peak at 1004 Hz instead). It appears as soon as device 4 is connected (condition 2) and grows stronger when the external speaker is powered on (condition 3, 30.91 dB).

## Interpretation

The low-frequency hum enters through the device 4 / external-speaker path, not the BL-2021USB mic itself. Likely causes: ground loop via the line-out cable, power-supply noise, or shielding contamination.

## Practical Recommendations

1. Try a different cable or grounding arrangement for device 4 before adding software denoising.
2. If hum persists, a narrow notch filter around 59 Hz is more appropriate than broadband denoising.
3. Keep the BL-2021USB cable away from the device 4 output cable.
4. Use dynamic energy-threshold calibration (approx -30 to -31 dB) for all live runs.
5. Do NOT jump to software denoising before fixing the fan — denoising can distort speaker embeddings.

## Artifacts

- `/tmp/noise_matrix_baseline_alone.json`
- `/tmp/noise_matrix_dev4_connected_idle.json`
- `/tmp/noise_matrix_dev4_plugged_in.json`
- `/tmp/noise_matrix_dev4_silent_stream.json`
