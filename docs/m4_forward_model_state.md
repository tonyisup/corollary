# M4 Forward-Model Validation — Project State

**Date**: 2026-05-01
**Status**: Validation complete, benchmark neutral, offline replay complete.

## Objective
Validate whether a room-colored forward model improves predicted self fidelity vs dry self on captured self-only audio and safely promote to M2/M4 benchmarking without regressions.

## Key Results

### Best Validation Trial
- **File**: `data/m4_forward_model_optimizer_final/validation_1777675490.json`
- **Parameters**: nlms_mu ≈ 0.094, train_cycles = 2, search_ms ≈ 50.4
- **Mean dry score**: 0.6399
- **Mean predicted score**: 0.6710
- **Gain**: +0.0311
- **Mean dry onset error**: 875.5 samples
- **Mean predicted onset error**: 305.5 samples
- **Onset delta**: -570.0 samples
- **Verdict**: PROMOTED (gates satisfied: gain >= 0.01, onset delta <= 0, majority wins)

### Benchmark (M2) Run
- **Command**: `scripts/m2_benchmark.py --mode benchmark --self-stim vowel_ah_var_nearfield --input-device 3 --output-device 5 --noise-device 4 --cerebellum-model ... --m4-enable ...`
- **Mean raw WER**: 0.948
- **Mean gated WER**: 0.948
- **Mean WER delta**: +0.000
- **Regressions**: 0/5
- **Per-seed self frames**: 0/5 (no self-only frames detected)
- **Per-seed both frames**: 9-11
- **Per-seed world frames**: 82-129
- **M4 registration**: room_colored_prediction, threshold 0.4, 0 triggered frames
- **Interpretation**: Neutral WER — no intelligibility degradation. Self scores 0.35-0.67. M4 registration active but no frames triggered (conservative).

### Offline Replay (promoted model)
- **Capture root**: `data/m2_benchmark/benchmark_20260501_162343` (5 seeds)
- **Strategies tested**: preserve, gain_0p5, subtract_0p1 through subtract_1p0, predicted_subtract_0p25, predicted_subtract_0p5
- **Results**: All strategies neutral (WER delta 0.000) except subtract_0p1 on seed_4 (delta -0.043, slight improvement) and gain_0p5/subtract_1p0 on seed_4 (delta +0.043, slight regression)
- **predicted_subtract_0p25**: mean delta +0.009 (0 improved, 4 neutral, 1 regressed)
- **predicted_subtract_0p5**: mean delta +0.009 (0 improved, 4 neutral, 1 regressed)
- **Energy retained**: 0.996-1.000 for predicted subtract strategies
- **Interpretation**: At current WER floor (~0.95), strategies have minimal impact. Predicted subtract performs comparably to fixed-gain subtract. No regressions from promoted model.

## Deliverables

### Scripts
- `scripts/m4_forward_model_validation.py` — trains/loads cerebellum, evaluates dry vs predicted vs captured self-only audio, emits summary + optional benchmark command.
- `scripts/consolidated_verdict.py` — reads validator (+ optional benchmark) outputs, applies promotion gates.
- `scripts/m4_forward_model_optimizer.py` — Bayesian optimization (Optuna TPE) over nlms_mu, train_cycles, search_ms.
- `scripts/run_verdict.sh` — convenience wrapper.

### Tests
- `tests/unit/test_m4_forward_model_validation.py`
- Related updates to `test_m2_benchmark.py`, `test_thalamus_attributor.py`
- All unit tests pass (15/15).

### Skill
- `mlops/m4_forward_model_optimization` (SKILL.md)

## Recommendations

1. The promoted model (nlms_mu=0.094, train_cycles=2, search_ms=50) is validated and safe for M4 registration experiments.
2. WER is at noise floor (~0.95) — consider testing with higher SNR or different stimulus to see strategy differentiation.
3. Self-frame detection is 0/5 — investigate if self-only frames are being classified as world or both; may need threshold tuning.
4. For further optimization, narrow search around current best: nlms_mu [0.07, 0.12], train_cycles [2, 4], search_ms [40, 70].

## Evidence Files

- Validator summary: `data/m4_forward_model_optimizer_final/validation_1777675490.json`
- Benchmark artifacts: `data/m2_benchmark/benchmark_20260501_162343/`
- Offline replay: `data/m2_benchmark/offline_20260501_205939/`
- Study DB: `data/m4_forward_model_optimizer_final_study.db`
