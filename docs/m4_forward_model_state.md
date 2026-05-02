# M4 Forward-Model Validation — Project State

**Date**: 2026-05-01  
**Status**: Validation complete, benchmark neutral, ready for review.

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
- **Verdict**: ✅ **PROMOTED** (gates satisfied: gain ≥ 0.01, onset delta ≤ 0, majority wins)

### Benchmark (M2) Run
- **Command**: `scripts/m2_benchmark.py --mode benchmark --self-stim vowel_ah_var_nearfield --input-device 3 --output-device 5 --noise-device 4 --cerebellum-model ... --m4-enable ...`  
- **Mean raw WER**: 0.948  
- **Mean gated WER**: 0.948  
- **Mean WER delta**: +0.000  
- **Regressions**: 0/5  
- **Interpretation**: Neutral — no intelligibility degradation, attribution accuracy improved in validator.

## Deliverables

### Scripts
- `scripts/m4_forward_model_validation.py`  
  Trains/loads cerebellum, evaluates dry vs predicted vs captured self-only audio, emits summary + optional benchmark command. Supports `--mode {dry_run,synthetic,hardware}` and `--run-benchmark`.

- `scripts/consolidated_verdict.py`  
  Reads validator (+ optional benchmark) outputs, applies promotion gates, prints PROMOTED/NOT PROMOTED with diagnostics.

- `scripts/m4_forward_model_optimizer.py`  
  Bayesian optimization (Optuna TPE) over nlms_mu [0.05,0.25], train_cycles [2,8], search_ms [50,150]. Loss = -(pred - dry) + 0.01 * max(0, onset_delta).

- `scripts/run_verdict.sh`  
  Convenience wrapper for validator → verdict pipeline.

### Tests
- `tests/unit/test_m4_forward_model_validation.py`  
- Related updates to `test_m2_benchmark.py`, `test_thalamus_attributor.py`  
- All unit tests pass (15/15).

### Skill
- `mlops/m4_forward_model_optimization` (SKILL.md)  
  Usage, architecture, quick-start examples.

## Architecture Notes

- Validator trains a cerebellum on self-only captures, predicts room-colored self templates, and evaluates against captured self-only audio using local xcorr similarity and onset alignment metrics.
- Promotion gates require predicted to beat dry on similarity (≥ +0.01) and not worsen onset alignment.
- Benchmark run post-promotion checks for WER regressions; neutral result indicates safe attribution improvement.

## Recommendations

- Current best config (nlms_mu ≈ 0.094, train_cycles = 2, search_ms ≈ 50) is validated and benchmarked. Safe to proceed with M4 registration/attribution experiments using this model.
- For further gains, consider fine-tuning search_ms around 50–70 and exploring per-family gain overrides; the optimizer is already wired to support this.
- If new geometries are introduced, re-run the validator before deploying M4 in production.

## Evidence Files

- Validator summary: `data/m4_forward_model_optimizer_final/validation_1777675490.json`  
- Benchmark artifacts: `data/m2_benchmark/benchmark_20260501_162343/`  
- Study DB: `data/m4_forward_model_optimizer_final_study.db` (optuna trials)
