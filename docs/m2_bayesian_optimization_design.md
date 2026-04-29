# M2 Hyperparameter Optimization via Bayesian Search

## Problem

M2 has many interacting parameters — onset latency, causal tolerance, similarity threshold, gain per family — that collectively determine attribution quality and speech preservation. Manual grid search is intractable: 5 knobs × 10 values each ≈ 100K trials. Each trial is ~6 seconds of audio.

## Solution

Bayesian optimization (Bayesian optimization using Gaussian processes or tree-structured Parzen estimators. TPE builds a probabilistic model of the loss landscape from previous trials and selects the next point expected to yield the highest improvement (expected improvement). This converges 10-100x faster than random search.

## Components

### 1. Loss Function

Single scalar combining multiple objectives:
- Primary: external speech WER (lower is better)
- Secondary: attribution accuracy (penalty for wrong label)
- Tertiary: WER degradation vs control

Loss = WER + α*attribution_penalty + β*wer_degradation

### 2. Parameter Space

**M2 audio gating parameters:**
- `onset_latency_ms`: [200, 600] (continuous)
- `causal_tolerance_ms`: [50, 100] (continuous)
- `similarity_threshold`: [0.10, 0.35] (continuous)
- `onset_method`: {energy, spectral flux} (categorical)

**M2 policy parameters:**
- `vowel_gain`: [0.3, 0.8] (continuous)
- `babble_gain`: [0.8, 1.0] (continuous)

### 3. Trial Runner

Single command that runs one configuration and outputs loss:

```bash
python scripts/m2_single_trial.py --config config.json --output trial_result.json
```

Returns: `{loss, wer, attribution_accuracy, metadata}`

### 4. Optimizer Loop

```python
study = optuna.create_study(direction="minimize")

for trial in range(N):
    params = study.suggest_params()
    result = run_trial(params)
    study.report(result.loss, params)
```

## Why not autoresearch?

Autoresearch is designed for neural architecture search and model training. This is classical hyperparameter optimization. Optuna is purpose-built for exactly this.

## Success Criteria

- Converges to settings that beat current manual calibration
- Reproducible across multiple optimization runs
- Works on both vowel and babble families
- Doesn't regress self-attribution