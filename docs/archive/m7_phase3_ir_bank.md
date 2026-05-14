# M7 Phase 3: Multi-Hypothesis IR Bank

> **Date:** 2026-05-14
> **Status:** Complete

## Concept

Maintain 3–5 candidate room IR models (hypotheses) simultaneously. On each frame, score each model's prediction error. Adapt the best one. When drift is detected, spawn a new candidate. When one model is clearly winning, prune the losers.

This provides faster recovery than single-filter re-adaptation because alternative hypotheses are already warmed up.

## Implementation

New module: `packages/cerebellum/ir_bank.py`

### IRHypothesis

A single room IR candidate with its own NLMS state:
- `score(known, captured)` — returns RMS prediction error
- `adapt(known, captured)` — runs one NLMS update step using stochastic gradient sampling
- `mean_error` — rolling average of recent errors
- `age` — number of adaptation steps

### IRBank

Manages multiple IR hypotheses:
- `score_all(known, captured)` — scores all hypotheses, returns error list
- `adapt_best(known, captured)` — adapts the best-scoring hypothesis
- `spawn_hypothesis()` — creates a new hypothesis (up to max_hypotheses)
- `prune_worst(target_count)` — removes worst hypotheses
- `best_hypothesis_index` / `best_hypothesis` — current best candidate

### Integration into BabblingCerebellum

- `set_ir_bank(bank)` attaches an `IRBank` to the cerebellum
- `_active_ir` property returns the bank's best hypothesis IR (or the single IR if no bank)
- `predict_pcm()` and `predict_mel()` use `_active_ir` automatically
- The drift alarm (phase 2) feeds into the bank's spawn/prune decisions

### Performance

The `adapt()` method uses stochastic gradient sampling (every Nth sample) rather than per-sample NLMS, making it fast enough for real-time use with 3–5 hypotheses.

## Tests

12 new unit tests in `tests/unit/test_cerebellum_ir_bank.py`:
- IRHypothesis error tracking and serialization
- IRBank scoring, adaptation, spawning, pruning
- Cerebellum integration: bank attachment and drift switching
