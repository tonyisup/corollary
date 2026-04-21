# M1c Order Rule Grid Results (2026-04-21)

> Status: calibration analysis note for structured labeled runs. Not a promoted milestone claim.

## Purpose

Summarize what the larger 2026-04-21 M1c order-rule grid did and did not support.

This note updates the earlier tiny-batch hypothesis that `self_evidence_db <= 2.62` might separate:
- `world_before_self`
- `self_before_world`

## Data source

Primary file:
- `data/m1c_calibration/order_grid_20260421.jsonl`

Supporting scripts:
- `scripts/m1c_order_rule_grid.py`
- `scripts/m1c_order_rule_analyze.py`

## Grid scope

The order-rule grid collected labeled sequential runs across:
- `sequential_world_first`
- `sequential_self_first`

with:
- world stimuli: `chirp`, `vowel_ah`, `tone`
- multiple seeds
- primary and stress blocks
- train / validation / test-style splits
- varied gap and amplitude settings in stress cases

## High-level result

The earlier tiny 6-row scalar hypothesis did **not** generalize.

Using the provisional rule:

```text
world_before_self if self_evidence_db <= 2.62
```

on the 216-row grid gave approximately:
- accuracy: `0.537`
- balanced accuracy: `0.537`

That is too weak to treat as a stable order-calibration rule.

## Best global simple rules

### Best global single-feature threshold

Feature:
- `world_evidence_db`

Rule:

```text
world_before_self if world_evidence_db <= 5.005
```

Performance:
- accuracy: `0.639`

### Best global two-feature AND rule

Rule:

```text
world_before_self if
    self_evidence_db <= 6.155
    AND world_evidence_db <= 5.005
```

Performance:
- accuracy: `0.644`

Interpretation:
- this is only a very small gain over the best global single-feature rule
- global non-stimulus-aware rules remain relatively weak

## Best stimulus-aware baseline

The best current simple interpretable baseline is **stimulus-aware** thresholding on `world_evidence_db`.

### Stimulus-aware single-feature rule

Overall accuracy:
- `0.708`

Per stimulus:

#### chirp
```text
world_before_self if world_evidence_db <= 5.025
```
- accuracy: `0.667`

#### tone
```text
world_before_self if world_evidence_db <= 5.000
```
- accuracy: `0.750`

#### vowel_ah
```text
world_before_self if world_evidence_db <= 1.580
```
- accuracy: `0.708`

Interpretation:
- conditioning on `world_stim` materially helps
- this suggests the order signal is at least partly stimulus-dependent

## Best richer interpretable rule

A slightly stronger but more complex option is a **stimulus-aware two-feature AND rule** using `world_evidence_db` plus `world_score`.

Overall accuracy:
- `0.722`

Per stimulus:

#### chirp
```text
world_before_self if
    world_evidence_db <= 5.025
    AND world_score >= 0.042
```
- accuracy: `0.681`

#### tone
```text
world_before_self if
    world_evidence_db <= 5.000
    AND world_score >= 0.758
```
- accuracy: `0.764`

#### vowel_ah
```text
world_before_self if
    world_evidence_db <= 1.580
    AND world_score >= 0.109
```
- accuracy: `0.722`

Interpretation:
- this beats the stimulus-aware single-feature baseline (`0.722` vs `0.708`)
- the gain is real but modest
- the added complexity may not yet justify operational promotion

## Recommended interpretation

Current best honest statement:

1. a single global scalar threshold is not enough
2. stimulus-aware thresholding is meaningfully better
3. `world_evidence_db` is the strongest current simple feature
4. adding `world_score` helps a little when done per stimulus
5. even the better stimulus-aware rules are not yet strong enough to claim robust order calibration is solved

## Recommendation for use

### Operational baseline
Use the simpler stimulus-aware rule as the current baseline:
- per-stimulus thresholds on `world_evidence_db`

Reason:
- materially better than global rules
- easier to understand
- easier to falsify
- only slightly worse than the richer rule

### Analysis-only richer rule
Keep the stimulus-aware two-feature rule in analysis for now:
- per-stimulus `world_evidence_db`
- plus per-stimulus `world_score` gating

Reason:
- best current interpretable performance
- but only a small lift over the simpler baseline

## Practical next steps

1. gather more labeled runs under additional geometry families
2. check whether the stimulus-aware `world_evidence_db` ordering remains stable
3. retest whether the two-feature gain persists out of family
4. only promote a richer rule if it shows a meaningful repeatable gain
5. continue to treat all of this as calibration work, not milestone closure

## Conclusion

The main positive result from the larger grid is not that order calibration is solved.
It is that:

> **stimulus-aware calibration is a better path than global scalar thresholding**

That is a useful design constraint for the next iteration of adaptive attribution calibration.
