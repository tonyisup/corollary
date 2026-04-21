# M1c Provisional Order Rule (2026-04-21)

> Status: provisional calibration hypothesis, not a promoted system constant.

## Purpose

Document the first tiny hand-checkable candidate rule for distinguishing:
- `world_before_self`
- `self_before_world`

from the first labeled M1c calibration rows gathered on 2026-04-21.

## Dataset used

Source:
- `data/m1c_calibration/runs.jsonl`

Current subset used for this hypothesis:
- 3 rows labeled `world_before_self`
- 3 rows labeled `self_before_world`
- same session family / geometry family
- same self stimulus: `babble_ba_var`
- world stimuli: `chirp`, `vowel_ah`, `tone`

This is a tiny dataset and should be treated accordingly.

## Candidate rule

The best single-feature separator on this 6-row batch was:

```text
if self_evidence_db <= 2.62:
    predict world_before_self
else:
    predict self_before_world
```

## Why this rule was chosen

Several simple scalar rules were checked on the 6 labeled rows, including:
- `world_score`
- `self_score`
- `world_score - self_score`
- `world_evidence_db`
- `self_evidence_db`
- `world_evidence_db - self_evidence_db`

Observed result on this batch:
- `self_evidence_db` achieved `6/6` correct
- `world_evidence_db` achieved `5/6`
- `world_score` achieved `4/6`

This makes `self_evidence_db` the current leading scalar candidate for sequence-order calibration.

## Evidence values

### `world_before_self`
- `chirp`: `self_evidence_db = 2.60`
- `vowel_ah`: `self_evidence_db = 2.21`
- `tone`: `self_evidence_db = 2.49`

### `self_before_world`
- `chirp`: `self_evidence_db = 3.22`
- `vowel_ah`: `self_evidence_db = 2.64`
- `tone`: `self_evidence_db = 3.07`

Current gap:
- max `world_before_self` = `2.60`
- min `self_before_world` = `2.64`

That leaves a narrow but clean split at roughly `2.62` in this batch.

## Interpretation

This suggests that, for the current setup family, sequence order may be reflected more strongly by:
- **self-window evidence strength**

than by raw `world_score` alone.

A plausible explanation is:
- when self occurs first, the self-expected window is energized more cleanly
- when self occurs second, self-window evidence is weaker or more contaminated by preceding world activity / room dynamics

This explanation is still a hypothesis.

## Important cautions

Do **not** treat `2.62` as a universal threshold yet.

Current limitations:
- only 6 rows
- one geometry family
- one session family
- likely sensitive to room, mic placement, gain, and device mapping

Therefore this rule should be treated as:
- a provisional calibration probe
- something to falsify on more runs

not as:
- a stable global constant
- a final attribution policy

## Implementation status

As of 2026-04-21, `scripts/m1c_hardware_harness.py` was updated to emit:
- `provisional_order_threshold_db`
- `provisional_order_prediction`

using this current hypothesis.

That change exists to make the hypothesis visible during future runs so it can be confirmed or rejected quickly.

## Next validation steps

1. repeat additional `world_before_self` and `self_before_world` batches on the same geometry
2. test a second geometry / placement family
3. see whether `self_evidence_db` remains the leading scalar feature
4. if it does not, move to a 2-feature calibration rule
5. only promote anything after held-out validation and replay-safety checks

## Current recommendation

Use this rule as:
- an explicit M1c calibration hypothesis
- a guide for further data collection

Do not promote it beyond provisional status until it survives more runs.
