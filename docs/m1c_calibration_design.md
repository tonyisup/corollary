# M1c Calibration Design Note

> Date: 2026-04-21
> Scope: first implementation slice for a learned M1c calibration layer

## Conversation summary

The discussion started from a question about whether M1 could use a neural
network with feedback and attention loops to learn self-attribution from
labeled external/internal sound data, with network parameters acting as a map
into sound-attribute parameters.

The answer we settled on was narrower:

- keep labeled M1c runs as supervision
- learn a bounded attribution/calibration model first
- use a latent probe decoder for sound-attribute tuning later
- do not treat raw network weights as the semantic map to sound attributes

## What we decided

The M1c learning path should be a bounded calibration layer, not a free-form
self-modifying network.

The first useful learned system is:

1. supervised on labeled M1c runs
2. trained on engineered low-level features, not raw waveforms
3. compared against the existing scalar and stimulus-aware thresholds
4. kept separate from the later probe-generation policy

## Data source

The first training data comes from the existing M1c JSONL runs in
`data/m1c_calibration/`, especially:

- `order_grid_20260421.jsonl`
- `runs.jsonl`

These records already contain the low-level evidence needed for calibration:

- `self_score`
- `world_score`
- `self_evidence_db`
- `world_evidence_db`
- onset and shift errors
- stimulus identity
- split labels

## Implementation shape

The first slice added a new package:

- `packages/m1c_calibration/`

with:

- `schema.py` for normalized rows and labels
- `features.py` for numeric feature extraction and one-hot categorical encoding
- `dataset.py` for JSONL loading and split resolution
- `baselines.py` for the reference threshold rules
- `metrics.py` for accuracy / balanced accuracy / macro F1
- `model.py` for a small Torch MLP with a latent layer
- `runner.py` for training, evaluation, and checkpoint IO

The new offline CLI is:

- `scripts/m1c_calibration_eval.py`

It trains the MLP on the training split, evaluates it on validation and test
splits, and prints results alongside the reference baselines.

## Baselines preserved

The implementation keeps the current published M1c calibration references
available for comparison:

- provisional `self_evidence_db` threshold
- stimulus-aware `world_evidence_db` thresholding

That matters because the learned model should beat a known interpretable rule,
not just exist.

## Next step after this slice

The next layer should be a feedback loop that uses the learned latent state to
propose the next probe parameters.

That would add:

- a history encoder
- a probe-parameter decoder
- a trust gate so the model only adapts when the held-out calibration behaves
  well

For now, the repository has the right scaffold to train and evaluate the first
bounded M1c calibration model.

## First evaluation result

On the current `order_grid_20260421.jsonl` split, the learned model is not yet
better than the published stimulus-aware threshold baseline.

Observed full-grid result from the first pass:

- reference stimulus-aware baseline test accuracy: about `0.685`
- learned MLP test accuracy: about `0.556`

That is a useful boundary condition, not a failure of the implementation:
the first slice successfully wires the training/eval loop, but the next version
needs a stronger inductive bias or a tighter feature set before it becomes
competitive with the existing rule-based baseline.
