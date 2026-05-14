# M4 Live BL-2021USB Fine-Tuning Probe

> **Date:** 2026-05-14
> **Status:** Documented, not closing the gap

## Goal

Test whether additional live BL-2021USB room-colored negatives can close the fixed-threshold scorer gap.

## Baseline

`data/m4_finetuned_ecapa_bl2021_feature2`, held-out artifact `160348`:
- θ=0.50: eligible F1 0.7635, negative FP 81
- θ=0.70: eligible F1 0.7979, negative FP 62
- θ=0.90: negative FP 1, but self recall collapses (eligible F1 0.0500)

## Experiment A: Live Negatives Added

`data/m4_finetuned_ecapa_bl2021_live_neg_v1`:
- Trained on three earlier live negative playback windows + existing human negatives
- Training split: 40 self vs 330 other; internal test acc 0.875
- Held-out θ=0.70: eligible F1 0.9179, negative FP 15
- Materially better, but still fails acceptance.

## Experiment B: Live Negatives + Extra Same-Session Self Positives

`data/m4_finetuned_ecapa_bl2021_live_balanced_v1`:
- Added three earlier `self_only/live_captured.npy` positives
- Training split: 115 self vs 330 other; internal test acc 0.9565
- Held-out θ=0.70: eligible F1 0.8167, negative FP 49
- Internal split improved, held-out generalization worsened.

## Conclusion

Feature-based live-negative tuning helps but does not close the fixed-threshold gap. The acceptance path is conversational registration + per-speaker adaptive thresholds, not a single global scorer. More fixed-threshold fine-tuning needs a deliberately balanced live dataset with multiple self utterances and held-out sessions; otherwise it overfits.
