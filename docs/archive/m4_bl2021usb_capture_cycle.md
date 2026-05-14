# BL-2021USB Capture Cycle — Complete

> **Date:** 2026-05-11
> **Status:** Complete

6 human negative speakers captured, artifacts generated, feature-based fine-tuned scorer trained.

## Artifacts

- `data/m4_clean_captures_bl2021/` — Clean self-registration
- `data/m4_human_negative_bl2021/` — 6 raw human negative recordings
- `data/m4_human_negative_bl2021_artifacts/` — 6 speaker artifact dirs
- `data/m4_validation_human_bl2021_speaker1/` through `speaker6/` — Full validation sets
- `data/m4_finetuned_ecapa_bl2021_feature2/` — Feature-based fine-tuned scorer

## Feature-Based Scorer Result

Best fixed-threshold result: F1=0.91, 13 FP at θ=0.50. See `references/m4-bl2021usb-hardware-rebaseline-20260511.md` for the full 5-phase cycle.

## Fine-Tuning Rule

For datasets with <500 training pairs, always use feature-based fine-tuning (frozen encoder + MLP head). Full encoder fine-tuning catastrophically forgets pre-trained speaker discrimination (F1=0.28 vs 0.83 feature-based).
