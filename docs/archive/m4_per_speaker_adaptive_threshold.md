# M4 Per-Speaker Adaptive Threshold — Complete

> **Date:** 2026-05-12
> **Status:** Complete

## Problem

Fixed M4 threshold produces 13-43 FP per human speaker. ECAPA's frozen encoder maps self-TTS and human speech to overlapping embedding regions.

## Solution

`SpeakerThresholdAdapter` tracks per-speaker confidence distributions and raises the decision boundary for overlapping speakers.

**Threshold logic:** `threshold = max(base_threshold, max_observed_confidence + margin)`, capped at `max_threshold`. Adapts from frame 1 — no warm-up delay. SELF frames always use base threshold (separate profile key).

## Results (6 BL-2021USB human speakers)

0 FP, >90% self-recall for all speakers.

## Pipeline

```
Audio → VAD → Attribution (with speaker_id) → SpeakerThresholdAdapter
  → per-speaker threshold → M4 gate decision
  → Frame log (with speaker_id) → SelfModel → SAG known_subjects
```

## Files

- `packages/thalamus/speaker_threshold_adapter.py` — Core adapter (15 unit tests)
- `packages/thalamus/attributor.py` — Integrated at 3 gate points
- `packages/pfc/streaming_runtime.py` — Speaker ID + adapter wired in
- `packages/pfc/self_model.py` — Per-speaker ID extraction from frame logs
- 23 total new tests across unit, integration, and e2e

## Key Insight

SELF confidences [0.30, 0.81] barely overlap with WORLD confidences [-0.13, 0.30]. Per-speaker adaptation exploits this gap; a fixed threshold cannot.

## Caveat

Speaker identity must come from upstream STG. Without `speaker_id`, falls back to single "unknown" profile. The streaming runtime wires speaker_id through — see `docs/archive/m4_conversational_registration.md`.
