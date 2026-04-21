# M2 Speech-Aware Results: Preservation and Separation

> **Status:** Active Research
> **Date:** 2026-04-20
> **Scope:** M2 Speech-Aware Self/World Separation (Hardware)

## Overview

M2 focuses on the interaction between self-generated sound and external human speech. The goal is to suppress self-caused audio while preserving the intelligibility of world speech.

## Key Technical Achievements

### 1. VAD Integration
We implemented a low-latency VAD (Voice Activity Detection) wrapper in `packages/cochlea/vad.py` using `webrtcvad`. This provides frame-level (30ms) speech detection used to inform attribution decisions.

### 2. Speech-Aware Thalamic Attributor
The `ThalamicAttributor` was updated to accept `is_speech` as an input. 
- During overlap, if speech is detected, the system defaults to a **BOTH** verdict rather than a hard **SELF** verdict.
- This conservative policy ensures that world speech segments are not accidentally zeroed out by self-voice suppression.

### 3. Speech Preservation Harness (Hardware Verified)
We created `scripts/m2_speech_preservation_harness.py` to evaluate lexical preservation under overlap.
- **Setup**: Self-Babble (Internal Speaker) vs. World-Speech (External Speaker).
- **Metric**: Comparison of Whisper ASR results on Raw vs. Gated audio.
- **Result**: Successfully preserved "This is a test of World Speech Preservation" in the gated signal even while self-babbling was occurring.

## Hardware Performance Matrix (M2)

| Self Stimulus | World Stimulus | Gating Policy | Raw ASR Result | Gated ASR Result | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ba` strings | Human Speech | Keep `BOTH` | Intelligible | Intelligible | ✅ SUCCESS |
| `ma` (loud) | Human Speech | Keep `BOTH` | Intelligible | Intelligible | ✅ SUCCESS |

## Key Findings

1.  **Timing Conservatism**: Onset-based timing checks naturally push overlapping frames into the `BOTH` category once the initial transient is passed. This behavior, while intended as a safety measure for M1, is perfectly suited for M2 speech preservation.
2.  **Lexical Robustness**: Whisper ASR is robust enough to handle the rhythmic masking of babble, and our gating logic (preserving `BOTH` frames) ensures that no critical speech information is lost.

## Next Steps

1.  **Speech-Aware Suppression (Soft Gate)**: Instead of just keeping `BOTH` frames, implement spectral subtraction on those frames to improve signal-to-noise ratio for the world speech.
2.  **Agentive Listening (M3)**: Distinguish between "My Voice" and "Another Voice" more explicitly using embeddings, moving toward contingent turn-taking.
