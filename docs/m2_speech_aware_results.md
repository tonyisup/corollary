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

### 3. Speech Preservation Harness (Early Hardware Prototype)
We created `scripts/m2_speech_preservation_harness.py` to evaluate lexical preservation under overlap.
- **Setup**: Self-Babble (Internal Speaker) vs. World-Speech (External Speaker).
- **Metric**: Comparison of Whisper ASR results on Raw vs. Gated audio.
- **Current evidence**: the harness and gating path exist, but any lexical-preservation claim should be treated as preliminary until rerun with saved outputs and clearer quantitative reporting.

## Prototype Observation Matrix (M2)

| Self Stimulus | World Stimulus | Gating Policy | Raw ASR Result | Gated ASR Result | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ba` strings | Human Speech | Keep `BOTH` | Preliminary / not independently rechecked here | Preliminary / not independently rechecked here | promising but not yet promotion-grade |
| `ma` (loud) | Human Speech | Keep `BOTH` | Preliminary / not independently rechecked here | Preliminary / not independently rechecked here | promising but not yet promotion-grade |

## Key Findings

1.  **Timing Conservatism**: Onset-based timing checks naturally push overlapping frames into the `BOTH` category once the initial transient is passed. This behavior, while intended as a safety measure for M1, is perfectly suited for M2 speech preservation.
2.  **Lexical robustness is still a hypothesis under test**: the current gating logic preserves `BOTH` frames, which is a sensible conservative prototype policy, but the repo still needs rerun evidence before claiming reliable speech preservation.

## Next Steps

1.  **Speech-Aware Suppression (Soft Gate)**: Instead of just keeping `BOTH` frames, implement spectral subtraction on those frames to improve signal-to-noise ratio for the world speech.
2.  **Agentive Listening (M3)**: Distinguish between "My Voice" and "Another Voice" more explicitly using embeddings, moving toward contingent turn-taking.

## 2026-04-29 M2 Baseline Runs

Hardware path:
- input: device `3` (`fifine Microphone`, 48 kHz input)
- self output: device `5` (`Mac mini Speakers`)
- world/noise output: device `4` (`External Headphones`)

Harness update:
- `scripts/m2_speech_preservation_harness.py` now accepts `--seed`, `--self-duration-ms`, and `--self-amplitude` so stronger overlap cases can be tested without changing attribution logic.
- The harness saves both binary `gating_mask.npy` and numeric `gating_gain.npy`; numeric gain is used for soft attenuation on `BOTH` frames.

Saved runs:

| Save dir | Stimulus | BOTH gain | Self score | Speech ratio | SELF/BOTH/WORLD frames | Raw WER | Gated WER | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `data/m2_baseline/babble_ba_var_seed0_in3_out5_noise4_ref_20260429_090827` | `babble_ba_var`, nominal 350 ms, amp 0.7 | 0.50 | 0.4266 | 98.1% | 0 / 12 / 148 | 0.125 | 0.125 | Good speech-preservation baseline; light overlap only. |
| `data/m2_baseline/babble_ba_repeat_tight_seed0_in3_out5_noise4_strong_ref_20260429_090912` | `babble_ba_repeat_short_tight_var`, nominal 900 ms, amp 1.0 | 0.50 | 0.2645 | 100.0% | 0 / 29 / 131 | 0.500 | 0.500 | Stronger overlap, preservation not worsened, but ASR quality is poor. |
| `data/m2_baseline/babble_ba_repeat_tight_seed0_in3_out5_noise4_gain1p0_20260429_090947` | same strong overlap | 1.00 | 0.2788 | 98.1% | 0 / 29 / 131 | 0.500 | 0.500 | Control: no soft attenuation; same WER as raw. |
| `data/m2_baseline/babble_ba_repeat_tight_seed0_in3_out5_noise4_gain0p25_20260429_090959` | same strong overlap | 0.25 | 0.2888 | 100.0% | 0 / 29 / 131 | 0.625 | 1.125 | Too aggressive: world speech preservation degrades. |

Current conclusion:
- The M2 harness is now producing saved, reference-scored artifacts.
- Conservative `BOTH` preservation works as expected for the light baseline.
- In stronger overlap, `both_gain=0.5` does not make WER worse, but does not improve ASR either.
- `both_gain=0.25` is too destructive on the current hardware path.
- Current evidence still shows zero hard `SELF` frames; near-term M2 work should improve frame attribution / overlap calibration before claiming self-suppression performance.

Next tests:
1. run a small seed matrix at `both_gain=0.5` for light and strong overlap
2. add a no-self world-speech control for the same device geometry
3. add a self-only control to verify expected `SELF` or `BOTH` attribution behavior without world speech
4. only then tune soft-gate policy beyond fixed gain
