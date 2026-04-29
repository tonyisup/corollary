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

### 2026-04-29 Control / Calibration Follow-up

Additional harness controls were added:
- `--self-only` and `--world-only` isolate the two audio sources.
- `--speech-override {vad,true,false}` separates VAD conservatism from acoustic self-matching.
- `--similarity-threshold` allows controlled threshold sweeps.
- `--onset-latency-ms` compensates physical playback/capture latency.
- summaries now include onset diagnostics (`self_shift_samples`, `self_expected_shift_samples`, `self_onset_error_ms`).

Control findings:
- World-only control behaved as expected: no self match (`self_score=0.0`) and world speech remained recoverable.
- Self-only with normal VAD still produced `BOTH`, because the babble stimulus itself triggers speech detection; this is expected under the M2 conservative policy.
- Self-only with `--speech-override false` revealed the second bottleneck: physical onset timing. Without compensation, the attributor sees the self event as too late for the 25 ms causal window.
- With `--onset-latency-ms 400`, `--similarity-threshold 0.20`, and `--speech-override false`, self-only attribution produced hard `SELF` frames (`30 / 0 / 130` SELF/BOTH/WORLD), proving the path can still produce SELF when VAD conservatism and hardware latency are controlled.
- Mixed speech with the same latency/threshold and normal VAD produced `0 / 30 / 130` SELF/BOTH/WORLD, which is the desired conservative M2 behavior: self-like motor-window audio plus detected speech is preserved as `BOTH` rather than dropped as `SELF`.

Updated interpretation:
- The earlier `SELF=0` result was not a simple failure. It was caused by two interacting factors:
  1. babble is speech-like to VAD, so M2 intentionally avoids hard SELF during speech;
  2. the hardware path needs explicit onset-latency calibration before M1-style causal attribution is meaningful.
- M2 should therefore split calibration into two passes: first estimate hardware onset latency with self-only / speech-disabled controls, then run speech-preservation trials with VAD enabled and that latency compensation applied.

### 2026-04-29 Calibrated Seed Matrix

A tighter calibrated protocol was tested with:
- stimulus: `babble_ba_repeat_short_tight_var`
- seeds: `0,1,2`
- self duration/amplitude: `900 ms`, `1.0`
- similarity threshold: `0.20`
- onset latency compensation: `400 ms`
- onset method: forward energy onset (`--onset-method energy`, searching from the compensated expected onset)
- causal tolerance: `75 ms`
- mixed-trial `BOTH` gain: `0.5`

Results:

| Trial | SELF/BOTH/WORLD | Raw WER | Gated WER | Interpretation |
| --- | ---: | ---: | ---: | --- |
| self-only seed 0 | 30 / 0 / 130 | n/a | n/a | hard SELF recovered |
| self-only seed 1 | 28 / 0 / 132 | n/a | n/a | hard SELF recovered |
| self-only seed 2 | 30 / 0 / 130 | n/a | n/a | hard SELF recovered |
| mixed seed 0 | 0 / 30 / 130 | 0.375 | 0.375 | preserved, no WER gain |
| mixed seed 1 | 0 / 28 / 132 | 0.375 | 0.750 | regression; gated transcript leaked babble-like tokens |
| mixed seed 2 | 0 / 30 / 130 | 0.500 | 0.375 | improved by removing babble-like tokens |

Conclusion:
- The calibrated self-only path now consistently recovers hard `SELF` attribution across seeds `0,1,2`.
- The mixed path consistently routes motor-window speech-like overlap to `BOTH`, preserving the conservative M2 rule.
- Fixed `BOTH` gain is not yet robust: seed 2 improved, seed 0 was neutral, seed 1 regressed.
- Next implementation should stop treating `BOTH` as one global gain. It needs a more selective soft gate, likely using the self template/efference estimate inside `BOTH` windows rather than attenuating the whole frame.

### 2026-04-29 BOTH Template-Subtraction Prototype

Implementation update:
- Added `--both-mode {gain,subtract}` to `scripts/m2_speech_preservation_harness.py`.
- Added `--both-subtract-gain` for aligned self-template subtraction inside `BOTH` windows.
- The harness now saves `processed.npy` alongside `observed.npy`, `gating_mask.npy`, and `gating_gain.npy`.

Prototype matrix:
- calibrated settings from the prior matrix: `--onset-method energy`, `--onset-latency-ms 400`, `--causal-tolerance-ms 75`, `--similarity-threshold 0.20`
- stimulus: `babble_ba_repeat_short_tight_var`, seeds `0,1,2`
- save dir: `data/m2_baseline/m2_subtract_matrix_20260429_093920`

| Subtract gain | Seed deltas (gated WER - raw WER) | Mean delta | Interpretation |
| ---: | --- | ---: | --- |
| 0.25 | `0.000`, `0.000`, `-0.125` | `-0.042` | safest tested subtraction; neutral/slightly helpful |
| 0.50 | `+0.250`, `+0.250`, `0.000` | `+0.167` | too destructive / unstable |
| 0.75 | `0.000`, `0.000`, `+0.750` | `+0.250` | clearly too aggressive |

Updated conclusion:
- Template subtraction is directionally better than blunt global gain only at very conservative strength (`0.25`).
- Stronger template subtraction creates ASR artifacts and can damage world speech.
- The current physical setup has high lexical variability across repeated captures, so promotion should require paired controls and repeated runs rather than single-run WER deltas.
- Near-term default for continued M2 work: `both-mode=subtract`, `both-subtract-gain=0.25`, calibrated energy onset, and explicit no-processing / gain controls in the same run batch.
