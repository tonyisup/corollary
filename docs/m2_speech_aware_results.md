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

### 2026-04-29 Paired Eval Runner

Added `scripts/m2_eval_runner.py` to run paired M2 strategy evaluations with shared seeds and calibrated settings. The runner currently compares:
- `gain_1p0`: no attenuation / no suppression control
- `gain_0p5`: blunt half-gain on `BOTH` frames
- `subtract_0p25`: conservative aligned template subtraction on `BOTH` frames

Run:
```bash
./.venv/bin/python scripts/m2_eval_runner.py --seeds 0,1,2
```

Output:
- `data/m2_baseline/m2_paired_eval_20260429_094643/aggregate_summary.json`

Aggregate result:

| Strategy | Trials | Mean raw WER | Mean gated WER | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gain_1p0` | 3 | 0.833 | 0.958 | +0.125 | 0 / 1 / 2 |
| `gain_0p5` | 3 | 1.000 | 0.958 | -0.042 | 1 / 1 / 1 |
| `subtract_0p25` | 3 | 0.625 | 0.708 | +0.083 | 0 / 2 / 1 |

Interpretation:
- The paired runner is now the preferred M2 measurement tool because it keeps seeds and calibration consistent across strategies.
- Results remain noisy because each strategy is still physically recaptured rather than replayed from one identical captured mixture.
- In this paired run, blunt `gain_0p5` had the best mean ΔWER, but its win was unstable and driven by one large improvement.
- `subtract_0p25` was mostly neutral but regressed on one seed.
- Next refinement should separate acquisition from processing: capture one raw mixed trial per seed, then replay multiple processing strategies offline against the identical `observed.npy` so strategy comparisons are no longer confounded by live capture variability.

### 2026-04-29 Offline Strategy Evaluation

Added `scripts/m2_offline_eval.py` to evaluate multiple M2 processing strategies against the same saved `observed.npy` captures. This separates live acquisition variability from processing-strategy effects.

Run input:
- source captures: `data/m2_baseline/m2_paired_eval_20260429_094643/gain_1p0/seed{0,1,2}`
- output: `data/m2_baseline/m2_offline_eval_20260429_095203/offline_summary.json`

Offline aggregate:

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 3 | `0.000` | 0 / 3 / 0 |
| `gain_0p5` | 3 | `-0.083` | 1 / 2 / 0 |
| `subtract_0p25` | 3 | `+0.208` | 0 / 2 / 1 |

Offline conclusion:
- Against identical captures, blunt `gain_0p5` is currently safer than template subtraction.
- `subtract_0p25` regressed on seed 0, so the simple aligned-template subtraction is not yet promotion-worthy.
- This likely means the current self template is too acoustically mismatched to the captured room-transformed self component; subtracting it can distort world speech instead of cleanly removing self.
- Next implementation target should be room-aware/efference-aware subtraction: estimate a scale + short impulse response or use the learned forward model, rather than subtracting the dry generated template.

### 2026-04-29 Room-Aware Offline Subtraction Prototype

Extended `scripts/m2_offline_eval.py` with `room_subtract` strategies. This estimates a small ridge-regularized FIR room transform from the dry self template to the captured mixture inside `BOTH` windows, then subtracts a conservative room-shaped self estimate.

Run:
- output: `data/m2_baseline/m2_offline_room_eval_20260429_100212/offline_summary.json`
- source captures: same `gain_1p0` seed captures from `m2_paired_eval_20260429_094643`

Aggregate result:

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 3 | `0.000` | 0 / 3 / 0 |
| `gain_0p5` | 3 | `-0.083` | 1 / 2 / 0 |
| `subtract_0p25` | 3 | `+0.042` | 0 / 2 / 1 |
| `room_subtract_0p1_t64` | 3 | `+0.042` | 1 / 1 / 1 |
| `room_subtract_0p25_t64` | 3 | `-0.083` | 1 / 2 / 0 |

Interpretation:
- Room-aware subtraction at `0.25` now matches blunt `gain_0p5` on this offline batch.
- That is a meaningful improvement over dry-template subtraction, which regressed on one seed.
- It is not yet a promotion over `gain_0p5`; both are tied at mean ΔWER `-0.083` with 1 improved / 2 neutral / 0 regressed.
- The next useful test is a larger offline-only seed/capture set and/or lower-variance world speech before deciding whether room-aware subtraction should replace the simpler gain baseline.

### 2026-04-29 Six-Capture Offline Expansion

Expanded the offline evaluation set with three additional live captures:
- new captures: `data/m2_baseline/m2_capture_expansion_20260429_101014/gain_1p0/seed{3,4,5}`
- combined with earlier captures: `data/m2_baseline/m2_paired_eval_20260429_094643/gain_1p0/seed{0,1,2}`
- offline output: `data/m2_baseline/m2_offline_room_eval_6cap_20260429_101059/offline_summary.json`

Six-capture aggregate:

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 6 | `-0.042` | 2 / 4 / 0 |
| `gain_0p5` | 6 | `-0.021` | 1 / 3 / 2 |
| `subtract_0p25` | 6 | `+0.021` | 0 / 5 / 1 |
| `room_subtract_0p1_t64` | 6 | `+0.146` | 1 / 1 / 4 |
| `room_subtract_0p25_t64` | 6 | `-0.021` | 2 / 3 / 1 |

Updated interpretation:
- The larger offline set weakens the case for any active M2 suppression strategy on this hardware/world-speech setup.
- `gain_1p0` (effectively preserve `BOTH`) is the safest current baseline: it had no regressions and the best mean ΔWER in this six-capture comparison.
- `room_subtract_0p25_t64` remains interesting but is not safer than preservation; it improved two captures but regressed one.
- M2's current validated result should therefore be framed as **speech-preserving conservative attribution**, not reliable speech-enhancing suppression.
- Next work should improve the benchmark itself before more suppression tuning: cleaner fixed world-speech playback, stronger saved reference controls, and a capture-once/process-offline workflow as the default.

### 2026-04-29 M2 BOTH-Policy Risk Gate Slice

Added the first bounded ML-adjacent policy layer:
- `packages/m2_policy/dataset.py` loads offline M2 rows and extracts compact trial/strategy features.
- `packages/m2_policy/risk_gate.py` implements a deliberately tiny conservative baseline gate.
- `scripts/m2_policy_eval.py` evaluates policy selection against the current fallback, `gain_1p0` / preserve `BOTH`.

This is not yet a neural model. It is the interpretable baseline a learned model must beat.

Run:
```bash
./.venv/bin/python scripts/m2_policy_eval.py \
  data/m2_baseline/m2_offline_room_eval_6cap_20260429_101059/offline_summary.json
```

Result on the six-capture offline set:
- full-data best mean strategy remains `gain_1p0` at mean ΔWER `-0.0417`.
- leave-one-seed-out policy selection chose `room_subtract_0p25_t64` for seed 5 based on training means, but that regressed badly on held-out seed 5 (`-0.125` fallback vs `+0.375` chosen).
- leave-one-seed-out mean fallback ΔWER: `-0.0417`.
- leave-one-seed-out mean chosen ΔWER: `+0.0417`.
- regressions vs fallback: `1`.

Conclusion:
- This is exactly why the ML pivot should start with a risk gate: naive strategy selection overfits immediately on the tiny dataset.
- No learned/policy strategy should be promoted yet.
- The current safe operational policy remains preserve `BOTH` (`gain_1p0`).
- Next ML slice should add more data and evaluate a stricter gate that only allows active suppression when it can beat preserve-`BOTH` with zero held-out regressions.

### 2026-04-29 M2 Dataset Collector Pilot

Added `scripts/m2_collect_dataset.py`, which implements the capture-once/process-offline workflow:
1. capture preserve-`BOTH` trials (`gain_1p0`)
2. run `scripts/m2_offline_eval.py` over those identical captures
3. export canonical policy rows to `m2_policy_dataset.jsonl`

Pilot run:
```bash
./.venv/bin/python scripts/m2_collect_dataset.py --seeds 6-9
```

Output:
- root: `data/m2_policy/collection_20260429_105748`
- dataset: `data/m2_policy/collection_20260429_105748/m2_policy_dataset.jsonl`
- offline summary: `data/m2_policy/collection_20260429_105748/offline_eval/offline_summary.json`

Pilot aggregate (`seeds 6-9`):

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 4 | `0.000` | 0 / 4 / 0 |
| `gain_0p5` | 4 | `-0.031` | 1 / 3 / 0 |
| `subtract_0p25` | 4 | `0.000` | 0 / 4 / 0 |
| `room_subtract_0p1_t64` | 4 | `+0.031` | 0 / 3 / 1 |
| `room_subtract_0p25_t64` | 4 | `-0.031` | 1 / 3 / 0 |

Combined policy evaluation over seeds `0-9` (`50` strategy rows):
- `gain_1p0`, `gain_0p5`, and `room_subtract_0p25_t64` all tie at mean ΔWER `-0.025` on full data.
- leave-one-seed-out still fails promotion: chosen mean ΔWER `+0.0375` vs fallback `-0.025`, with `2` regressions vs preserve-`BOTH`.

Conclusion:
- More data preserved the central rule: active suppression sometimes helps, but naive policy selection still regresses on held-out seeds.
- The collector is now the right path to scale data safely.
- Preserve-`BOTH` remains the operational baseline until a gate can beat it with zero held-out regressions.

### 2026-04-29 Seeds 10-19 Collection and Combined 0-19 Policy Evaluation

Collected a bounded next batch with:
```bash
./.venv/bin/python scripts/m2_collect_dataset.py --seeds 10-19
```

Output:
- root: `data/m2_policy/collection_20260429_113544`
- dataset: `data/m2_policy/collection_20260429_113544/m2_policy_dataset.jsonl`
- offline summary: `data/m2_policy/collection_20260429_113544/offline_eval/offline_summary.json`

Seeds `10-19` aggregate:

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 10 | `+0.250` | 0 / 9 / 1 |
| `gain_0p5` | 10 | `+0.100` | 3 / 5 / 2 |
| `subtract_0p25` | 10 | `-0.013` | 1 / 9 / 0 |
| `room_subtract_0p1_t64` | 10 | `0.000` | 3 / 4 / 3 |
| `room_subtract_0p25_t64` | 10 | `+0.013` | 3 / 4 / 3 |

Notable issue:
- seed `16` produced a severe `gain_1p0` offline ASR regression (`+2.500` ΔWER) despite preserve-`BOTH`; this shows the current ASR-scored benchmark itself has stochastic / decoding instability, not just policy instability.

Combined policy evaluation over seeds `0-19` (`100` strategy rows):

| Strategy | Mean ΔWER |
| --- | ---: |
| `gain_1p0` | `+0.1125` |
| `gain_0p5` | `+0.0375` |
| `subtract_0p25` | `0.0000` |
| `room_subtract_0p1_t64` | `+0.0500` |
| `room_subtract_0p25_t64` | `-0.00625` |

Leave-one-seed-out policy result:
- fallback preserve-`BOTH` mean ΔWER: `+0.1125`
- chosen policy mean ΔWER: `+0.2125`
- regressions vs fallback: `4`

Updated conclusion:
- More data did not make the current gate promotable; it made the benchmark weakness clearer.
- The current WER scoring path is too noisy for fine policy selection, because even preserve-`BOTH` can show large offline ASR swings.
- Next step should not be blindly collecting more babble seeds. It should add benchmark controls: repeat offline ASR on the same `observed.npy` to measure decoder variance, and add a cleaner stimulus/control condition such as `vowel_ah_var_nearfield`.

### 2026-04-29 ASR Repeatability and Vowel Control Pilot

Added `scripts/m2_asr_repeatability.py` to measure Whisper repeatability on identical saved audio.

Repeatability checks:
- seed `16` observed audio (`data/m2_policy/collection_20260429_113544/captures/gain_1p0/seed16/observed.npy`): 8 repeats, `3` unique transcripts, WER range `0.375-0.500`.
- seed `10` observed audio: 5 repeats, `1` unique transcript, WER range `0.375-0.375`.

Interpretation:
- ASR repeatability is not perfectly deterministic, but the extreme seed-16 offline outlier (`+2.5` ΔWER) was not reproduced by repeated direct transcription of the same observed audio or all-ones masked audio.
- The benchmark issue appears to be a combination of ASR variability plus processing/evaluation path sensitivity, not pure random decoder variance alone.

Vowel control pilot:
```bash
./.venv/bin/python scripts/m2_collect_dataset.py \
  --seeds 0-3 \
  --self-stim vowel_ah_var_nearfield \
  --output-root data/m2_policy/vowel_control_20260429_115524
```

Aggregate:

| Strategy | Trials | Mean ΔWER | Improved / Neutral / Regressed |
| --- | ---: | ---: | ---: |
| `gain_1p0` | 4 | `0.000` | 0 / 4 / 0 |
| `gain_0p5` | 4 | `-0.3125` | 2 / 2 / 0 |
| `subtract_0p25` | 4 | `-0.1250` | 1 / 3 / 0 |
| `room_subtract_0p1_t64` | 4 | `-0.3125` | 2 / 2 / 0 |
| `room_subtract_0p25_t64` | 4 | `-0.3125` | 2 / 2 / 0 |

Updated direction:
- The vowel-control pilot is much more favorable to active suppression than the repeated-babble set.
- This supports the hypothesis that repeated babble is a harder / more ASR-hostile stress condition, while `vowel_ah_var_nearfield` may be a cleaner control for learning the M2 BOTH-policy gate.
- Next: expand vowel-control seeds before further babble scaling, then train/evaluate policy separately by stimulus family.

### 2026-04-29 Expanded Vowel-Control Evaluation

Expanded `vowel_ah_var_nearfield` control with:
```bash
./.venv/bin/python scripts/m2_collect_dataset.py \
  --seeds 4-9 \
  --self-stim vowel_ah_var_nearfield \
  --output-root data/m2_policy/vowel_control2_20260429_120112
```

Seeds `4-9` aggregate:
- all strategies were neutral: mean ΔWER `0.000`, 0 improved / 6 neutral / 0 regressed.

Combined vowel seeds `0-9` policy evaluation (`50` strategy rows):

| Strategy | Mean ΔWER |
| --- | ---: |
| `gain_1p0` | `0.000` |
| `gain_0p5` | `-0.125` |
| `subtract_0p25` | `-0.050` |
| `room_subtract_0p1_t64` | `-0.125` |
| `room_subtract_0p25_t64` | `-0.125` |

Leave-one-seed-out vowel policy:
- selected `gain_0p5` for all held-out seeds
- fallback preserve-`BOTH` mean ΔWER: `0.000`
- chosen mean ΔWER: `-0.125`
- regressions vs fallback: `0`

Interpretation:
- For `vowel_ah_var_nearfield`, active suppression is now promotion-worthy under the current conservative criterion.
- The simplest promoted vowel-family policy is `gain_0p5`: it matches the best mean ΔWER and has zero held-out regressions.
- This should be scoped narrowly: promote `gain_0p5` for vowel-family M2 controls, while repeated-babble remains preserve-`BOTH` / stress-test only.

### 2026-04-29 Family-Aware BOTH Policy Implementation

Implemented the first deterministic M2 BOTH-frame policy:
- `packages/m2_policy/policy.py`
- harness flag: `--both-policy family`

Policy:
- `vowel_ah_var_nearfield*` → `mode=gain`, `gain=0.5`
- repeated babble (`babble_ba_repeat*`) → preserve `BOTH` with `gain=1.0`
- unknown / other stimuli → preserve `BOTH` with `gain=1.0`

Rationale:
- Vowel-family seeds `0-9` met the conservative promotion criterion for `gain_0p5`: mean ΔWER improved vs preserve and zero held-out regressions.
- Repeated-babble stress captures did not meet promotion criteria; active suppression caused held-out regressions.

Validation:
- Added `tests/unit/test_m2_policy_family.py`.
- Policy selector sanity check:
  - `vowel_ah_var_nearfield` → `BothPolicyDecision(mode='gain', gain=0.5, reason='vowel_ah_var_nearfield_promoted_gain_0p5')`
  - `babble_ba_repeat_short_tight_var` → `BothPolicyDecision(mode='gain', gain=1.0, reason='repeated_babble_preserve_both')`
  - `chirp` → `BothPolicyDecision(mode='gain', gain=1.0, reason='unknown_preserve_both')`

### 2026-04-29 M3 Bridge Started

M2 now feeds the first M3 proto-turn-taking slice:
- speech-like `BOTH` frames are treated as agentive external speech for turn-taking
- the PFC controller defers output while listening and resumes after silence/cooldown

See `docs/m3_start.md` and `scripts/m3_turn_taking_demo.py`.
