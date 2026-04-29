# M2 BOTH-Frame Policy

> Status: first promoted deterministic policy slice  
> Date: 2026-04-29

## Summary

M2 handles speech-aware self/world separation after M1c attribution. The current
safe rule is conservative: when self-like audio overlaps detected speech, route
the frame to `BOTH` rather than hard `SELF`, so world speech is preserved.

The first promoted BOTH-frame policy is stimulus-family aware:

| Stimulus family | BOTH policy | Rationale |
| --- | --- | --- |
| `vowel_ah_var_nearfield*` | `gain_0p5` | Vowel seeds `0-9` improved mean WER with zero held-out regressions. |
| `babble_ba_repeat*` | preserve `BOTH` / `gain_1p0` | Repeated babble remains ASR-hostile; active suppression regressed on held-out seeds. |
| unknown / other | preserve `BOTH` / `gain_1p0` | Safe default until family-specific evidence exists. |

Implementation:

- `packages/m2_policy/policy.py`
- `scripts/m2_speech_preservation_harness.py --both-policy family`

## Calibrated M2 settings

The hardware path used for this policy work:

- input device `3`: Fifine mic
- self output `5`: Mac mini Speakers
- world/noise output `4`: External Headphones

Useful calibrated attribution settings:

```bash
--similarity-threshold 0.20 \
--onset-latency-ms 400 \
--onset-method energy \
--causal-tolerance-ms 75
```

These settings recovered hard `SELF` on self-only repeated-babble controls while
still routing mixed speech/self windows to conservative `BOTH`.

### Optimized M2 settings (Bayesian search, 2026-04-29)

A 30-trial Bayesian optimization via Optuna TPE confirmed and refined these
values. Trial 14 achieved loss 0.243 vs random-start baseline of 0.778.

```bash
--similarity-threshold 0.23 \
--onset-latency-ms 405 \
--onset-method energy \
--causal-tolerance-ms 74
```

Compared to manual calibration:
- `similarity_threshold`: 0.23 vs 0.20 (looser matching preferred)
- `onset_latency_ms`: 405 vs 400 (essentially unchanged)
- `causal_tolerance_ms`: 74 vs 75 (essentially unchanged)
- `onset_method`: energy confirmed

The optimizer also found adjusted gains: `vowel_gain 0.615` (vs 0.5) and
`babble_gain 0.928` (vs 1.0). These suggest slightly less aggressive vowel
suppression and a small babble attenuation benefit.

## Promotion criterion

A strategy is only promotable for a stimulus family if it beats preserve-`BOTH`
without held-out regressions.

For the current vowel-family run, leave-one-seed-out selected `gain_0p5` for
all vowel seeds `0-9`:

- fallback preserve-`BOTH` mean ΔWER: `0.000`
- chosen `gain_0p5` mean ΔWER: `-0.125`
- regressions vs fallback: `0`

For repeated babble, the policy gate regressed on held-out seeds. Preserve
`BOTH` remains the baseline.

## Reproduction commands

Collect vowel-control data:

```bash
./.venv/bin/python scripts/m2_collect_dataset.py \
  --seeds 0-9 \
  --self-stim vowel_ah_var_nearfield \
  --output-root data/m2_policy/vowel_control_repro
```

Evaluate policy rows:

```bash
./.venv/bin/python scripts/m2_policy_eval.py \
  data/m2_policy/vowel_control_20260429_115524/offline_eval/offline_summary.json \
  data/m2_policy/vowel_control2_20260429_120112/offline_eval/offline_summary.json
```

Run the family policy live:

```bash
./.venv/bin/python scripts/m2_speech_preservation_harness.py \
  --input-device 3 \
  --output-device 5 \
  --noise-device 4 \
  --self-stim vowel_ah_var_nearfield \
  --self-duration-ms 900 \
  --self-amplitude 1.0 \
  --both-policy family \
  --seed 0 \
  --speech-override vad \
  --similarity-threshold 0.20 \
  --onset-latency-ms 400 \
  --onset-method energy \
  --causal-tolerance-ms 75 \
  --expected-text "this is a test of world speech conservation"
```

## Current limitation

The WER/Whisper benchmark is useful but noisy. `scripts/m2_asr_repeatability.py`
exists to measure repeat transcription variance on saved artifacts. Policy
promotion should continue to use held-out regressions, not single-run wins.
