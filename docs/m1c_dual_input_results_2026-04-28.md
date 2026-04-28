# M1c Dual-Input Hardware Validation Results

> **Date:** 2026-04-28  
> **Status:** Validated on current Mac mini + Fifine + BL-2021 geometry  
> **Scope:** Mixed-source attribution: SELF vs WORLD vs BOTH under dual-input hardware capture

## Summary

M1c crossed from single-input mixed-source attribution into a dual-input evidence-fusion result. The key finding is that the previously observed failures were not primarily caused by bad geometry or microphone gain. They came from asking a single microphone path to solve a two-source attribution problem.

After locking the physical geometry with the live calibration GUI and adding a dual-input validation harness, the current setup passed the full seed `0,1,2` matrix:

| Case | Result |
|---|---:|
| `self_only` | 3/3 `SELF` |
| `world_only` | 3/3 `WORLD` |
| `replay_control` | 3/3 `WORLD` |
| `sequential_world_first` | 3/3 pass, `BOTH` |
| `sequential_self_first` | 3/3 pass, `BOTH` |
| `partial_overlap_self_first` | 3/3 `BOTH` |
| `partial_overlap_world_first` | 3/3 `BOTH` |
| `full_overlap` | 3/3 `BOTH` |

The final validation run is stored at:

```text
data/m1c_dual_input/m1c_dual_input_world_aim_adjusted_012_v2.jsonl
```

## Hardware setup

Current calibrated geometry:

- Input device `3`: Fifine microphone, treated as the self-favored channel.
- Input device `1`: BL-2021USB microphone, treated as the world-favored channel.
- Output device `5`: Mac mini Speakers, treated as the self output.
- Output device `4`: External speakers/headphones, treated as the world output.
- Self stimulus: `babble_ba_var`.
- World stimulus: `babble_ma_var`.

The final speaker-direction adjustment improved world audibility without requiring Fifine gain changes.

## What failed before dual input

The single-input extended validation harness passed world-only and replay-control safety, but failed some world-first and overlap cases because it listened only through the Fifine path. In those failures, the system could hear self strongly but did not reliably see enough world evidence on that same channel.

This made the single-input harness over-report `SELF` in cases where the physical setup actually contained mixed self+world audio.

## Dual-input evidence fusion

The dual-input harness captures both microphones simultaneously and computes per-channel evidence:

- self-template match score
- world-template match score
- self-window RMS evidence over baseline
- world-window RMS evidence over baseline

Fusion uses geometry-aware assumptions:

1. Channel 0 is self-favored.
2. Channel 1 is world-favored.
3. Self presence requires self-template match plus self energy on the self-favored channel.
4. World presence can be established by:
   - sufficient world RMS evidence on either channel, or
   - strong world-template match on the world-favored channel even when its RMS baseline is high.
5. Replay control suppresses self presence because there is no motor command.

The reusable implementation now lives in:

```text
packages/thalamus/dual_input_fusion.py
```

The hardware runner remains:

```text
scripts/m1c_dual_input_validation.py
```

## Interpretation

The important result is not merely that a script passed. The stronger conclusion is:

> M1c mixed-source attribution needs geometry-aware multi-input evidence. With two calibrated input paths, Corollary can distinguish self-only, world-only, replayed self-like world audio, and self+world overlap under the current babble-class hardware setup.

This supports treating M1c as hardware-validated for the current developmental milestone, with the caveat that the validation is still tied to the current two-mic geometry and babble-class stimuli.

## Caveats

- The current verdict for sequential cases is a window-level summary, so sequential world-first and self-first often resolve to `BOTH` because both source events are present in the total captured trial window.
- `babble_ba_var` / `babble_ma_var` are still synthetic developmental stimuli, not external human speech.
- The BL-2021 path can have a high baseline, so world detection uses strong world-template match as a first-class cue rather than RMS alone.
- This should be rerun if the physical geometry changes materially.

## Next steps

1. Treat this as the M1c dual-input baseline.
2. Run one broader robustness pass, e.g. seeds `0..9`, if promotion-grade confidence is needed.
3. Transition to M2 by replacing the world babble stimulus with speech-bearing audio and verifying that dual-input fusion preserves external speech during self babble.
4. Start moving from trial-level verdicts toward frame/segment-level verdict streams so sequential cases can report ordered events rather than a single whole-window verdict.
