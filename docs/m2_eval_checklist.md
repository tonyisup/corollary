# M2 Evaluation Checklist

M2 is a measurement-first bridge from speech-aware attribution to soft gating. The first pass should establish saved evidence before claiming improved self suppression.

## Baseline Run

1. Confirm VAD sees speech on the intended input device:

   ```bash
   ./.venv/bin/python scripts/m2_speech_detection_test.py --device 2 --duration 5
   ```

2. Run the current speech-preservation harness and save artifacts:

   ```bash
   ./.venv/bin/python scripts/m2_speech_preservation_harness.py \
     --input-device 2 \
     --output-device 3 \
     --noise-device 3 \
     --self-stim babble_ba_var \
     --save-dir data/m2_baseline/babble_ba_var_seed0
   ```

## Success Criteria

For M2, the first success condition is speech preservation under overlap, not maximum self suppression.

- Speech frames are detected during the overlap window.
- Speech overlap is attributed as `BOTH`, not hard `SELF`.
- Gated audio does not destroy world speech.
- Raw vs gated transcript is equal or better when a reference transcript is available.
- Artifacts are saved so future changes can be compared against the same run.

## Required Artifacts

Each saved run should include:

- `observed.npy`: captured 16 kHz audio.
- `gating_mask.npy`: legacy hard keep/drop mask for compatibility.
- `gating_gain.npy`: soft sample-level gain applied before ASR.
- `summary.json`: transcripts, frame counts, speech ratio, self-match score, and any WER metrics.

## Next Implementation Step

After one baseline is saved, move from binary keep/drop to soft treatment of `BOTH` frames. The first implementation should be conservative: keep `WORLD`, drop hard `SELF`, and attenuate `BOTH` rather than zeroing it. Spectral subtraction can replace the attenuation step once there is a stable predicted self-signal path for the overlap frames.
