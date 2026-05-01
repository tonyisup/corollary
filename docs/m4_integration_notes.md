# M4 Integration — Implementation Notes & Findings

> Date: 2026-05-01
> Status: Partially complete — architecture wired, blocking issue identified

## What Was Done

1. **Replaced mock SpeakerIdentifier with real ECAPA-TDNN** (SpeechBrain, 192-dim embeddings)
   - Confirmed self/world discrimination works: AriaNeural vs GuyNeural similarity ~0.07-0.11
   - Same speaker similarity: ~0.70

2. **Integrated M4 into ThalamicAttributor** via `m4_detector` parameter
   - Added sliding window buffer (1s) since ECAPA needs ~500ms+ for reliable classification
   - M4 override: when `is_playing AND is_self_like AND is_timely AND is_speech`,
     M4 checks if the accumulated audio matches the self template
   - If M4 confirms self (similarity > threshold), routes to SELF instead of conservative BOTH

3. **All 180 unit tests pass** — backward compatible (m4_detector=None falls through to M2 conservative)

## Blocking Issue: Room-Colored Self Template

**Problem:** M4 registration uses the dry self PCM (the generated stimulus), but the
captured audio at the mic contains the room-colored version. The ECAPA embedding
of the dry template doesn't match the embedding of the captured self signal.

Evidence:
- TTS registration vs actual stimulus PCM similarity: 0.036 (essentially uncorrelated)
- M4 never triggers on captured audio because the buffer contains mixed self+world
  speech, and the self component doesn't match the dry template

**Root cause:** The biological system solves this by transforming the efference copy
through the cerebellar forward model *before* comparing to sensory input. The forward
model predicts what the self signal will sound like at the mic (including room coloration).

**Solution required:** Train the MultiBandBabblingCerebellum on the current room
to learn the room impulse response, then use `cerebellum.predict_pcm(self_pcm)` to
generate the room-colored self template for M4 registration.

## Path Forward

### Immediate (blocked by forward model training):
1. Run babbling adaptation in the current room (the existing M1b babbling loop)
2. Save the trained cerebellum model
3. Generate room-colored self template: `predicted_self = cerebellum.predict_pcm(self_dry_pcm)`
4. Register M4 with the room-colored template
5. Re-run M4-enabled benchmark

### Alternative (doesn't require forward model):
- Skip M4 for now, focus on **room-colored spectral subtraction**
- Use the learned room IR to predict the self component in BOTH frames
- Subtract the predicted self instead of the dry template

## Architecture Changes Made

### `packages/stg/speaker.py`
- `SpeakerIdentifier.get_embedding()`: now uses SpeechBrain ECAPA-TDNN (real)
- Falls back to mock if SpeechBrain not available
- `SelfVoiceDetector`: unchanged interface, now uses real embeddings

### `packages/thalamus/attributor.py`
- `ThalamicAttributor.__init__()`: added `m4_detector` and `m4_self_threshold` params
- `ThalamicAttributor.attribute()`: added `frame_audio` param, M4 override logic
- M4 buffer: accumulates last 1s of frame_audio, runs ECAPA when buffer >= 500ms
- Backward compatible: all existing callers work without changes

### `scripts/m2_benchmark.py`
- `process_trial()`: passes `frame_audio` to `attributor.attribute()`

### `scripts/m2_speech_preservation_harness.py`
- Same frame_audio pass-through
