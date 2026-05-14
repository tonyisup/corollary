# M4 Live Hardware Validation — 2026-05-13

> **Date:** 2026-05-13
> **Status:** Validated with known limitations

## Initial Validation Result

Ran `scripts/m4_live_adaptive_threshold.py` on BL-2021USB (in=1) + Mac mini Speakers (out=5).

- Audio I/O working: 240 frames, TTS played and captured (mean energy -41.6 dB)
- M4 attempted on 166/240 frames (not buffer-blocked)
- M4 accepted: 0 frames at base threshold 0.10
- ECAPA confidence on live audio: mean -0.004, max 0.217 (all BOTH verdict)
- Self-match scores: mean 0.002, max 0.200 — room filter not matching well
- Speaker identification: no IDs returned (no pre-registered speakers in world model)
- Adapter: incorrectly ratcheted a synthetic `unknown` profile upward, self-poisoning the live fallback path

## Root-Cause Fix

`ThalamicAttributor.attribute()` now only engages per-speaker adaptive thresholds for real identified speakers. When `speaker_id` is absent, the live path stays on the base threshold instead of collapsing room-colored self playback into an adaptive `unknown` profile.

## Post-Fix Live Reruns at θ=0.10

- `m4_accepted` improved from `0` to `12-18` frames across reruns
- Final verdicts still remain mostly `BOTH`, with only `1` final `SELF` verdict observed
- Speaker identification still returns no IDs because the world-model registry is empty
- The live harness now correctly treats "no unknown-profile ratcheting" as pass behavior when no speaker IDs are present

## Speaker Pre-Registration Workflow

- `scripts/m4_live_adaptive_threshold.py` now supports `--register-speaker`, `--register-speaker-dir`, `--play-world-audio`, and `--speaker-registry-threshold`
- BL-2021USB human-negative clips can be embedded and loaded into the live `SpeakerRegistry` before the run
- Concurrent world-audio playback can inject a known negative clip during the live run

## Pre-Registration Result

- Registry successfully populated with 6 BL-2021USB human-negative speakers
- Concurrent playback of a matching registered clip works during the live run
- Two speaker-ID bugs fixed: (1) silent call to non-existent `MultiSpeakerWorldModel.identify_speaker()`, and (2) WORLD/BOTH buffer accumulation only happened on the periodic speaker-ID cadence
- After fixes, live `speaker_id` resolves and best-match diagnostics are populated
- In mixed self+world run, IDs resolved mainly to `human_negative_speaker4` / `human_negative_speaker5` with best-match similarities up to `0.196`

## Embedding-Domain Analysis

File-based reference embeddings for several speakers don't match current live BL-2021USB captures. The fix: extract embeddings from the live path itself (play clip through device 4, capture on device 1, extract 2s window embedding). This improved mixed-condition speaker-ID from avg sim 0.055 to 0.45-0.74.

Within-clip embedding variance is huge — different 2s windows within the same clip can have cosine similarity as low as 0.01. Single full-clip embeddings are not robust reference points.

## Full 6-Speaker Live-Path Validation

All 6 reference speakers tested as world-playback clips using live-path single-window embeddings. All 6 resolved to identifiable speaker IDs. Speakers 2 and 6 have the strongest matches (avg sim 0.33-0.35). Speakers 1, 3, 4, 5 have lower similarity (0.05-0.12) due to reference/live embedding mismatch.

## Centroid Window Embeddings

Tested but does NOT outperform single-window live-path embeddings. Centroid can be less distinctive. Recommendation: stick with single-window live-path embeddings.

## Cross-Speaker Generalization

Pairwise embedding analysis: speaker1↔speaker5 sim=0.65, speaker3↔speaker4 sim=0.54, speaker3↔speaker6 sim=0.59. Cross-speaker confusion expected when vocal characteristics overlap.

## Key Findings

1. Unknown-fallback poisoning was real and is now fixed.
2. Embedding-domain mismatch was the dominant bottleneck for speaker-ID.
3. Speaker identification now works in mixed conditions with live-path embeddings.
4. PC fan noise was a real but secondary contaminant.
5. Device 4 / line-out path introduces 59 Hz mains hum.
