# M4 Self-Voice — Calibrated Assessment

> **Status:** Integrated, calibration phase
> **Date:** 2026-05-03
> **Scope:** M4 speaker-embedding attribution with M2 gating

## What We Know

### M4 is functional but not magical

ECAPA-TDNN (192-dim embeddings, SpeechBrain) provides real speaker discrimination. But:

| Observation | Value | Implication |
|-------------|-------|-------------|
| TTS self-similarity (identical audio) | 1.0 | Perfect, expected |
| TTS self-similarity (different text) | 0.51-0.84 | Content-independent for same voice |
| TTS vs human speech | 0.05-0.75 | **Significant overlap zone exists** |
| Dry TTS registration vs room-colored | 0.75→0.65 for human speech | Room-colored helps but doesn't fully separate |
| M4 confidence threshold (old 0.10) | Accepted almost everything | Far too permissive |
| M4 confidence threshold (new 0.50) | Rejects most human speech | Conservative; some genuine overlap may be lost |

### The fundamental limitation

ECAPA-TDNN was trained on real human speech (VoxCeleb). TTS voices occupy embedding space that can genuinely overlap with human voices. Room-colored registration (through the cerebellum forward model) shifts the template to include room acoustics, improving separation by ~0.1 in cosine similarity. This helps but doesn't eliminate the overlap.

### Combined M2+M4 gating is the right approach

M4 should not act independently. The ThalamicAttributor now requires:
1. M2 evidence: `match_score >= similarity_threshold` (is_self_like)
2. M4 evidence: `confidence >= 0.50` AND `is_self_voice` from ECAPA

This reclassified 93 false SELF frames as BOTH in the m3_with_m4_final data (800 frames, 5 turns). These frames had M4 confidence 0.14-0.48 with near-zero M2 evidence (match=0.054, causal=0.000).

### Dry vs room-colored registration

With dry TTS registration, `world_speech_sample.npy` scored 0.75 similarity to the self template — solidly in SELF territory. With room-colored registration (using `forward_model_1777730332.json`), this dropped to 0.65. Better, but still above the 0.50 threshold for some frames.

The 17 "below_threshold" rejections in the original data were cases where M4 was attempted but confidence was 0.05-0.10 — the old, permissive regime.

## Evaluation Results

`scripts/m4_threshold_sweep_synthetic.py`:
- 10 synthetic speakers, thresholds 0.05-0.95
- Perfect separation only at threshold 0.95 (synthetic signals too clean)
- Real speech-like signals show 0.61-0.92 self-vs-other similarity

`scripts/m4_real_tts_test.py`:
- 6 TTS probes: confidence 0.51-1.00 (mean 0.77)
- Short text ("Hello"): 0.51 (weakest)
- Human speech rejection: silence 0.02, noise -0.01, benchmark 0.08, sample 0.75

`scripts/m4_world_rejection_test.py`:
- Dry registration: 4/6 correct (world_sample false positive at 0.75)
- Room-colored: 5/6 correct (world_sample reduced to 0.65, still above 0.50)

## What "Working" Means Here

M4 is **working** in the sense that:
- Real ECAPA-TDNN is loaded and producing embeddings
- Combined M2+M4 gating prevents false SELF assignments
- Room-colored registration improves separation

M4 is **not yet proven** in the sense that:
- We haven't demonstrated it improves turn-taking outcomes
- The overlap zone between TTS and human speech means some genuine self-speech may be classified as BOTH
- We don't have enough real interruption data to measure the end-to-end impact

## Next Steps

1. **Collect real interruption data**: Record scenarios where Corollary is speaking and a human interrupts
2. **Use room-colored registration by default**: Wire `--cerebellum-model` into standard M3 runs
3. **Consider lower thresholds for specific scenarios**: 0.50 is conservative; some genuine self-speech at 0.40-0.50 may be worth accepting if M2 timing is strong
4. **Evaluate end-to-end**: Does M4 improve false interrupt / false defer rates in real turn-taking?
