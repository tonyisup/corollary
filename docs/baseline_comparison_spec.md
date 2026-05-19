# Spec: Corollary vs. Standard VAD/AEC Baseline Comparison

**Date:** 2026-05-19
**Type:** Benchmark experiment specification
**Goal:** Produce empirical, side-by-side data showing what corollary-discharge architecture buys over a standard VAD+AEC stack.

---

## 1. Why This Exists

Grant reviewers and skeptical engineers will ask: "Why not just use WebRTC VAD + a standard AEC + a wake-word model?" This experiment produces a fair, apples-to-apples comparison on the same audio artifacts, so we can answer that with data rather than architecture diagrams.

## 2. Baselines Compared

| Stack | Components | What it represents |
|-------|-----------|-------------------|
| **A: Corollary (full)** | Forward model (room-colored FIR) + M4 speaker-embedding scorer + thalamic gating + AgentiveController | The hypothesis — physical self-prediction + speaker attribution drives better full-duplex behavior |
| **B: VAD-only** | WebRTC VAD (aggressiveness=1) on the raw mic input, applied as a mute gate during agent playback | The cheapest common baseline — "just turn off the mic when I'm talking" |
| **C: VAD + simple AEC** | Linear echo cancellation (adaptive filter, ~256 taps) applied before VAD | A plausible open-source stack — what a careful hobbyist would build |
| **D: VAD-only (conservative)** | WebRTC VAD (aggressiveness=3, high sensitivity) — never mutes during overlap | The "never lose a word" baseline — maximal false positives for self, minimal human loss |

Stack B and D bracket the practical VAD space: aggressive vs. conservative. If Corollary beats both, it beats the whole range.

## 3. Test Artifacts (inputs)

We use existing mixed-source captures from the repo — no new recording required:

- **M4 balanced captures:** `data/m4_balanced_current_room/` — self + synthetic agent + human negatives
- **Stress tests:** `data/stress_test/` — phone music, room noise, severe interference
- **Live TTS-to-Mic captures:** Any recent `data/m3_streaming_repair_live_*` or `data/m6_introspective_*` directory that contains captured mic audio during agent playback

Each capture is a mono 16kHz WAV of the room mic while the agent is (or was) speaking, with known ground truth about when self-speech, world speech, and overlap occurred.

## 4. Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Self-leak dB** | Residual self-voice RMS after suppression | The raw "did I silence myself" question |
| **Human WER loss** | WER of human transcripts under each stack vs. clean baseline | The real cost: how many human words get lost |
| **False-mute rate** | % of human speech frames incorrectly suppressed by the stack | How often the system accidentally silences a person |
| **Self-attribution F1** | Frame-level SELF/WORLD classification accuracy (Corollary only; N/A for VAD baselines) | The internal mechanism's precision |
| **Overlap survival rate** | % of overlap frames where at least partial human speech is preserved (not fully muted) | The full-duplex test |
| **Wake-word-free recall** | For wake-word simulation: % of human utterances caught during agent speech vs. a wake-word system that dead-zones | Demonstrates the "always listening, never self-triggered" advantage |

## 5. Test Protocol

### Per capture:
1. Feed the raw mic WAV into each stack (A, B, C, D).
2. Each stack produces an output WAV (suppressed/enhanced).
3. Run a reference ASR (Whisper base) on each output.
4. Compare ASR transcripts and per-stack metrics.

### Wake-word simulation (separate run):
1. For captures with known human utterance timing, simulate a push-to-talk/wake-word system that mutes during agent playback.
2. Count how many human utterances fall entirely inside agent speech and are therefore lost.
3. Compare to Corollary's recall on the same captures.

## 6. Expected Outputs

### 1. Summary table
```
Capture          | Stack | Self-leak dB | Human WER loss | False-mute % | Overlap survival %
-----------------+-------+--------------+----------------+--------------+-------------------
m3_live_001      | A     | -28.3        | 2.1%           | 0            | 97.2
m3_live_001      | B     | -12.1        | 18.7%          | 15.3         | 68.4
m3_live_001      | C     | -18.5        | 8.4%           | 6.1          | 82.7
m3_live_001      | D     | -3.2         | 1.8%           | 0            | 98.1
```

### 2. Per-stack aggregate
| Stack | Avg self-leak dB | Avg human WER loss | Avg false-mute % | Notes |
|-------|-----------------|-------------------|------------------|-------|
| A: Corollary | TBD | TBD | TBD | Full pipeline |
| B: VAD-only (aggressive) | TBD | TBD | TBD | Cheap but lossy |
| C: VAD+AEC | TBD | TBD | TBD | Middle ground |
| D: VAD-only (conservative) | TBD | TBD | TBD | Never mutes, self-leaks |

### 3. Plots
- Bar chart: human WER loss by stack (lower is better)
- Bar chart: self-leak dB by stack (lower is better — more negative = less leak)
- Overlap survival rate by stack

## 7. Script Structure

```
scripts/baseline_vad_comparison.py
├── load_capture(path) → raw audio + ground truth labels
├── run_vad_stack(audio, config) → suppressed audio, mute mask
├── run_aec_vad_stack(audio, config) → suppressed audio, mute mask
├── run_corollary_stack(audio, ground_truth) → suppressed audio, attribution mask
├── run_wake_word_simulation(audio, ground_truth) → recall stats
├── measure_wer_loss(audio, reference_text) → WER
├── measure_self_leak(audio, self_regions) → dB
├── measure_false_mute(mute_mask, human_regions) → %
├── measure_overlap_survival(mute_mask, overlap_regions) → %
└── produce_table(captures, stacks) → markdown + CSV + plots
```

## 8. What's Not in Scope

- Commercial AEC systems (NVIDIa Broadcast, Krisp) — no public API for fair comparison.
- End-to-end commercial wake-word detectors (Hey Siri, OK Google) — same reason.
- Subjective listening tests — this is an objective benchmark.

## 9. Success Criteria

Corollary wins the grant argument if:
- Self-leak is competitive with or better than AEC (Stack C), while VAD-only (Stacks B/D) either leaks badly or mutes too aggressively.
- Human WER loss is strictly better than VAD-only aggressive (Stack B), which is the most common cheap baseline.
- Overlap survival rate is >90% for Corollary, showing it doesn't just mute the whole channel.

If Corollary loses to a simple VAD, we learn something real — and that's also publication-worthy.
