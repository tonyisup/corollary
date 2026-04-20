# M1c Hardware Results: Source Attribution

> **Status:** Active Research
> **Date:** 2026-04-20
> **Scope:** Hardware-in-the-loop source attribution (Self vs. World)

## Overview

This phase transition Corollary from synthetic source separation to real-world hardware attribution. Using a Mac mini (internal speakers) + Fifine Microphone + External Speaker setup, we successfully implemented a motor-aware "Thalamic Attributor" that distinguishes between self-caused and world-caused acoustic events.

## Key Technical Achievements

### 1. Motor-Aware Thalamic Attributor
We implemented the `ThalamicAttributor` (`packages/thalamus/attributor.py`) which provides high-level source verdicts:
- **SELF**: High acoustic match + Active motor command + Correct causal timing.
- **WORLD**: High/Low acoustic match + **No motor command**.
- **BOTH**: Energy present during motor command + Poor match/timing (indicates masking).
- **SILENCE**: Energy below threshold.

### 2. The Replay Control (M1c-F)
Successfully passed the most critical negative control:
- **Scenario**: Replaying a recorded "self" probe through an external speaker.
- **Result**: The system correctly identified the signal as **WORLD** (0.90 confidence) despite its high spectral similarity (0.64 match) to the self-template. This proves that top-down motor gating is functional.

### 3. Syllable Performance Matrix (M1c)
We mapped the performance of different syllabic babble classes in overlapping noise (Chirp):

| Stimulus | Sequential Match | Full Overlap Match | Timing Error (ms) |
| :--- | :---: | :---: | :---: |
| `babble_ba_var` | 0.55 | 0.14 | > 140ms |
| `babble_da_var` | 0.49 | 0.18 | ~80ms |
| `babble_ma_var` | **0.75** | **0.34** | ~80ms |

**Conclusion**: `ma` is the strongest spectral probe, likely due to the distinctive nasal-to-vowel transition.

### 4. Temporal Redundancy Advantage
Repeated babble strings (`babble_ba_repeat_short_tight_var`) demonstrated a significant stability advantage over single tokens:
- **Single Token Timing Error**: 80ms - 200ms.
- **Repeated String Timing Error**: **5ms - 11ms**.
- **Lesson**: Rhythmic repetition provides a powerful secondary cue for self-attribution that survives acoustic crowding better than spectral matching alone.

## Experimental Setup

- **Input**: Fifine K669B USB Microphone (Index 2).
- **Output (Self)**: Mac mini Internal Speakers (Index 4).
- **Output (World)**: External Speakers / Monitor (Index 3).
- **Sample Rate**: 16 kHz (Resampled from 48 kHz hardware native).

## Summary Verdicts (Hardware Harness)

| Mode | Self Score | Verdict | Interpretation |
| :--- | :---: | :---: | :--- |
| **Sequential** | 0.55 - 0.75 | `SELF` / `BOTH` | Strong attribution when separated. |
| **Partial Overlap**| 0.20 - 0.40 | `BOTH` | Correct identification of confounded stream. |
| **Full Overlap** | 0.14 - 0.34 | `BOTH` | System "knows" it is hearing more than itself. |
| **Replay Control** | 0.64 | `WORLD` | **SUCCESS**: Top-down inhibition of self-like external audio. |

## Next Steps

1. **Competitive Agents**: Test if the system can distinguish between different vocal agents (e.g., Self plays `ma`, World plays `ba`).
2. **Speech-Aware Separation (M2)**: Move from pre-linguistic babble to word-level suppression and transcript-aware attribution.
3. **Trust-Gated Learning**: Use the `BOTH` verdict to pause room-model adaptation during contaminated audio segments.
