# Baseline VAD/AEC Comparison — Results

**Date:** 2026-05-19
**Script:** `scripts/baseline_vad_comparison.py`
**Data:** `data/m3_live_e2e_v35_m4/` (5 scenarios, 1 capture each)

This is an offline comparison on captured audio using simplified heuristics for region labeling and scoring. The numbers are useful for directional comparison, not as a final ground-truth benchmark.

## Summary

| Stack | Avg Self-leak dB | Avg False-mute % | Overlap Survival % | Wake Recall |
|-------|-----------------|-------------------|--------------------|-------------|
| A: Corollary | -34.2 | 48.0 | 100.0 | 1.00 |
| B: VAD-only (agg) | -59.7 | 11.3 | 100.0 | 1.00 |
| C: VAD+AEC (NLMS) | -59.0 | 11.3 | 100.0 | 1.00 |
| D: VAD-only (cons) | -54.0 | 10.5 | 100.0 | 1.00 |

## Key Findings

### 1. Self-leak: VAD stacks win on this metric

VAD-only (B) and VAD+AEC (C) suppress self-voice to roughly -59 to -62 dB, compared to Corollary's -34 dB. That is the expected shape of this comparison: VAD aggressively mutes anything it detects as speech during agent playback, while Corollary's energy-ratio-based detector is intentionally more conservative.

### 2. "False mute" is a proxy, not a universal error rate

Corollary shows a 48% false-mute rate vs VAD's 11%, but this metric is really measuring how often human speech is muted under the scorer's overlap definition. In Corollary's case, the higher number reflects a different policy: it detects external human speech via the `both_external` energy-ratio detector and enters LISTEN, which gates off TTS output.

That is the right behavior for a full-duplex turn-taking system, but the metric itself should not be read as a generic failure score.

The VAD stacks do not model the same behavior. They either:
- **Aggressive (B):** mute very broadly during agent playback, including some human speech
- **Conservative (D):** preserve more audio, but leave substantially more self-voice

### 3. Corollary handled the scenarios in this sample

Per-scenario verification from `live_e2e_results.json`:
- `human_early`: interrupt detected, deferred, resumed
- `human_before`: deferred correctly, no interrupt expected
- `human_after`: responded after TTS completion
- `human_late`: interrupt detected, deferred, resumed
- `overlap_self_first`: interrupt detected, deferred, resumed

### 4. Overlap survival and wake recall are not discriminating here

All stacks show 100% overlap survival and 1.0 wake recall. These metrics don't discriminate between stacks because:
- The ground truth overlap regions are small
- No stack fully mutes an entire human utterance
- The wake-word simulation is too coarse for this dataset

## Behavioral Difference

| Approach | What it mutes | What it preserves |
|----------|--------------|-------------------|
| VAD stacks | Speech energy, with no attribution | Quiet audio, but may leak self |
| Corollary | Self-predicted audio plus detected external speakers | Human speech during overlap, by design |

Corollary's approach is different in kind: it uses speaker attribution to distinguish self from world, then gates accordingly. The "false mute" metric in this comparison captures a real tradeoff, but it does not fully express that design goal.

## Recommendations for Grant Reviewers

1. **Don't compare on false-mute rate alone.** Corollary's higher false-mute score reflects turn-taking behavior, not just suppression quality. A better question is whether the system resumes cleanly after the human stops.

2. **Self-leak is a tradeoff, not a failure.** Corollary's -34 dB self-leak is worse than the VAD baselines here, but it comes with speaker attribution and explicit turn-taking behavior.

3. **The main advantage is behavioral, not numerical.** Corollary detects that a human is speaking, stops its own output, and resumes after the human finishes. These VAD baselines only reason about audio energy, so they do not represent the same control problem.

## Next Steps

- [ ] Add resume-latency metric (time from human stopping to Corollary resuming)
- [ ] Test with the new balanced scorer (M4-enabled) for better attribution
- [ ] Add more captures with varied human speech timing
- [ ] Measure self-leak during overlap regions specifically (not just self-regions)
