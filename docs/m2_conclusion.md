# M2 Speech Preservation — Conclusion

> Date: 2026-05-01
> Status: Investigation complete — fundamental limitation identified

## What We Tried

| Strategy | Approach | Result |
|---|---|---|
| Blunt gain (0.5) on BOTH frames | Attenuate all BOTH frames by 50% | Harmful (28/63 regressions) |
| Dry-template subtraction (0.25) | Subtract aligned self template from BOTH frames | Marginally better than gain, still not helpful |
| Mel-domain spectral subtraction | Subtract self mel from observed mel | SVSR 24-30 dB but can't convert back to audio for ASR |
| Self-window zeroing | Zero out audio during self stimulus window | Harmful — removes world speech during overlap |
| Energy-ratio soft gate | Suppress proportionally to self/world energy ratio | Inconsistent — works on some seeds, neutral on others |

## Fundamental Limitation

**Without a room-colored self template, M2 gating cannot improve speech preservation.**

The core problem:
1. VAD detects self stimulus as speech → attributor routes to BOTH (conservative)
2. BOTH frames contain mixed self + world energy
3. Any suppression in BOTH frames attenuates world speech along with self
4. The dry self template doesn't match the room-colored self at the mic
5. Energy-based estimates are unreliable because the room transfer function is unknown

## What Would Make M2 Work

### Option A: Train the cerebellar forward model (enables M4 too)
- Run M1b babbling in current room → learn room IR
- Use `cerebellum.predict_pcm(self_dry)` to generate room-colored template
- Register M4 with room-colored template → hard SELF frames during speech
- This is the biological solution: efference copy transformed by forward model

### Option B: Bypass VAD for known self stimuli
- If we know the self stimulus is not speech (e.g., chirp), skip VAD conservatism
- Route to SELF directly → hard suppression
- Limited: only works for non-speech self stimuli

### Option C: Accept BOTH, improve ASR robustness
- Don't suppress BOTH frames
- Focus on making ASR handle mixed self+world audio better
- Not really a gating improvement

## Recommendation

**Proceed to M4 + forward model training.** The M2 gating problem is fundamentally a classification problem (is this self or world?), not a suppression problem. Once M4 can correctly classify self vs world during speech, the attributor will produce SELF frames (hard suppression) instead of BOTH frames (conservative preservation), and gating will actually improve speech preservation.

The forward model training is the prerequisite for both:
- M4: room-colored self template for embedding comparison
- M2: room-colored self template for spectral subtraction
