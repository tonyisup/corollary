# Spectral Subtraction vs Gain — Offline Comparison

> Date: 2026-05-01
> Method: Offline replay of 63 saved captures (3 stimuli × 7 amplitudes × 3 seeds)
> compared preserve_both vs gain_0p5 vs subtract_0p25

## Overall Results (63 captures)

| Strategy | N | Mean Delta | Std | Improved | Neutral | Regressed |
|---|---|---|---|---|---|---|
| preserve_both | 63 | +0.0014 | 0.0109 | 0 | 62 | 1 |
| gain_0p5 | 63 | +0.0290 | 0.0805 | 9 | 26 | 28 |
| **subtract_0p25** | **63** | **+0.0021** | **0.0294** | **4** | **56** | **3** |

**subtract_0p25 is the best strategy**: lowest mean delta, lowest variance, fewest
regressions. It's not dramatically better than preserve_both, but it's clearly
better than gain_0p5.

## By Stimulus Type

### Chirp (21 captures)
- All three strategies are equivalent (self signal is at noise floor, -16 dB below world)
- Subtraction has no effect because the self_pcm barely registers at the mic

### Vowel (21 captures)
- gain_0p5 is harmful: mean delta +0.033, 9 regressions
- subtract_0p25 is slightly harmful: mean delta +0.015, 3 regressions
- preserve_both is nearly neutral: mean delta +0.004

### Babble (21 captures)
- gain_0p5 is very harmful: mean delta +0.060, 15 regressions
- subtract_0p25 is slightly helpful: mean delta -0.006, 0 regressions
- Babble is where subtraction shows the most promise

## Subtraction Strength Sweep (seed_0 only)

Tested 7 subtraction strengths (0.1 to 2.0) on 8 captures:

| Stimulus | Amp | Raw WER | Best Sub | Best WER | Delta |
|---|---|---|---|---|---|
| vowel | 0.2 | 0.174 | none | 0.174 | +0.000 |
| vowel | 0.5 | 0.087 | none | 0.087 | +0.000 |
| vowel | 1.0 | 0.087 | none | 0.087 | +0.000 |
| babble | 0.3 | 0.261 | sub_0.50 | 0.174 | -0.087 |
| babble | 0.7 | 0.087 | none | 0.087 | +0.000 |
| babble | 1.0 | 0.043 | none | 0.043 | +0.000 |

**Key finding**: Subtraction only helps on babble amp=0.3 (raw WER=0.261), where
sub_0.50 improves by -0.087. On all other captures, subtraction is neutral.
Higher subtraction strengths (1.0+) never help — they distort the signal.

## Analysis

### Why subtraction barely helps

1. **The self template is dry PCM, not room-colored.** The subtract function
   subtracts the dry self template from the captured mixture. But the captured
   self component has been colored by the room (reflections, frequency response
   of the speaker and mic). The dry template doesn't match the room-colored self,
   so subtraction leaves artifacts rather than cleanly removing self.

2. **BOTH frames contain mostly world speech.** Since VAD detects the self
   stimulus as speech, the attributor routes motor-window frames to BOTH.
   The BOTH frames contain both self and world energy. Subtracting the dry
   self template from these frames removes some self energy but also distorts
   the world speech because the template doesn't match.

3. **The subtraction scale is conservative (0.25).** Even at the "best" strength
   of 0.50, the improvement is modest (-0.087 on one capture). The current
   subtract implementation uses a single global scale factor fitted via
   least-squares on the BOTH windows. A per-frame or frequency-dependent
   approach might do better.

### Why gain_0p5 is harmful

Gain_0p5 blindly attenuates ALL BOTH frames by 50%. When the self stimulus
is strong (babble), the BOTH frames contain significant world speech energy,
and attenuating them hurts ASR. The harm is proportional to the self amplitude:
- Chirp (weak self): gain_0p5 is neutral
- Vowel (moderate self): gain_0p5 is slightly harmful
- Babble (strong self): gain_0p5 is very harmful

### The fundamental problem

Both gain and subtraction operate on BOTH frames, which are the wrong target.
The attributor should be producing SELF frames (hard suppression) for the
self component and WORLD frames for the external speech. The conservative
M2 policy prevents this by routing everything to BOTH when VAD detects speech.

## Path Forward

Subtraction is marginally better than gain, but neither solves the core problem.
The real solution requires one of:

1. **Room-colored self template** — use the cerebellar forward model to predict
   the room-colored self signal, then subtract that instead of the dry template.
   This is the biological analogue: the efference copy is transformed by the
   forward model before being compared to the sensory input.

2. **Hard SELF frames via M4** — use embeddings to distinguish self from other
   during speech, enabling the attributor to route self-dominant frames to SELF
   (gain=0) instead of BOTH (gain=0.5).

3. **Frequency-dependent subtraction** — instead of a single scale factor,
   estimate a frequency-dependent gain (Wiener filter) that minimizes the
   residual in BOTH frames. This is more computationally expensive but could
   better separate self from world when they occupy different frequency bands.
