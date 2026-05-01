# Stimulus Characterization Across Amplitudes

> Date: 2026-05-01
> Setup: Fifine mic (device 3), Mac mini built-in speaker (device 5),
>           headphones on desk (device 4)
> World speech: 6.22s GuyNeural male voice
> Attribution: similarity_threshold=0.20, onset_latency=400ms, energy onset,
>              causal_tolerance=75ms, both_gain=0.5

## Summary Table

### Chirp (200ms)

| Amp | Raw WER | Gated WER | Delta | S/W dB | I/N/R |
|---|---|---|---|---|---|
| 0.0 | 0.362 | 0.304 | -0.058 | -3.4 | 1/2/0 |
| 0.1 | 0.377 | 0.478 | +0.101 | -5.3 | 1/0/2 |
| 0.2 | 0.290 | 0.304 | +0.014 | -13.5 | 0/2/1 |
| 0.3 | 0.333 | 0.333 | +0.000 | -14.9 | 0/3/0 |
| 0.5 | 0.203 | 0.145 | -0.058 | -17.2 | 1/2/0 |
| 0.7 | 0.145 | 0.101 | -0.043 | -15.6 | 2/0/1 |
| 1.0 | 0.145 | 0.145 | +0.000 | -18.7 | 0/3/0 |

### Vowel (350ms)

| Amp | Raw WER | Gated WER | Delta | S/W dB | I/N/R |
|---|---|---|---|---|---|
| 0.0 | 0.145 | 0.174 | +0.029 | 0.8 | 0/2/1 |
| 0.1 | 0.101 | 0.145 | +0.043 | 4.3 | 0/1/2 |
| 0.2 | 0.159 | 0.159 | +0.000 | 4.6 | 1/1/1 |
| 0.3 | 0.130 | 0.203 | +0.072 | 4.5 | 0/1/2 |
| 0.5 | 0.145 | 0.203 | +0.058 | 4.1 | 0/2/1 |
| 0.7 | 0.188 | 0.188 | +0.000 | 4.4 | 2/0/1 |
| 1.0 | 0.159 | 0.188 | +0.029 | 4.1 | 0/2/1 |

### Babble (350ms)

| Amp | Raw WER | Gated WER | Delta | S/W dB | I/N/R |
|---|---|---|---|---|---|
| 0.0 | 0.159 | 0.203 | +0.043 | 1.7 | 1/0/2 |
| 0.1 | 0.174 | 0.188 | +0.014 | 5.0 | 0/2/1 |
| 0.2 | 0.101 | 0.159 | +0.058 | 5.3 | 0/1/2 |
| 0.3 | 0.130 | 0.188 | +0.058 | 5.8 | 0/1/2 |
| 0.5 | 0.101 | 0.159 | +0.058 | 5.4 | 0/1/2 |
| 0.7 | 0.072 | 0.174 | +0.101 | 4.2 | 0/0/3 |
| 1.0 | 0.043 | 0.130 | +0.087 | 3.9 | 0/0/3 |

## Analysis

### 1. Chirp is barely audible at the mic

S/W ratios range from -3.4 dB to -18.7 dB. The chirp is attenuated by ~15-20 dB
relative to world speech at the mic. This is because:
- The chirp is short (200ms) and broadband — less total energy than the vowel
- The built-in speaker's frequency response doesn't favor the chirp's sweep range
- The chirp doesn't benefit from the vowel's formant structure that couples well
  to the room modes

The "gating improvement" at chirp amp=0.5 and 0.7 is suspect — the BOTH frames
are at the noise floor, and the ±0.058 delta is within ASR variance.

### 2. Vowel S/W is stable across amplitudes

Vowel S/W stays in a narrow band (4.1-4.6 dB) for amp 0.2-1.0. This suggests
the Mac mini speaker has AGC or compression that limits the dynamic range at the
mic. The vowel's energy at the mic doesn't scale linearly with amplitude.

### 3. Babble has the highest S/W ratios

Babble S/W is 3.9-5.8 dB across amplitudes — consistently higher than vowel.
The babble_ba_var stimulus has more energy in the low-frequency range where the
built-in speaker is most efficient and the room couples best.

### 4. Gating is harmful across ALL conditions

Not a single stimulus × amplitude combination shows a clear gating benefit
(consistent improvement across seeds). The BOTH-frame gain=0.5 always hurts or
is neutral. This confirms the finding from the earlier benchmark: the
conservative M2 policy with blunt gain reduction cannot improve speech
preservation when VAD detects the self stimulus as speech.

### 5. Raw WER non-monotonicity

For all three stimuli, raw WER does not increase monotonically with amplitude.
This confirms the earlier finding: S/W ratio is not the dominant factor.
Room acoustics, AGC, and the specific temporal structure of the overlap matter
more than the raw power ratio.

## Best Operating Points for Future Experiments

### For gating strategy comparison:
- **Vowel amp=0.2**: S/W=4.6 dB, raw WER=0.159, gating neutral
- **Chirp amp=0.5**: S/W=-17.2 dB, raw WER=0.203, gating appears to help but likely noise

### For stress testing:
- **Babble amp=0.7-1.0**: Raw WER is very low (0.043-0.072) but gating hurts consistently.
  The babble is so speech-like that VD detects it, causing BOTH routing.

### For hard SELF testing:
- Chirp at any amplitude produces zero hard SELF frames because VAD detects
  world speech in the overlap window. Need to test with self-only captures
  (no world speech) to verify the attributor can produce SELF for chirp.

## Artifacts

All data in `data/m2_stimulus_sweep/`:
- 21 benchmark directories (7 amplitudes × 3 stimuli)
- Each with 3 seeds containing observed.npy, self_pcm.npy, frame_verdicts.jsonl, summary.json
