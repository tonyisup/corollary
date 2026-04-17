# M1 Parameter Tuning Guide

> **Status:** Draft
> **Created:** 2026-04-17
> **Scope:** Practical tuning guide for improving metrics across M1a, M1b, and M1c

## Purpose

This document complements the M1 source-attribution plan and experiment matrix by answering a practical question:

**When a metric is bad, what should we tune first?**

The goal is to make M1 experiments operational rather than interpretive.

For each stage, this guide maps:
- target metrics
- likely failure modes
- tunable parameters
- expected tradeoffs

## Tuning Philosophy

Early M1 should optimize for **source attribution quality**, not transcription quality.

If a metric improves only because the downstream ASR happened to behave better, that is not a clean win for M1.

The highest-value parameters in M1 are the ones that improve:
- timing alignment
- self/not-self discrimination
- adaptation stability
- overlap robustness

## Parameter Families

## 1. Stimulus Parameters

These control what kind of sound Corollary emits into the environment.

### Parameters
- stimulus class
  - tone
  - click
  - click train
  - chirp
  - sustained vowel
  - babble syllable
  - nonsense syllable string
- stimulus duration
- inter-stimulus interval
- playback amplitude
- onset/offset ramp shape
- repetition count per trial
- corpus diversity across a run

### Main effects
- clicks improve temporal precision and onset evaluation
- tones improve stability for simple spectral matching
- vowels and babble add spectral richness
- longer stimuli can help alignment but may blur onset/offset precision
- high amplitude improves SNR until it begins to clip or distort

### Main risks
- overfitting the system to one stimulus class
- using overly rich stimuli too early, which hides primitive attribution failures
- clipping from aggressive playback level

## 2. Timing and Alignment Parameters

These control the temporal matching between motor command, predicted sensory effect, and observed microphone return.

### Parameters
- pre-capture duration
- post-capture tail duration
- frame size
- hop size
- onset detection threshold
- offset detection threshold
- allowed shift search window
- temporal alignment scoring rule
- minimum event duration
- silence detection threshold

### Main effects
- tighter alignment can sharply improve self attribution
- shorter frames improve temporal precision but may increase noise sensitivity
- larger shift windows improve robustness to device/room delay at the cost of ambiguity

### Main risks
- false attribution caused by bad alignment rather than bad modeling
- unstable event boundaries when thresholds are too aggressive
- label thrashing if minimum event duration is too short

## 3. Forward-Model Parameters

These control the cerebellum’s prediction of the sensory consequences of self-generated sound.

### Parameters
- impulse response length
- adaptive filter step size (`nlms_mu`)
- adaptation iteration count
- warm-up period before evaluation
- model trust weighting
- smoothing / averaging factor
- residual threshold used for attribution
- per-band weighting across frequency bands

### Main effects
- longer impulse responses can model more room tail
- larger step sizes adapt faster but can oscillate
- trust weighting helps decide how heavily attribution should depend on prediction

### Main risks
- overfitting room noise or transient artifacts
- slow adaptation mistaken for model failure
- unstable convergence from overly aggressive adaptation

## 4. Front-End Acoustic Parameters

These are the DSP knobs before attribution logic.

### Parameters
- sample rate
- normalization strategy
- pre-emphasis / filtering
- noise calibration duration
- noise subtraction factor (`alpha`)
- spectral floor (`beta`)
- band-pass limits
- mel bin count
- STFT window length
- STFT hop length
- denoising on/off

### Main effects
- better front-end stability improves every downstream metric
- noise subtraction can reduce false positives in quiet tests
- too much subtraction can destroy useful signal structure

### Main risks
- “improved metrics” caused by over-cleaning rather than real attribution improvement
- front-end settings that work for one room but fail elsewhere

## 5. Decision / Gating Parameters

These determine how the thalamus classifies observed sound.

### Parameters
- self-attribution threshold
- world threshold
- overlap threshold
- confidence cutoff
- hysteresis
- temporal smoothing window
- minimum label duration
- agreement weighting between timing match and spectral match

### Main effects
- thresholds determine precision/recall tradeoffs
- hysteresis improves label stability under overlap
- confidence cutoffs reduce brittle decisions but may increase abstention

### Main risks
- threshold tuning that looks good only on one dataset
- overlap being collapsed into self or world because no explicit overlap criterion exists

## Repo-Visible Knobs Already Present

The current repo already exposes a useful subset of these knobs, especially in `scripts/m1_babble_test.py`.

### Observed current knobs
- `--iterations`
- `--input-device`
- `--output-device`
- `--ir-length-ms`
- `--nlms-mu`
- `--no-noise-cal`
- `--no-shush`
- `NOISE_CALIBRATION_DURATION`
- spectral subtraction oversubtraction factor `alpha`
- spectral floor `beta`
- mel-frame shift search window

### Immediate implication

Corollary is already close to an autoresearch-friendly tuning setup for M1b. The main missing piece is not knobs, but a more systematic harness for:
- recording metrics
- sweeping parameter combinations
- saving comparable results
- ranking outcomes by stage-specific objectives

## Stage-by-Stage Tuning Map

## M1a: Pre-Linguistic Self-Sound Attribution

### Primary metrics
- self/not-self attribution accuracy
- onset alignment error
- offset alignment error
- false self-attribution rate
- confidence stability

### Tune first
1. playback amplitude
2. onset threshold
3. offset threshold
4. frame size / hop size
5. allowed shift window
6. minimum event duration
7. self-attribution threshold

### If false self-attribution is high
Likely causes:
- silence threshold too low
- self threshold too permissive
- bad alignment window
- ambient noise insufficiently modeled

Try:
- raise silence/background threshold
- tighten self-attribution threshold
- narrow allowed shift window if it is matching spurious events
- increase negative-control testing
- increase noise calibration duration

Tradeoff:
- precision improves, but recall may drop

### If self recall is poor
Likely causes:
- playback too quiet
- onset threshold too high
- shift search too narrow
- event duration threshold too strict

Try:
- increase playback amplitude carefully
- lower onset threshold slightly
- widen shift search range
- reduce minimum event duration

Tradeoff:
- recall improves, but false positives may increase

### If onset/offset alignment is unstable
Likely causes:
- frames too large
- capture window poorly chosen
- threshold noise sensitivity

Try:
- shorten frame size
- reduce hop size
- increase pre-capture or post-capture tail
- smooth envelope before thresholding

Tradeoff:
- precision improves, but computational load may rise

## M1b: Babbling-Driven Room Model Learning

### Primary metrics
- prediction error
- convergence slope
- attribution improvement after adaptation
- room-transfer degradation

### Tune first
1. `nlms_mu`
2. impulse response length
3. number of iterations
4. stimulus diversity
5. playback amplitude
6. trust weighting for prediction-driven attribution

### If convergence is too slow
Likely causes:
- step size too low
- stimuli not informative enough
- playback amplitude too weak

Try:
- increase `nlms_mu` gradually
- use clicks or rich babble instead of only tones
- increase playback amplitude within clean range
- add more iterations

Tradeoff:
- faster learning, but higher instability risk

### If convergence oscillates or degrades
Likely causes:
- step size too high
- impulse response too long for data volume
- noisy observations dominating adaptation

Try:
- reduce `nlms_mu`
- shorten IR length
- improve noise subtraction or calibration
- add a warm-up phase before scoring

Tradeoff:
- stability improves, but adaptation slows

### If room transfer is poor
Likely causes:
- overfitting to one room
- stimulus set too narrow
- trust in cached forward model too high

Try:
- diversify stimulus corpus
- reduce model trust after environment change
- re-adapt with a short babbling warm-up in new room
- compare shorter vs longer IRs

Tradeoff:
- generalization improves, but peak in-room performance may dip

## M1c: Mixed-Source Attribution

### Primary metrics
- frame-level self/world/overlap F1
- chunk-level attribution accuracy
- false self-attribution under interference
- confusion matrix quality

### Tune first
1. self threshold
2. overlap threshold
3. temporal smoothing / hysteresis
4. per-band weighting
5. confidence cutoff
6. allowed shift range under overlap

### If overlap is misclassified as self
Likely causes:
- overlap criterion too weak
- self threshold too permissive
- smoothing too sticky toward self

Try:
- raise overlap sensitivity
- tighten self threshold
- reduce hysteresis toward self state
- weight frequency bands that distinguish sources better

Tradeoff:
- overlap detection improves, but pure self recall may dip

### If world sounds are mislabeled as self
Likely causes:
- timing match dominating too much
- confounding spectral similarity
- replay confusion

Try:
- require stronger agreement between timing and spectral evidence
- add explicit replay negative controls
- lower trust in prediction when no current motor command exists
- tighten confidence threshold for self labels

Tradeoff:
- false self claims decrease, but undecided cases may rise

### If labels flicker under mixed audio
Likely causes:
- no hysteresis
- smoothing window too short
- thresholds too close together

Try:
- add or increase hysteresis
- increase smoothing window modestly
- separate self/world/overlap thresholds more clearly
- impose minimum label duration

Tradeoff:
- stability improves, but transitions may look less sharp

## Failure-Mode-to-Knob Lookup

## Symptom: high false self-attribution in silence
Tune:
- silence threshold
- self threshold
- noise calibration duration
- noise subtraction factor
- confidence cutoff

## Symptom: poor attribution despite clear emitted sound
Tune:
- playback amplitude
- onset threshold
- shift window
- frame/hop size
- minimum event duration

## Symptom: adaptation never converges
Tune:
- `nlms_mu`
- IR length
- iteration count
- playback amplitude
- noise subtraction settings

## Symptom: good in one room, bad in another
Tune:
- IR length
- trust weighting
- warm-up iterations in new room
- stimulus diversity
- front-end normalization

## Symptom: overlap mostly collapses to self or world
Tune:
- overlap threshold
- hysteresis
- temporal smoothing
- per-band weighting
- confidence logic

## Symptom: replayed prior self audio is called current self
Tune:
- dependence on active motor-command timing
- self threshold
- confidence cutoff
- agreement rule between prediction and observed onset

## Suggested Tuning Discipline

To avoid tuning chaos:

1. change one parameter family at a time
2. use a fixed benchmark subset first
3. save every run with config + metrics
4. rank results by stage-specific metrics, not intuition
5. do not let WER dominate M1 decisions

## Recommended First Sweep Set

If starting now, I would run these controlled sweeps first:

### M1a starter sweeps
- playback amplitude: low / medium / high
- onset threshold: 3 values
- shift window: narrow / medium / wide
- minimum event duration: 3 values

### M1b starter sweeps
- `nlms_mu`: 0.05, 0.1, 0.25, 0.4
- IR length: 100 ms, 200 ms, 300 ms, 500 ms
- iterations: 5, 10, 20
- noise calibration: on / off

### M1c starter sweeps
- self threshold: 3 values
- overlap threshold: 3 values
- hysteresis: off / low / medium
- smoothing window: short / medium / long

## Closing

The important shift is this:

M1 should not just tell us whether the system works. It should tell us **which knobs move which failure modes**.

That turns the project from a loose set of experiments into a controllable developmental program.
