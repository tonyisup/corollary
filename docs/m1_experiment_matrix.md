# M1 Experiment Matrix

> **Status:** Draft
> **Created:** 2026-04-17
> **Scope:** Runnable experiment checklist for early M1 source attribution work

## Purpose

This document turns the M1 source-attribution strategy into a structured experiment matrix.

It is designed to answer three practical questions:

1. What exact experiments should we run?
2. What variables are we changing?
3. What should success look like at each stage?

This matrix is intentionally pre-linguistic first. Full-word and sentence-level speech tasks are deferred until self/world attribution is stable.

## Stage Summary

- **M1a**: self-sound attribution
- **M1b**: babbling-driven room learning
- **M1c**: mixed-source attribution

## Common Axes

Each experiment should specify these axes explicitly.

### Stimulus type
- tone burst
- click
- click train
- chirp
- sustained vowel
- babble syllable
- nonsense syllable string
- short word fragment

### Environment
- quiet room
- mild room noise
- noisy room
- alternate room

### Device configuration
- default mic + default speaker
- alternate mic
- alternate speaker
- headset / near-field
- remote capture path

### Playback level
- low
- medium
- high

### Interference type
- none
- ambient noise
- external transient
- external sustained sound
- external speech-like source
- replayed prior self audio

### Temporal relationship
- self only
- world only
- world before self
- self before world
- partial overlap
- full overlap

### Labels
- self
- world
- overlap
- silence/background

## Standard Logging Schema

Each run should log at minimum:

- experiment id
- timestamp
- room / environment tag
- input device
- output device
- stimulus type
- playback level
- interference type
- overlap condition
- expected label
- predicted label
- confidence
- onset error ms
- offset error ms
- prediction error
- notes

## Stage 1, M1a Self-Sound Attribution

## Goal

Answer: **Did I just make that sound?**

## M1a-A: Chirp attribution baseline

**Purpose**
- Establish the strongest current synthetic self-attribution baseline.

**Setup**
- quiet room
- default devices
- single chirp
- no interference

**Expected label**
- self

**Metrics**
- self/not-self accuracy
- onset alignment error
- offset alignment error
- confidence stability across trials

**Pass signal**
- strong and repeatable self attribution in quiet conditions
- currently the best-performing M1A primitive in repo experiments

## M1a-B: Vowel-like attribution comparison

**Purpose**
- Compare a simple vowel-like primitive against the chirp baseline.

**Setup**
- quiet room
- default devices
- single synthetic vowel-like sound such as `ah`
- no interference

**Expected label**
- self

**Why this matters**
- vowel-like sounds appear developmentally meaningful and currently perform strongly in the synthetic M1A harness

## M1a-C: Tone burst attribution comparison

**Purpose**
- Compare a simpler periodic waveform against the chirp baseline.

**Setup**
- quiet room
- default devices
- single tone burst
- no interference

**Expected label**
- self

## M1a-D: Click attribution challenge

**Purpose**
- Test attribution on sharp transients.

**Setup**
- quiet room
- click or click train
- no interference

**Expected label**
- self

**Why this matters**
- clicks are useful for precise onset timing and room-delay estimation
- but currently act more like a detector stress test than a stable baseline

## M1a-E: Sustained vowel attribution

**Purpose**
- Move from simple transients to richer spectra.

**Setup**
- quiet room
- sustained vowel like `aa` or `oo`
- no interference

**Expected label**
- self

**Metrics**
- self accuracy
- alignment quality
- consistency across vowel types

## M1a-F: Babble syllable attribution

**Purpose**
- Test pre-linguistic speech-like sounds without semantic pressure.

**Setup**
- quiet room
- `ba`, `da`, `ma`, repeated or isolated
- no interference

**Expected label**
- self

## M1a-G: Silence negative control

**Purpose**
- Ensure the system does not hallucinate self-authorship.

**Setup**
- no emitted stimulus
- ambient recording only

**Expected label**
- silence/background or world

**Critical metric**
- false self-attribution rate

## M1a-H: Ambient-noise negative control

**Purpose**
- Stress the self/not-self boundary under passive environmental noise.

**Setup**
- no self emission
- fan, HVAC, keyboard noise, room rustle

**Expected label**
- world or silence/background

## Stage 2, M1b Babbling-Driven Room Learning

## Goal

Answer: **If I make this sound here, what should I expect to hear back?**

## M1b-A: Fixed-room convergence test

**Purpose**
- Measure forward-model improvement in one stable setup.

**Setup**
- fixed room
- fixed mic/speaker pair
- repeated babbling corpus

**Metrics**
- prediction error by iteration
- convergence slope
- attribution improvement pre/post adaptation

**Pass signal**
- clear improvement over repeated trials

## M1b-B: Alternate-room transfer test

**Purpose**
- Measure generalization failure and re-adaptation cost.

**Setup**
- train or calibrate in room A
- test in room B

**Metrics**
- attribution drop on transfer
- iterations needed to recover

## M1b-C: Volume sensitivity test

**Purpose**
- Verify that attribution and prediction do not collapse under amplitude changes.

**Setup**
- repeat same babbling corpus
- low / medium / high playback

**Metrics**
- attribution stability by level
- convergence change by level

## M1b-D: Stimulus generalization test

**Purpose**
- Check whether learning transfers across stimulus classes.

**Setup options**
- calibrate on tones, test on vowels
- calibrate on vowels, test on babble syllables
- calibrate on babble, test on clicks

**Metrics**
- post-adaptation attribution by unseen stimulus type

## Stage 3, M1c Mixed-Source Attribution

## Goal

Answer: **Which parts of the acoustic scene are self, world, or overlap?**

## M1c-A: Sequential world-before-self

**Purpose**
- Test clean temporal boundary detection.

**Setup**
- external source first
- self stimulus follows

**Expected labels**
- world then self

**Metrics**
- transition timing accuracy
- confusion near boundary

## M1c-B: Sequential self-before-world

**Purpose**
- Reverse ordering to catch asymmetric failures.

**Expected labels**
- self then world

## M1c-C: Partial overlap

**Purpose**
- Test mixed attribution during overlapping sound.

**Setup**
- external source begins during self output, or vice versa

**Expected labels**
- self → overlap → world, or inverse depending on schedule

**Metrics**
- overlap F1
- false self-attribution during overlap

## M1c-D: Full overlap with distinct spectra

**Purpose**
- Start with an easier overlap case.

**Setup**
- self tone or vowel
- world source with distinct spectral character

## M1c-E: Confounding-source challenge

**Purpose**
- Detect overclaiming of self authorship.

**Setup**
- world source designed to resemble self stimulus
- similar pitch, timbre, or envelope

**Critical metric**
- world → self confusion rate

## M1c-F: Replayed-self negative control

**Purpose**
- Ensure prior self audio is not mistaken for current self without present motor command.

**Setup**
- replay a previously recorded self sound externally
- no current self emission

**Expected label**
- world

This is one of the most important controls in the matrix.

## Suggested Execution Order

Run experiments in this order unless a blocker appears:

1. M1a-A chirp baseline
2. M1a-B vowel-like comparison baseline
3. M1a-C tone comparison baseline
4. M1a-G silence negative control
5. M1a-H ambient-noise negative control
6. M1a-D click challenge
7. M1a-E sustained vowel attribution
8. M1a-F babble syllable attribution
7. M1b-A fixed-room convergence
8. M1b-C volume sensitivity
9. M1b-B alternate-room transfer
10. M1b-D stimulus generalization
11. M1c-A sequential world-before-self
12. M1c-B sequential self-before-world
13. M1c-C partial overlap
14. M1c-D full overlap, distinct spectra
15. M1c-E confounding-source challenge
16. M1c-F replayed-self negative control

## Minimal First Benchmark Set

If we want the smallest useful starting set, I’d begin with these five:

1. chirp baseline
2. vowel-like comparison baseline
3. silence negative control
4. fixed-room babbling convergence
5. partial overlap attribution

That set gives us:
- a primitive self baseline
- a transient self baseline
- a false-positive check
- an adaptive learning test
- a first real mixed-source stress test

## Suggested Result Table Template

| Experiment | Room | Devices | Stimulus | Interference | Expected | Key Metric | Result | Notes |
|------------|------|---------|----------|--------------|----------|------------|--------|-------|
| M1a-A | quiet | default | tone burst | none | self | attribution accuracy | TBD | |
| M1a-E | quiet | default | none | ambient | world/silence | false self rate | TBD | |
| M1b-A | quiet | default | babble corpus | none | self | convergence slope | TBD | |
| M1c-C | quiet | default | babble | external tone | overlap | overlap F1 | TBD | |

## Exit Criteria by Stage

### Exit criteria for M1a
- self attribution works on simple sounds
- false self-attribution is low in silence and ambient conditions
- timing alignment is stable enough to support adaptation

### Exit criteria for M1b
- forward-model adaptation measurably improves attribution
- convergence is reproducible
- room changes cause understandable, measurable degradation

### Exit criteria for M1c
- self/world/overlap labeling is usable under structured interference
- false self claims are acceptably low
- replay and confounding-source controls are passed

## Design Reminder

If a result looks good only when judged by ASR transcripts, that is not enough for M1.

The core artifact of this stage should be an attribution decision grounded in timing, prediction, and acoustic evidence, not just a transcript side effect.

## Recommended Next Artifact

After this matrix, the most useful follow-up would be one of:

- a machine-readable experiment manifest, or
- a first implementation of `scripts/m1_tone_attribution.py`, or
- a standardized JSON schema for recording M1 eval runs

My recommendation would be the tone-attribution script first, because it gives us the simplest runnable anchor.