# M1 Source Attribution Plan

> **Status:** Draft
> **Created:** 2026-04-17
> **Scope:** Early hardware milestones for pre-linguistic self/world source separation and attribution

## Purpose

This document refines M1 into a developmental sequence focused on **source separation and attribution before language**.

Corollary should not begin by asking, "what word did I just say?" It should begin by asking:

- Did a sound occur?
- Did **I** cause that sound?
- If I caused it, how did the room transform it?
- If another source is present, which parts are self, world, or overlap?

This mirrors the infant-development analogy that has guided the project: early selfhood is not lexical. It is sensorimotor attribution.

## Core Principle

**Attribution precedes transcription.**

Word-level ASR and lexical self-suppression should be treated as later capabilities built on top of a stable self/world acoustic boundary.

For M1, the primary success condition is not low WER. It is reliable identification of whether observed sound is:

- self-caused
- externally caused
- mixed

## Relationship to Existing Repo State

The repo already contains work that points in this direction:

- `scripts/m1_babble_test.py`
- `scripts/m1_wandering_babble.py`
- `scripts/m1_loop_test.py`
- `scripts/m1_mic_capture.py`
- `packages/cerebellum/adaptive_model.py`
- `packages/thalamus/gate.py`
- `packages/cochlea/capture.py`

This plan treats those not as side experiments, but as the canonical path for early M1.

## Developmental Breakdown

## M1a: Pre-Linguistic Self-Sound Attribution

### Goal

Determine whether an observed acoustic event is self-caused, independent of lexical content.

### Guiding question

**"Did I just make that sound?"**

### Allowed stimulus types

Start with simple, controllable sound classes:

- tones
- clicks
- chirps
- sustained vowels
- babble syllables (`ba`, `da`, `ma`)
- nonsense syllable strings
- short synthetic speech fragments only after simpler sounds work

**Current repo guidance:** use **chirps** as the primary M1A benchmark stimulus for now, with **vowel-like sounds** and tones as secondary comparison stimuli. In current synthetic M1A experiments, chirps are the most robust across the search space, vowel-like sounds are also strong and developmentally meaningful, and clicks remain a transient-onset challenge case rather than a stable baseline.

Do **not** begin with long, meaningful sentences as the primary benchmark.

### Required capabilities

- Emit a sound with known onset and duration
- Capture microphone audio during and after emission
- Align observed acoustic events with motor command / efference-copy timing
- Produce a self/not-self attribution label with confidence

### Success criteria

- High self/not-self attribution accuracy in quiet conditions
- Stable temporal onset/offset alignment between emitted and observed signal
- Reproducible results across repeated runs in the same environment
- Graceful degradation when room conditions change

**Current interpretation:** the present harness supports chirps, vowel-like sounds, and tones as viable M1A stimuli. Chirps are currently the most robust default, vowel-like sounds are especially interesting for developmental realism, and clicks are still serving mainly as a stress test for transient onset estimation rather than as a stable baseline.
### Recommended first experiments

1. **Chirp attribution test**
   - Emit a short chirp
   - Record mic return
   - Detect whether the captured event aligns with expected onset
   - Score self attribution
   - Treat this as the current strongest and most robust synthetic baseline

2. **Vowel-like attribution test**
   - Emit a simple vowel-like primitive (for example `ah`)
   - Use as the developmental comparison stimulus closest to early vocal imitation
   - Compare robustness and onset quality against chirp

3. **Tone attribution test**
   - Emit a short tone burst
   - Use as a secondary comparison stimulus
   - Verify that attribution remains good under a less onset-rich waveform

4. **Click train attribution test**
   - Emit short clicks with fixed spacing
   - Keep as an explicit transient-onset challenge case
   - Do not treat clicks as the primary M1A benchmark until transient onset detection improves

5. **Babble syllable attribution test**
   - Emit `ba`, `da`, `ma`, etc.
   - Measure whether attribution remains stable on richer spectra

6. **Silence negative-control test**
   - No self emission
   - Ensure ambient events are not mislabeled as self

## M1b: Babbling-Driven Room Model Learning

### Goal

Learn how the environment transforms self-generated sound by repeated emission and observation.

### Guiding question

**"If I make this sound here, what should I expect to hear back?"**

### Why it matters

This is where the cerebellum becomes more than a static predictor. It becomes an adaptive forward model shaped by the room.

The current babbling scripts are already pointed at this problem. That is the correct direction.

### Required capabilities

- Repeat controlled self-generated sounds across multiple trials
- Update the forward model from prediction error
- Measure whether prediction improves over time
- Feed improved prediction back into attribution

### Success criteria

- Prediction error decreases over repeated babbling iterations
- Attribution improves after adaptation
- Convergence is reproducible in the same room/device setup
- Performance drop from room changes is measurable, not mysterious

### Recommended experiments

1. **Fixed-room convergence test**
   - Same room, same devices, repeated babbling
   - Measure convergence over N iterations

2. **Room-shift sensitivity test**
   - Learn in room A, evaluate in room B
   - Quantify degradation and re-adaptation time

3. **Volume sensitivity test**
   - Vary playback amplitude
   - Measure whether attribution remains stable

4. **Stimulus-class generalization test**
   - Train on tones/clicks
   - Evaluate on vowels/babble
   - Then invert

## M1c: Mixed-Source Attribution

### Goal

Handle self-only, world-only, and overlapping sound sources.

### Guiding question

**"Which parts of this acoustic scene are me, which are world, and which are mixed?"**

### Required capabilities

- Label frames or chunks as one of:
  - self
  - world
  - overlap
  - silence/background
- Maintain low false self-attribution rate
- Remain useful under partial overlap and near-simultaneous onset

### Success criteria

- Strong frame-level or chunk-level attribution metrics
- Low false self-attribution rate
- Accurate overlap labeling under structured interference

### Recommended experiments

1. **Sequential self/world test**
   - world sound before self emission
   - self before world
   - measure temporal boundary accuracy

2. **Partial overlap test**
   - world begins during self emission or vice versa
   - score overlap labeling

3. **Confounding-source test**
   - introduce external sound with similar spectral shape to self stimulus
   - measure whether the model overclaims self authorship

4. **Playback-of-self negative-control**
   - replay a prior self sound without issuing a current motor command
   - ensure the system does **not** treat it as current self merely because it sounds similar

## Proposed Metrics

## Primary M1a metrics

- **Self/not-self attribution accuracy**
- **Precision / recall / F1** for self attribution
- **Onset alignment error** in ms
- **Offset alignment error** in ms
- **False self-attribution rate** under silence / ambient conditions

## Primary M1b metrics

- **Prediction error** over iteration
- **Convergence slope**
- **Attribution improvement pre/post adaptation**
- **Room-transfer degradation**

## Primary M1c metrics

- **Frame-level attribution F1** for self / world / overlap
- **Chunk-level attribution accuracy**
- **Confusion matrix**, especially:
  - overlap → self
  - world → self
  - self → world

## Secondary metrics

- latency from sound onset to attribution decision
- robustness across speaker volume and mic gain
- robustness across rooms and devices
- calibration drift over time

## Metrics to defer until later

These should not be the primary gate for early M1:

- full-word WER
- sentence transcription quality during duplex overlap
- lexical self-suppression benchmarks

Those belong later, once attribution is stable.

## Suggested Implementation Order

## Phase 1: Tighten the primitive event loop

Use the existing codebase to make a clean primitive pipeline:

1. `Broca` or signal generator emits simple sound
2. `Cochlea` captures mic return
3. `Cerebellum` predicts expected observation
4. `Thalamus` produces attribution decision
5. evaluation harness scores outcome

### Deliverables

- a simple sound generator utility for tones/clicks/vowels
- timestamped capture and alignment utilities
- attribution decision object with confidence and timing
- first reproducible local eval script

## Phase 2: Make babbling the default developmental protocol

Promote current babbling work from exploratory script to milestone driver.

### Deliverables

- standard babbling corpus
- convergence plots
- saved calibration artifacts per room/device
- repeatable benchmark script for convergence

## Phase 3: Introduce structured world interference

Add controlled non-self sources instead of jumping straight to open-ended conversation.

### Deliverables

- mixed-source test harness
- overlap labeling format
- benchmark scenarios for self/world/overlap

## Phase 4: Only then reintroduce speech content

Once primitive attribution is stable, begin testing:

- short syllables
- short words
- then phrases
- then real duplex speech

This keeps language from obscuring the more primitive attribution problem.

## Recommended Repo Additions

Suggested new files:

```text
docs/
  m1_source_attribution_plan.md
  m1_experiment_matrix.md
scripts/
  m1_tone_attribution.py
  m1_click_attribution.py
  m1_mixed_source_eval.py
  m1_room_transfer_eval.py
tests/
  eval/
    test_m1_self_attribution.py
    test_m1_babbling_convergence.py
    test_m1_mixed_source.py
```

## Experiment Matrix Sketch

A useful next companion doc would define experiments by axes:

- stimulus type
- room
- device pair
- playback volume
- interference type
- overlap condition
- expected label set

That would make M1 evaluation more systematic and less anecdotal.

## Failure Modes to Watch Closely

- **False self-attribution from spectral similarity**
  - external sound resembles self-generated sound
- **Echo mistaken for new self action**
  - delayed room reflections treated as separate self event
- **Replay confusion**
  - prior self audio replayed externally is mistaken for current self
- **Timing drift**
  - attribution depends on poorly aligned onset timestamps
- **Overfitting to one room**
  - model looks good only in the training environment
- **ASR contamination of evaluation**
  - word-level metrics mask the real attribution problem

## Design Stance

The thalamus should be judged first as a **source-attribution gate**, not as a word-suppression filter.

The STG should not dominate early M1 evaluation. At this stage, language is downstream.

The developmental order should be:

1. sound event detection
2. self-causation attribution
3. room-model adaptation
4. mixed-source separation
5. speech-aware suppression
6. language understanding under overlap

That sequence better matches both the infant analogy and the engineering dependency chain.

## Immediate Next Recommendation

The next practical step should be to create a minimal, reproducible **M1a attribution harness** using simple stimuli only.

If I were ordering the next implementation work, I would do this:

1. build tone/click attribution script
2. add silence and ambient negative controls
3. standardize output metrics and plots
4. promote babbling convergence into a formal eval
5. add mixed-source harness
6. only then revisit word-level suppression targets

## Closing

This reframing is not a retreat from the original goal. It is a correction to milestone order.

Corollary still aims at self-listening speech, memory, and introspection. But to get there cleanly, it should first solve the more primitive problem of **acoustic selfhood**.

Before it can know what it said, it should know whether it said anything at all.
