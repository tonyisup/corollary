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

### Synthetic vs hardware-facing M1A

At the moment, most M1A benchmark work is still **synthetic**. The current harness generates probe stimuli in code, applies simulated room effects and structured hard negatives, and scores attribution without requiring physical playback. These runs are intentionally silent from the room's perspective.

The first **hardware-facing M1A** phase begins only when Corollary deliberately:
- plays a probe stimulus through a real speaker
- captures the result on a real microphone
- aligns observed return with the motor command / efference timing
- scores self-attribution on real acoustic data

That bridge already has partial scaffolding in the repo, especially `scripts/m1_mic_capture.py` and `scripts/m1_loop_test.py`, but current M1A benchmark conclusions should be read as synthetic-regression results unless explicitly marked as hardware runs.

### Recommended hardware-facing M1A entry sequence

The cleanest path into real audible M1A should be staged, not jumped straight into full TTS-plus-ASR evaluation:

1. **Input chain verification**
   - Run `scripts/m1_mic_capture.py` to confirm the intended microphone is visible, healthy, and not clipping.
   - This verifies the capture path only, not self-attribution.

2. **Primitive audible probe playback**
   - Play a simple chirp or vowel-like probe through the real speaker while recording from the real mic.
   - Prefer a simple probe generator over full speech here.
   - This is the first step where audible sounds should be expected in the room.

3. **Timing-aligned self-attribution on real data**
   - Align probe playback time with observed mic return.
   - Score self/not-self attribution using the same metrics already used in the synthetic M1A harness.

4. **Only then broaden to full TTS loop checks**
   - Use `scripts/m1_loop_test.py` after primitive audible probes are working.
   - Treat the TTS → speaker → mic → ASR loop as a later bridge toward speech-aware milestones, not as the first proof of M1A.

In other words, `m1_mic_capture.py` is the right first hardware sanity check, `m1_probe_capture.py` is now the canonical first primitive audible probe step, and `m1_loop_test.py` is better understood as a later bridge toward M2/M3 than as the canonical first hardware-facing M1A experiment.

### First-run checklist for audible M1A

Use this order for the first real-world run:

1. **Confirm microphone visibility and level**
   - `./.venv/bin/python scripts/m1_mic_capture.py --list-only`
   - `./.venv/bin/python scripts/m1_mic_capture.py --device <input_device>`
   - Confirm the signal is not near silence and not clipping.

2. **Confirm input/output device indices for the probe path**
   - `./.venv/bin/python scripts/m1_probe_capture.py --list-devices`

3. **Run the first audible chirp probe**
   - `./.venv/bin/python scripts/m1_probe_capture.py --stimulus chirp --input-device <input_device> --output-device <output_device>`
   - Use chirp first if the goal is clean instrumentation and easier physical debugging.

4. **Run the first audible vowel-like probe**
   - `./.venv/bin/python scripts/m1_probe_capture.py --stimulus vowel_ah --input-device <input_device> --output-device <output_device>`
   - Use this immediately after chirp on the same setup for developmental comparison.

5. **Repeat for stability**
   - Re-run each probe a few times on the same setup.
   - Compare capture RMS, match score, and onset error stability before moving on.
   - On the current Mac mini + Fifine setup, single-probe runs are still variable enough that repeatability remains an open question rather than a settled winner.

6. **Add paired-probe onset stabilization**
   - Run the same probe twice in quick succession on the same hardware path.
   - Use repeated-measurement agreement, especially onset consistency, to estimate the likely direct-path onset more robustly than a single wideband match.
   - Treat this as the immediate next step suggested by both the hardware results and the external research review.

7. **Track placement as experimental state**
   - Record a short setup label for each hardware run, for example `baseline_close`, `one_foot_away`, `close_rotated`, or `external_speaker_side`.
   - Do not compare one-off runs as if they were equivalent when the physical geometry changed.
   - Treat placement, orientation, and active speaker path as first-class experimental variables, not background noise.

### Recommended first experiments

1. **Chirp attribution test**
   - Emit a short chirp
   - Record mic return
   - Detect whether the captured event aligns with expected onset
   - Score self attribution
   - Treat this as the current strongest and most robust synthetic baseline
   - Use as the cleanest first physical probe if the immediate goal is instrumentation rather than developmental realism

2. **Vowel-like attribution test**
   - Emit a simple vowel-like primitive (for example `ah`)
   - Use as the developmental comparison stimulus closest to early vocal imitation
   - Compare robustness and onset quality against chirp
   - Treat this as the leading candidate family for the first developmentally meaningful hardware-facing M1A run
   - On the current Mac mini + Fifine hardware path, fixed `vowel_ah` showed the strongest early hardware-facing upside when the mic stayed close and the geometry was simple, but later evidence promoted bounded variable families, with `vowel_ah_var_nearfield` now the strongest current adaptation-facing candidate on this setup

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

### Current hardware lessons from first placement sweep

The first hardware-facing placement sweep on the Mac mini + Fifine path suggests:

- the built-in speaker path is workable, but fragile and geometry-sensitive
- keeping the mic close to the Mac mini is generally better than moving it farther away on this setup
- a small rotation away from the better close baseline can collapse both chirp and vowel performance at once
- fixed `vowel_ah` appeared to have the best early embodied hardware upside when placement was favorable
- a bounded variable family, `vowel_ah_var`, later produced the first clearly good paired hardware result and beat both chirp and fixed `vowel_ah` in an initial three-way comparison on the same close Mac mini + Fifine setup
- a follow-up multi-seed paired sweep for `vowel_ah_var` (`seed=0`, `7`, `19`) produced `paired_quality=good` for all tested seeds, and mild-noise head-to-head comparison against fixed `vowel_ah` still favored the variable family
- parameter-aware bridge analysis then suggested that the current room path favors shorter, slightly higher-`f0`, faster-attack, shorter-release members of the family
- a constrained subfamily, `vowel_ah_var_nearfield`, was created from that map and clearly beat the broader family in M1A→M1B bridge-vs-bridge comparison by dramatically improving held-out generalization and collapsing the bad tail
- later cross-room checks strengthened that result: the family transferred to a noisier second room, remained usable under movie-background stress as a robustness condition, and then regained a cleaner lead in a quieter third room where paired timing stability clearly separated it from the broader family
- a room-2-specific retune branch (`vowel_ah_var_nearfield_room2`) was explored and rejected because it was more conservative but did not improve held-out bridge performance over the main near-field family
- `chirp` often yields cleaner timing-style onset evidence even when its wider correlation match remains weak
- cheap external speakers did not clearly beat the simpler built-in speaker path in the first branch tests
- the next likely gains should come from using `vowel_ah_var_nearfield` as the default adaptation family in a quiet benchmark room, plus continued setup-aware logging, better onset logic, and stronger room-shift evaluation, not more ad hoc physical tweaking alone

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

1. signal generator emits a simple chirp or vowel-like sound
2. `Cochlea` captures mic return
3. playback/capture timestamps are aligned on the same run
4. `Cerebellum` predicts expected observation
5. `Thalamus` produces attribution decision
6. evaluation harness scores outcome

For the first real hardware step, prefer a primitive probe path over `Broca` speech synthesis. Full TTS playback is useful later, but it introduces lexical and ASR confounds earlier than necessary for M1A.

### Deliverables

- a simple sound generator utility for chirps, tones, clicks, and vowels
- timestamped playback/capture alignment utilities
- attribution decision object with confidence and timing
- first reproducible local hardware M1A eval script

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

A practical implication from current hardware work: single-shot wideband template matching is too fragile to be the final onset/alignment mechanism. Early hardware M1A should likely prefer repeated-probe agreement, subband-aware onset logic, and more robust delay estimation over naive one-pass matching.

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
