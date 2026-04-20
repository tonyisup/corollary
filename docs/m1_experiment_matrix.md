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
- sustained fixed vowel like `aa` or `oo`
- no interference

**Expected label**
- self

**Metrics**
- self accuracy
- alignment quality
- consistency across vowel types

**Note**
- treat the fixed vowel as the calibration probe and the later variable vowel family as the developmental robustness probe

## M1a-F: Variable vowel-family attribution

**Purpose**
- Test whether Corollary can recognize self-caused vocal output across small natural variation rather than one exact canned waveform.

**Setup**
- quiet room
- seeded `vowel_ah` family with bounded jitter in duration, amplitude, pitch, and envelope
- no interference

**Expected label**
- self

**Why this matters**
- infant-like babbling is not waveform-identical repetition
- success here would show robustness to self-variation rather than template memorization
- this should come after the fixed `vowel_ah` baseline, not replace it
- current repo evidence now promotes this family to the leading hardware-facing developmental probe on the Mac mini + Fifine path

**Recommended first jitter bounds**
- duration: roughly 220 to 420 ms
- amplitude: small gain jitter only
- pitch: small bounded fundamental shift
- envelope: slight onset/decay softness variation

**Logging requirements**
- family name
- random seed
- realized duration
- realized amplitude
- realized pitch / fundamental estimate
- realized envelope parameters

## M1a-G: Variable-vowel mild-noise validation

**Purpose**
- Test whether the variable vowel family remains the better self-attribution probe when the room is not pristinely quiet.

**Setup**
- same close hardware baseline
- mild ambient noise / ordinary room state
- compare fixed `vowel_ah` against seeded `vowel_ah_var`
- log setup label explicitly

**Expected label**
- self

**Pass signal**
- `vowel_ah_var` remains at least as stable as fixed `vowel_ah`
- variable family keeps better match quality under noise
- seed-to-seed degradation stays bounded rather than collapsing unpredictably

**Current repo observation**
- on the current Mac mini + Fifine setup, `vowel_ah_var` remained `good` across multiple seeds under mild noise and beat fixed `vowel_ah` head-to-head, promoting it from comparison stimulus to first-class hardware probe family
- later parameter-aware bridge analysis refined that family into `vowel_ah_var_nearfield`, which now looks like the stronger default for adaptation-oriented hardware work on this machine

## M1a-H: Babble syllable attribution

**Purpose**
- Test pre-linguistic speech-like sounds without semantic pressure.

**Setup**
- quiet room
- `ba`, `da`, `ma`, repeated or isolated
- no interference

**Expected label**
- self

## M1a-H: Silence negative control

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

## M1b-A: Variable-vowel adaptation bridge

**Purpose**
- Bridge M1a into M1b using repeated bounded self-like vowel families before broader syllabic babble.

**Setup**
- fixed close hardware setup with explicit setup label
- repeated variable-vowel emissions across a small seed set
- adaptive forward model update after each capture

**Metrics**
- prediction error by iteration
- attribution quality across seeds before vs after adaptation
- whether adaptation on one small seed set improves neighboring seeds

**Pass signal**
- prediction error decreases across repeated variable-vowel trials
- self-attribution remains strong while the forward model learns across natural self-variation
- adaptation generalizes across nearby seeds better than a model trained on one rigid fixed vowel alone

**Why this matters**
- this is the cleanest developmental bridge from primitive self-attribution into room learning
- it pressures the cerebellum to learn a class of self-generated sounds, not one exact utterance template

**Current repo observation**
- the broad family `vowel_ah_var` made the bridge viable on real hardware
- the constrained subfamily `vowel_ah_var_nearfield` then won a direct bridge-vs-bridge comparison by making both training behavior and held-out generalization dramatically more consistent
- cross-room validation now supports keeping that ranking: the family transferred across a noisier second room, survived a movie-background stress test as a robustness condition, produced the cleanest current benchmark in `room3_close_quiet_varbridge_v1` with held-out SVSR `39.49 dB` and `39.98 dB`, and then survived an explicit live room move with only about `0.10 dB` immediate held-out SVSR loss
- an exploratory retune branch, `vowel_ah_var_nearfield_room2`, did not materially improve room-2 held-out bridge performance and should remain a non-promoted branch
- the short post-transfer re-adaptation pass in the live move experiment made held-out performance worse rather than better, suggesting that automatic rebabbling after moderate room shift may over-specialize instead of help
- the bridge is no longer vowel-only: `babble_ba_var`, `babble_da_var`, and `babble_ma_var` all produced strong quiet-room gated bridge results, with `ba` strongest/cleanest, `da` somewhat rougher, and `ma` onset-messier but still strongly generalizing
- a direct replicated room-3 comparison against `vowel_ah_var_nearfield` now supports a stronger but still disciplined interpretation: the near-field vowel family remains an established bridge baseline, and `babble_ba_var` now matches it closely enough to be promoted to co-baseline status
- subsequent replication sharpens the babble picture into a tiered family structure rather than a flat class: `babble_da_var` and `babble_ma_var` remain genuinely strong peer families, but they sit below the co-baseline band and behave more like challenge families than default baselines
- this now makes `vowel_ah_var_nearfield` and `babble_ba_var` the leading co-baseline adaptation families on the current setup, with `room3_close_quiet` as the preferred benchmark condition and `da` / `ma` serving as secondary peer families for harder follow-up transfer questions

## M1b-B: Fixed-room convergence test

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
- keep room labels explicit, for example quiet benchmark room versus noisy or media-on stress room

**Metrics**
- attribution drop on transfer
- held-out SVSR drop on transfer
- iterations needed to recover
- whether media/noise conditions behave like robustness stressors or invalidate ranking entirely

**Current repo lesson**
- ordinary room changes are informative and can preserve the near-field family ranking
- active media in the background is a stronger confound and should be treated as a separate stress-test condition, not the main benchmark for promotion decisions
- moderate live room moves may impose very small immediate held-out cost, so the next evaluation question should be when re-adaptation is actually justified rather than assuming rebabbling is always needed
- first trust-gating evidence now supports a simple policy: small held-out drops should skip re-adaptation, while larger drops, like the hallway shift, justify rebabbling because short re-adaptation can recover part of the loss
- matched `room3_close_quiet` → `room4_transfer_target` tests now add a second-axis lesson: quiet-room bridge quality and transfer robustness are not identical rankings. `vowel_ah_var_nearfield` and `babble_ba_var` transfer at parity, and both `babble_da_var` and `babble_ma_var` also transfer surprisingly well despite sitting below the co-baseline band in quiet-room held-out quality

**Stress-condition protocol (new rule)**
- Do not treat labels like `hallway`, `room4`, or `transfer_target` as sufficient descriptions of difficulty by themselves.
- For any experiment intended as a *hard* transfer or recovery condition, record at least:
  - room label
  - mic placement relative to speaker
  - mic orientation / rotation
  - speaker orientation
  - approximate distance to nearest reflective surface or wall
  - whether nearby human speech, media, or other intermittent world audio was present
- Use an explicit setup label that encodes the geometry, not just the room, for example `hallway_open_side_rotated`, `room4_wall_close`, or `hallway_far_wall_oblique`.
- Before promoting a setup as a real stress condition, require one calibration transfer run that actually produces a meaningful immediate held-out drop, for example around or above the current `1.0 dB` trust-gate threshold.
- If a supposed stress setup does not produce that drop, treat it as an ordinary transfer condition, not a failed hard-condition run.
- Once a geometry does produce a real drop, reuse that exact geometry label for matched family comparisons instead of re-creating the condition loosely from memory.
- Keep a separate label family for contaminated-learning cases where the world source is intentionally near-field and strong enough to corrupt adaptation, for example `phone_music_max_volume_directly_between`, rather than mixing them into ordinary room-transfer labels.

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
- compare within the new early babble class, for example `ba` vs `da` vs `ma`
- compare isolated babble tokens against short repeated trains, for example a single `ba` versus `babababa`

**Metrics**
- post-adaptation attribution by unseen stimulus type
- whether trust-gated stopping still behaves sensibly across different syllabic babble families
- whether repeated self-generated sequences preserve attribution better than isolated tokens under controlled interference

**Current repo lesson**
- the same trust-gated bridge logic now survives a first small syllabic repertoire (`ba`, `da`, `ma`) in the quiet benchmark room, which is the first evidence that the M1 bridge generalizes beyond sustained vowels into a real early babble class
- matched room-3 comparison now suggests that this babble class is not just viable but competitive with the established near-field vowel bridge; after replication, the right current claim is co-baseline status for `babble_ba_var`, not outright replacement of the vowel family
- further replication shows the babble class is structured: `ba` forms the first babble co-baseline, while `da` and `ma` remain strong but more fragile peer families that sometimes approach the trust-gate continuation threshold more closely than vowel or `ba`
- transfer replication now supports a fuller two-axis interpretation: `da` and `ma` are weaker than `ba` in quiet-room bridge ranking, but both still survive the `room3_close_quiet` → `room4_transfer_target` move without triggering re-adaptation, so transfer-friendliness appears at least partly decoupled from quiet-room held-out ranking
- Stress testing in hallway environments with TV interference confirms that qualitative room labels are insufficient predictors of model stability. Data suggests that the geometric orientation of the cardioid mic pattern relative to noise sources is a more significant variable than TV amplitude alone. Under these conditions, specific geometry (e.g., `hallway_transfer_tv_facing`) produced a moderate transfer drop of about `0.62 dB`, which is measurable but still below the `1.0 dB` operational threshold.
- Stronger near-field media contamination can behave very differently from ordinary hallway transfer. Manual runs in `data/stress_test/` showed phone-music interference crossing the gate decisively: `trial_1-phone_music.json` produced an immediate drop of about `2.00 dB`, while `trial_2-phone_music_max_volume.json` and `trial_3-phone_music_max_volume_directly_between.json` produced catastrophic immediate drops of about `22.77 dB` and `21.87 dB` respectively.
- Those same phone-music runs also showed that forced re-adaptation under ongoing heavy interference can further degrade held-out performance instead of recovering it. Current interpretation: extreme media-on contamination should be tracked as a mixed-source / contaminated-learning stress class, not treated as an ordinary room-transfer benchmark.
- Follow-up `babble_ba_var` replications showed that even the moderate `phone_music_moderate` setup is not yet reproducible enough for clean cross-family promotion decisions. Across recent runs, immediate drops ranged from about `0.21 dB` to `7.57 dB`, so the setup itself is still wobbling too much to support a stable claim that `ba` is more robust than the vowel family under phone-music interference.
- Under that same moderate phone-music condition, room-B re-adaptation remained unreliable for `babble_ba_var`: some runs degraded further after readaptation, one recovered only modestly, and none supported blind rebabbling as a safe default under contamination.
- Current operational policy should therefore split more sharply by regime: ordinary room transfer remains trust-gated M1b work, while media-contaminated conditions should be evaluate-first and generally no-learn unless the explicit purpose is to study mixed-source attribution failure or contaminated-learning safety.
- A new developmental hypothesis now emerges from the contamination results: repeated babble trains may be a more natural and more robust self-attribution strategy than isolated single tokens in noisy environments. The motivation is temporal redundancy, stronger self-consistency, and better averaging against irregular world audio.
- Immediate next experiment for that hypothesis: compare single-token `babble_ba_var` probes against short repeated `babababa`-style trains under a tightened `phone_music_moderate` protocol before making broader claims about babble-family robustness.

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