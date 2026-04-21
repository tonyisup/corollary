# Adaptive Attribution Calibration

> Purpose: define the first bounded self-learning layer for Corollary.

## Core idea

Corollary should begin self-learning by improving how it interprets its own low-level attribution evidence, not by attempting open-ended policy learning or unconstrained self-modification.

The first self-learning milestone is:

> **Corollary improves attribution decisions by calibrating against its own labeled experimental history.**

This keeps learning grounded in structured experiments and explicit safety constraints.

## Why this is the right first learning layer

Corollary already has meaningful low-level signals:
- self-template match
- world-template match
- local energy in expected windows
- onset timing
- residual energy after predicted-self subtraction
- later, directional asymmetry across channels

What is still unstable is the mapping from those signals to decisions such as:
- self
- world
- overlap
- silence/background

That mapping is the correct first target for adaptation.

## Boundary: what stays hand-authored

These should remain explicit and human-authored for now:
- milestone order (`M1a -> M1b -> M1c -> M2 ...`)
- safety constraints
  - replayed-self must not be promoted to current self
  - false-self rate remains a hard requirement
- experiment structure
  - world-first
  - self-first
  - partial overlap
  - replay negative control
  - confounding-source challenge
- feature definitions
  - self score
  - world score
  - local RMS / evidence deltas
  - onset error
  - residual energy
  - later directional asymmetry metrics

In short:
- humans define what evidence exists
- humans define what failure is unacceptable

## Boundary: what becomes learned

The first adaptive layer should learn how to combine evidence into decisions.

Examples:
- thresholds for `world_score`
- thresholds for `world_evidence_db`
- weighting of:
  - self score
  - world score
  - residual energy
  - onset timing
  - later binaural asymmetry
- per-setup calibration across:
  - room
  - device pair
  - geometry label

In short:
- do not learn the ontology yet
- learn the decision surface

## Training data for the first loop

Only use labeled, structured runs at first.

Recommended sources:
- sequential world-before-self
- sequential self-before-world
- partial overlap
- replay negative control
- confounding-source challenge

Each example should log at least:
- setup label
- date/time
- input/output device mapping
- stimulus ids
- self score
- world score
- local self/world/baseline RMS
- energy deltas
- onset deltas
- residual metrics
- expected label / ground-truth condition

This provides a supervised calibration loop rather than an open-ended self-editing loop.

## What not to learn from yet

Do not use these for the first self-learning layer:
- freeform transcripts
- open-ended chat outcomes
- long-term memory writes
- arbitrary code edits
- unconstrained policy changes

Those should come later, after low-level attribution is stable.

## Candidate model forms

The first learned model can stay small and interpretable.

Good candidates:
- bounded threshold search
- logistic regression
- tiny decision tree
- weighted evidence fusion optimized by autoresearch

Example input features:
- `self_score`
- `world_score`
- `self_evidence_db`
- `world_evidence_db`
- `onset_error_ms`
- `residual_ratio_db`
- later: `channel_asymmetry`

Example outputs:
- `self`
- `world`
- `overlap`
- `confidence`

## Objective function

The adaptive loop should optimize a bounded objective such as:
- improve world-first detection accuracy
- improve overlap attribution quality
- preserve low replay false-self rate
- penalize overfitting to one geometry or one device configuration

It should not optimize a single scalar threshold in isolation if the evidence is clearly multi-dimensional.

## Safety and evaluation constraints

The adaptive loop must never silently override hard constraints.

Hard constraints should include:
- replay false-self remains bounded
- world-to-self confusion must not regress past acceptable limits
- calibration changes must be evaluated on held-out conditions before promotion

Preferred evaluation split:
- train/calibrate on a subset of labeled runs
- validate on held-out runs from the same setup family
- retest on a different geometry or room label before promoting a new calibration profile

## Practical milestone framing

A good first milestone for this adaptive layer is:

### M1c-Cal: Adaptive Attribution Calibration

**Goal**
- improve self/world/overlap decisions by learning how to interpret low-level evidence from structured runs

**Acceptance style**
- better world-first detection than hand-tuned thresholds
- no regression on replay negative control
- clearer decision behavior across repeated runs in the same labeled setup

## Recommended next implementation step

1. define a compact per-run logging schema for M1c
2. gather a small labeled dataset from current world-first, self-first, replay, and overlap runs
3. fit a tiny interpretable calibration rule
4. compare that rule against current hand-tuned thresholds on held-out runs

That is enough to count as real self-learning while staying grounded, measurable, and safe.
