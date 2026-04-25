# Neural Directional Filtering Paper Notes for Corollary (2026-04-24)

> Paper discussed: `https://arxiv.org/html/2511.07185v4`
>
> Working summary: compact microphone arrays can use a learned directional filter to approximate steerable virtual microphone patterns more flexibly than classic compact-array beamforming.

## Why this paper matters for Corollary

Corollary is currently struggling with a problem that this paper is directly adjacent to:
- small microphone setups
- imperfect physical directivity
- strong cross-coupling between self and world sources
- wanting cleaner self-vs-world separation without needing a large array aperture

The paper is relevant because it suggests a future path in which Corollary can use:
- a compact multi-mic setup
- learned spatial filtering
- steerable virtual directivity

to improve separation beyond what raw hardware geometry alone provides.

## The paper's core idea, translated for Corollary

Instead of relying only on:
- raw microphone directivity
- classic fixed beamforming
- or fragile explicit spatial parameter estimation

train a model that takes multichannel audio and produces a complex mask that turns a reference channel into a **virtual directional microphone**.

Corollary interpretation:
- one virtual output could be steered toward the self speaker / self mouth region
- another virtual output could be steered toward the external world source region
- attribution could operate on those virtual directional channels rather than only on raw mic channels

## What seems applicable now

### 1. Directionality should be treated as a first-class design target

The paper reinforces a design lesson Corollary has already been learning empirically:

> the question is not just "what do the raw mics hear?" but "what pickup pattern do we want?"

This is useful immediately, even before any learned directional filter exists.

Practical implication for current work:
- evaluate geometry in terms of desired pickup asymmetry
- explicitly seek self-facing vs world-facing behavior
- treat dual-mic layout as the beginning of a controllable directional frontend

### 2. Compact arrays may still be worth it even if naive geometry disappoints

The single-mic and early dual-mic calibration work has shown that raw compact geometry can easily land in cross-coupled regimes.

The paper suggests that this does not automatically mean the compact setup is doomed. It may mean:
- raw channels are not yet sufficient
- but they could become more useful once processed jointly through a learned directional model

That is encouraging for Corollary's broader direction.

### 3. Steering toward self vs world is a natural abstraction for Corollary

The paper's steerable direction-conditioning maps cleanly onto a future Corollary framing:
- self-steered virtual microphone
- world-steered virtual microphone

This is a better conceptual fit for Corollary than generic "speech enhancement" because the project's problem is fundamentally about:
- source ownership
- attribution
- self/world discrimination

## What should wait

### 1. Do not jump straight into reproducing the paper now

Corollary is not ready yet for a faithful implementation of this idea.

Reasons:
- the current dual-mic hardware geometry is still being stabilized
- calibration is still noisy and setup-sensitive
- there is not yet a clean reproducible multi-mic dataset pipeline for training/evaluation
- the project still needs clearer baseline self/world separation with simple sounds

So this paper should not displace the current immediate work.

### 2. Do not mistake this for a replacement for directional hardware

The paper does not remove the value of better physical capture.

For Corollary, that means:
- a cardioid mic can still matter a lot
- better geometry still matters
- cleaner raw channel asymmetry will likely make any learned directional layer work better later

Best interpretation:
- learned directional filtering should be viewed as **augmenting** better hardware, not replacing the need for it

## How this maps onto the M1 roadmap

## Current / near-term

### M1c (current practical focus)
Goal:
- achieve cleaner self/world separation with simple sounds using real hardware

Paper relevance now:
- conceptual only
- reinforces the need to think in terms of pickup asymmetry and directional ownership
- suggests that dual-mic experimentation is worth continuing even if raw geometry is imperfect

### Immediate next step after current checkpoint
Goal:
- get a better dual-mic baseline using simple sounds
- likely with improved world-side microphone directionality (e.g. cardioid mic)

Paper relevance now:
- motivational / architectural
- not implementation-ready

## Possible later step

### Candidate future milestone: M1d or M1e — Learned Spatial Frontend

Possible goal:
- transform a compact multi-mic capture into steerable self-facing and world-facing virtual channels before attribution

Possible structure:
1. record synchronized multi-mic data from known geometry
2. define target steering directions / target pickup behavior
3. learn a directional filter that outputs a self-steered and/or world-steered virtual channel
4. feed those outputs into attribution rather than relying only on raw mic channels

This would be a natural place to borrow from the paper.

## What to borrow conceptually even without reproducing the full method

### 1. Reference-channel + learned mask framing

A simple future starting point for Corollary could be:
- choose one mic as reference
- use multichannel input to predict a mask
- apply it to the reference channel to form a virtual directional output

This is simpler than trying to build a full custom beamformer stack from scratch.

### 2. Evaluation should include directional behavior, not only task outputs

The paper emphasizes directivity-pattern behavior and directional suppression, not only downstream audio quality.

Corollary should carry that lesson forward.

Meaning:
- evaluate whether self-steered output prefers self
- evaluate whether world-steered output prefers world
- preserve geometry-aware metrics, not just ASR or attribution verdicts

### 3. Frequency-stable directional behavior matters

The paper's claim that learned directional patterns can be more frequency-stable than naive compact-array methods is especially relevant for Corollary because the project needs:
- simple sounds now
- speech later
- ideally one frontend that does not collapse when the spectrum changes

## Risks / cautions for Corollary

### 1. Data hunger and setup sensitivity

A learned spatial frontend will likely require:
- consistent multi-mic synchronization
- careful geometry documentation
- enough training/evaluation data across conditions

This is nontrivial.

### 2. Easy to overcomplicate too early

Corollary should not jump from unstable raw geometry directly into a heavy learned directional system without first demonstrating:
- reproducible baseline capture
- stable hardware mapping
- basic separation value from better dual-mic setup

### 3. Deep nulls may still be hard

The paper itself reportedly notes limits on deep null suppression in practice.

So even if Corollary adopts a similar direction later, expectations should stay realistic:
- improvement is plausible
- perfect cancellation is not guaranteed

## Current recommendation

### Use the paper as a future architectural direction, not an immediate implementation task

That means:
- continue current simple-sound dual-mic work
- improve raw physical separation first
- adopt better world-side mic directionality if available
- keep this paper in mind as the next layer once the hardware baseline is credible

## Best one-line takeaway for Corollary

> This paper is good evidence that compact multi-mic Corollary does not have to stop at raw geometry; once the dual-mic setup is stable, a learned steerable spatial frontend could become a strong next-stage path for self/world separation.

## Suggested future revisit trigger

Revisit this paper seriously when Corollary can reliably show:
- a stable dual-mic setup
- repeatable self/world asymmetry on simple sounds
- synchronized multi-mic logging suitable for training/evaluation

That is the right moment to ask whether Corollary should prototype a learned directional frontend inspired by this work.
