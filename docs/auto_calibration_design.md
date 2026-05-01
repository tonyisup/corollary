# Automatic Room-Adaptive Calibration Trigger

> Date: 2026-05-01
> Status: Design note — not yet implemented
> Context: Discussion of M2 benchmark calibration portability across room changes

## Problem

The M2 benchmark calibration establishes a self/world signal ratio (S/W) at the
microphone that is specific to the current room geometry. When the room changes
(speaker/headphone positions, room acoustics, noise floor), the calibration no
longer holds. The current workflow requires manual re-calibration via the full
amplitude sweep.

## Biological Analogue

The biological system solves this with three layered mechanisms:

1. **Lombard effect** — brainstem reflex that increases vocal effort when the
   noise floor rises. Automatic, ~100ms latency, no conscious involvement.

2. **Sidetone-based self-monitoring** — the auditory cortex compares the
   expected self-sound (from the efference copy / cerebellar forward model)
   with the actual received sound. Mismatch triggers rapid motor adjustment.

3. **Room-adaptive vocal projection** — walking from outdoors into a cathedral
   triggers automatic adjustment within a few syllables. This is a cerebellar
   forward-model update, not a PFC decision. The PFC handles *social*
   modulation (speaking quietly in a library), not acoustic adaptation.

## Proposed Architecture

```
Room change detected
  → Emit low-amplitude self probe (chirp or short babble, ~50ms)
  → Measure captured self signal
  → Compare against forward-model prediction
  → If prediction error exceeds threshold:
      → Run brief babble adaptation (M1b-style, ~5 seconds)
      → Update forward model weights
      → Re-establish S/W operating point
      → Resume normal operation
  → If adaptation fails (error persists):
      → Escalate to PFC (full recalibration or user alert)
```

## Trigger Conditions

The calibration probe should be triggered by any of:

1. **Prediction error spike** — cerebellar forward model's prediction error
   exceeds a threshold. The self-match score drops because the room-colored
   self no longer matches the current forward model's prediction.

2. **Periodic probing** — biological systems continuously emit small self-sounds
   (breathing, swallowing, subvocalization) that serve as ongoing calibration
   probes. Corollary could emit a low-amplitude chirp every N seconds during
   idle periods.

3. **Performance degradation** — ASR WER on world speech degrades beyond a
   threshold. The "I can't understand the other person" signal.

4. **Novelty detection** — the captured self-only signal during a babble probe
   doesn't match any stored room context (hippocampal mismatch).

## Implementation Notes

- The M1b babbling loop IS the calibration process. What's missing is the
  trigger that initiates it automatically.
- The probe should be short (~50ms) and low-amplitude to avoid disrupting
  ongoing conversation.
- The adaptation loop (if triggered) is a sub-second process that does NOT
  require PFC involvement. The PFC is only notified if adaptation fails.
- This maps onto existing Corollary modules:
  - Probe generation: Broca (chirp/babble stimulus)
  - Prediction comparison: Cerebellum (forward model error)
  - Adaptation: M1b babbling loop (update forward model)
  - Performance monitoring: STG (ASR WER tracking)
  - Escalation: PFC (full recalibration trigger)

## When to Implement

Not needed for current benchmark work. The current room is fixed, and the
manual calibration approach is sufficient for evaluating gating strategies.

Implement this when:
- Moving to a multi-room setup
- Building a product that needs to work in unmapped environments
- Testing Corollary in different physical locations
- Before the "move indoors" test scenario (dog barking vs human conversation
  adaptation)

## Open Questions

1. What's the right probe interval? Too frequent = annoying. Too rare = slow
   adaptation. Biological systems seem to probe continuously at low level.

2. What's the prediction error threshold? Too sensitive = constant recalibration
   in noisy environments. Too insensitive = won't adapt to real room changes.

3. Should the probe be perceptually masked? (e.g., emitted during a pause in
   world speech, or at a frequency the user won't notice)

4. How does this interact with the current M2 BOTH-frame policy? If the forward
   model updates, does the self-match threshold need to change too?
