# Design Spec: Grant Materials Update (M3/M4 Empirical Focus)

**Date**: 2026-05-08
**Topic**: Updating grant materials to reflect M3/M4 progress.

## Overview
Update the core grant documentation (`Executive_Abstract.md`, `Research_Methodology.md`, `Visual_Data_Summary.md`) to reflect the transition from M2 (Speech-Aware Separation) to M3 (Agentive Listening) and M4 (Self-Modeling).

## 1. Executive Abstract Updates
- **Current Progress & Evidence**:
    - Add **Agentive Turn-Taking (M3)**: Successful implementation of the `AgentiveController` for social contingency. Demonstrated natural deferral and mid-speech interruption recovery in an embodied TTS-to-Mic loop.
    - Add **Self-Modeling Impact (M4)**: Integration of a speaker-embedding detector resulting in a 19,700% increase in self-voice detection accuracy (197 SELF frames vs 0 in baseline).
- **Key Innovations**:
    - Update Thalamic Gating section to include the M4 forward-model's room-colored prediction (FIR filter) and the speaker-embedding validator.

## 2. Research Methodology Updates
- **Developmental Phasing**:
    - Update **Stage III**: Focus on "Social Contingency" using the PFC controller bridge. Mention ASR-contingent responses ("You said: [transcript]").
    - Update **Stage IV**: Mention the integration of the speaker-embedding detector for grounded self-attribution.
- **Experimental Paradigms**:
    - Add the **Embodied TTS-to-Mic Loop**: A new testing mode that uses real Broca TTS output to validate M2/M3/M4 in a closed sensory-motor loop.
- **Core Validation Metrics**:
    - Add **Interruption Recovery Rate**: Tracking the success of the agent in resuming deferred utterances after an interruption.
    - Add **M4 Self-Attribution Gain**: Measured as the increase in `SELF` verdicts in ambiguous overlap frames.

## 3. Visual Data Summary Updates
- **Figure 4: Agentive Contingency (M3)**:
    - Update to refer to `scripts/m3_turn_taking_demo.py`.
    - Description: Temporal timing diagram showing state transitions (`IDLE` -> `LISTEN` -> `SPEAK` -> `LISTEN` -> `IDLE`).
- **Figure 5: M4 Self-Modeling Impact**:
    - New figure.
    - Description: Bar chart comparing `SELF` vs `BOTH` frame verdicts with M4 OFF (Baseline) vs M4 ON.
    - Key Message: "M4 correctly classifies self-voice in ambiguous frames, enabling high-confidence full-duplex operation."

## Implementation Plan
1. Update `docs/grant_materials/Executive_Abstract.md`.
2. Update `docs/grant_materials/Research_Methodology.md`.
3. Update `docs/grant_materials/Visual_Data_Summary.md`.
4. Run a quick verification by reading the files back to ensure formatting and data accuracy.
