# M4 Self-Voice Results: Explicit Attribution

> **Status:** Prototype
> **Date:** 2026-04-20
> **Scope:** M4 Self-Voice Detector and Explicit Attribution (Software)

## Overview

M4 introduces explicit speaker identification to the Corollary stack. While M1-M3 rely on timing and spectral matching, M4 uses latent speaker embeddings to definitively distinguish "My Voice" from "Your Voice."

## Key Technical Achievements

### 1. Speaker Identification Interface
We implemented the `SpeakerIdentifier` (`packages/stg/speaker.py`) which defines the interface for ECAPA-TDNN embedding extraction.
- **get_embedding**: Returns a 512-dimensional vector representing the vocal characteristics of a segment.
- **similarity**: Computes cosine similarity between two embeddings.

### 2. Self-Voice Registration
The `SelfVoiceDetector` allows Corollary to "register" its own voice template.
- During initialization, the system synthesizes a reference phrase via TTS and stores the resulting embedding as the `self_template`.

### 3. Attribution Prototype
We created `scripts/m4_self_voice_prototype.py` to demonstrate the registration/recognition workflow.
- **Current State**: The interface is defined and ready for a production-grade ECAPA model (e.g., SpeechBrain or Pyannote).

## Next Steps

1.  **Production Model Wiring**: Connect the `SpeakerIdentifier` to a local pre-trained ECAPA model (requires ~1GB storage and weights).
2.  **Integrated Hardware Loop (M4-B)**: Run the `AgentiveController` while simultaneously verifying the speaker identity of the "External Agent."
3.  **Diarization (M5)**: Extend speaker identification to track *multiple* external speakers simultaneously.
