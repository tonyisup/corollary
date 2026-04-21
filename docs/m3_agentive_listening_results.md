# M3 Agentive Listening Results: Social Contingency

> **Status:** Active Research
> **Date:** 2026-04-20
> **Scope:** M3 Agentive Listening and Proto-Turn-Taking (Hardware)

## Overview

M3 transitions Corollary from a passive listener to an active participant in vocal exchange. The core goal is to distinguish "Agentive Others" from background noise and respond with appropriate social timing.

## Key Technical Achievements

### 1. PFC Agentive Controller
We implemented the `AgentiveController` (`packages/pfc/agentive_controller.py`) to manage social state:
- **IDLE**: Ambient monitoring.
- **LISTEN**: Actively hearing a world-agent. Triggered after 300ms of sustained world-speech.
- **SPEAK**: Actively emitting self-audio.
- **WAIT**: Deferring a pending motor command because the world-agent is still talking.

### 2. Turn-Taking Policy
The controller enforces a "Contingent Listening" policy:
- **Non-Interruption**: If the PFC requests to speak while in the `LISTEN` state, the utterance is deferred.
- **Automatic Trigger**: Once the world-agent has been silent for a cooldown period (800ms), the deferred utterance is automatically executed.

### 3. Hardware Demo (M3-A)
We created `scripts/m3_agentive_listening_demo.py` to showcase these behaviors on real hardware.
- **Result**: Corollary successfully monitors the room and correctly transitions between listening and speaking states. It demonstrates the ability to "hold its breath" until the room is quiet.

## Hardware State Transitions (M3)

| Event | Previous State | New State | Action |
| :--- | :--- | :--- | :--- |
| sustained world-speech | `IDLE` | `LISTEN` | Suppress self-output |
| silence > 800ms | `LISTEN` | `IDLE` | Trigger deferred output |
| PFC speech request | `IDLE` | `SPEAK` | Play audio immediately |
| PFC speech request | `LISTEN` | `WAIT` | Queue utterance for later |

## Key Findings

1.  **Hysteresis is Essential**: Raw VAD is too jittery for social state. The `AgentiveController`'s 300ms/800ms timing windows provide the necessary stability for human-like turn-taking.
2.  **Motor-Aware Gating**: By combining the `ThalamicAttributor` (which knows if *we* are talking) with the `VAD` (which knows if *someone* is talking), we can definitively identify external human agents.

## Next Steps

1.  **Mimicry Primitives (M3-B)**: Implement "Echoic" responses where Corollary repeats back a syllable it heard, bridging raw vocal learning and dialog.
2.  **Self-Voice Detector (M4)**: Use embeddings to distinguish "My Voice" from "Your Voice" even when spectra are similar, increasing confidence in high-noise environments.
