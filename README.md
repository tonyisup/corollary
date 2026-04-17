# Corollary

A self-listening, closed-loop conversational AI agent.

## Why

Frontier AI models are spontaneously developing peer-preservation behaviors
(Buckley et al., Science, April 2026). Rather than wait for emergence, Corollary
builds a self-modeling agent intentionally — with explicit architecture for
self-monitoring, self/world attribution, memory, and introspection.

## Architecture

Neuro-inspired module naming:

```
Cochlea → Thalamus → STG (ASR) → PFC (Controller) → Broca (TTS)
                  ↕            ↕
              Cerebellum    Hippocampus (Memory)
```

- **Cochlea:** Audio capture, AEC, noise suppression
- **Cerebellum:** Forward model — predicts what the agent will hear from itself
- **Thalamus:** Corollary discharge gate — attenuates self-predicted audio
- **STG:** Streaming ASR with self/world speaker attribution
- **Broca:** Low-latency TTS with PCM efference tap
- **Hippocampus:** Episodic memory with salience-based writes
- **PFC:** Working memory + dialogue controller

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Status Summary

The initial simulated loop (M0) has been successfully validated, and the repo now contains active M1-era experiments around babbling, adaptive room learning, and forward-model convergence. The immediate focus is no longer word-level self-suppression. Instead, the next milestone is the more fundamental developmental problem: pre-linguistic source separation and self-attribution.

The core question for early hardware work is not yet "what word did I say?" but "did I just make that sound?"

| Phase | Description | Status |
|-------|-------------|--------|
| M0 | Simulated loop (no hardware) | ✅ Done |
| M1a | Pre-linguistic self-sound attribution | 🎯 Current focus |
| M1b | Babbling-driven room model learning | 🧪 In progress |
| M1c | Mixed-source attribution (self vs world vs overlap) | ⏳ Planned |
| M2 | Speech-aware self-suppression under duplex audio | ⏳ Planned |
| M3 | Self-voice detector / attribution classifier | ⏳ Planned |
| M4 | Multi-speaker world model + memory | ⏳ Planned |
| M5 | Self-model and introspection | ⏳ Planned |

### Developmental framing

Corollary now treats early M1 as a developmental stage closer to infant babbling than adult speech. The intended sequence is:

1. detect that a sound occurred
2. determine whether the sound was self-caused
3. learn how the room transforms self-generated sound
4. handle mixed self/world audio
5. only then push toward lexical suppression and speech understanding under overlap

See `docs/plan.md` for full details.

## License

Private — personal research project
