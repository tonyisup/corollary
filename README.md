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

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| M0 | Simulated loop (no hardware) | 🔲 Not started |
| M1 | Half-duplex with mic+speakers | 🔲 |
| M2 | Full-duplex AEC + efference copy | 🔲 |
| M3 | Self-voice detector (F1 ≥ 0.95) | 🔲 |
| M4 | Multi-speaker world model + memory | 🔲 |
| M5 | Self-model and introspection | 🔲 |

See `docs/plan.md` for full details.

## License

Private — personal research project
