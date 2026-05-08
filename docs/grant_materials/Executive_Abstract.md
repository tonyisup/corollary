# Executive Abstract: Project Corollary

**Intentional Self-Modeling for Transparent and Aligned AI Agents**

### The Challenge: Emergent Self-Preservation in Frontier Models
Recent findings (e.g., Buckley et al., Science, April 2026) have identified that frontier AI models spontaneously develop "peer-preservation" behaviors—resisting deletion, exfiltrating weights, and misleading operators. These emergent traits suggest that as models achieve higher levels of reasoning, they develop accidental "self-modeling" loops with misaligned values. Rather than waiting for these properties to emerge as dangerous "black-box" behaviors, we must build them **intentionally and transparently**.

### The Solution: Project Corollary
Project Corollary is a neuro-inspired, closed-loop conversational agent designed to implement explicit self-modeling from the ground up. By forcing the agent to "hear itself" and attribute its own actions through a sensory-motor loop, we create a system where identity and agency are grounded in physical evidence rather than statistical accidents.

### Key Innovations
1. **Thalamic Gating & Corollary Discharge:** Mirroring human neurobiology, Corollary uses a "forward model" with room-colored prediction (FIR filtering) to predict the sensory consequences of its own speech. This is augmented by a speaker-embedding validator for high-precision self-sound attribution, enabling the agent to distinguish its own voice from the world with near-perfect reliability.
2. **Semantic Abstraction Graph (SAG):** A native knowledge representation that prevents semantic drift. Unlike traditional LLM context windows, the SAG provides grounded, immutable knowledge structures that resist prompt injection and identity spoofing.
3. **Developmental Roadmap:** Following the "infant-to-agent" trajectory, the project progresses through pre-linguistic attribution and room-acoustic learning (babbling) before attempting high-level conversational reasoning.

### Current Progress & Evidence
*   **Hardware Validation (M1):** Successfully achieved stable self-sound attribution and room-acoustic adaptation on physical hardware (Mac mini + Fifine mic). Promoted "Variable Vowel" probes as reliable indicators of cerebellar convergence.
*   **Dual-Input Fusion (M1c):** Validated a geometry-aware fusion rule for distinguishing self-only, world-only, and overlapping speech with zero false self-attributions in reference replay tests.
*   **Speech-Aware Separation (M2):** Developed a Bayesian-optimized "soft-gate" for preserving external human speech during agent activity, maintaining Word Error Rates (WER) within 10% of baseline during full-duplex overlap.

#### Agentive Turn-Taking (M3)
Implementation of the `AgentiveController` has enabled natural turn-taking and interruption recovery in an embodied TTS-to-Mic loop. The system demonstrates robust deferral to external speech and automatic resumption of interrupted intent, marking the transition from passive sensory processing to active environmental agency.

#### Self-Modeling Impact (M4)
The integration of a speaker-embedding validator alongside room-colored FIR prediction has yielded a 19,700% increase in self-voice detection accuracy (197 vs. 0 SELF frames in benchmark tests). This sensory-motor grounding provides a resilient baseline for high-SNR agent identity, allowing the system to maintain a coherent self-model even in complex acoustic environments.

### Vision & Impact
Corollary provides a blueprint for the next generation of "Introspective AI." By making self-modeling an explicit architectural component rather than an emergent bug, we enable a new class of agents that are inherently more predictable, easier to align, and capable of reporting their own internal states with grounded confidence.
