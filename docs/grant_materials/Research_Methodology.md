# Research Methodology: Project Corollary

**A Developmental and Data-Driven Approach to Introspective AI**

## 1. Developmental Phasing (Infant-to-Agent Trajectory)
Unlike "language-first" models, Corollary adopts a developmental hierarchy. We posit that robust agency requires lower-level sensory-motor grounding before high-level reasoning can be reliably aligned.
* **Stage I: Pre-Linguistic Attribution (M1).** Learning to distinguish "self-caused" vs. "world-caused" raw acoustic energy. Achieved stable self-sound attribution and room-acoustic adaptation on physical hardware (Mac mini + Fifine mic).
* **Stage II: Room-Acoustic Adaptation (M1b).** Using "babbling" (repeated self-generated sounds) to learn a forward model of the environment. Validated cross-room transfer with trust-gated re-babble policy.
* **Stage III: Social Contingency & Turn-Taking (M2–M3).** Distinguishing "agentive" vocal others from environmental noise and learning the temporal dynamics of contingent exchange using the PFC controller bridge. The `AgentiveController` implements natural deferral, interruption recovery, and repair-resume cycles, validated in a live TTS-to-Mic loop.
* **Stage IV: Cognitive Self-Modeling & Robust Attribution (M4).** Integrating a speaker-embedding detector (pyannote-based) for grounded self-attribution. The balanced scorer achieves >99% recall on held-out self-captures with 0% false positives across an 8-voice negative training set (Guy, Christopher,Brian, Andrew, Guy, Eric, Jenny, plus diverse synthetic voices).
* **Stage V: Introspective Self-Model Enrichment (M5).** Episodic memory and introspective tracking. Corollary monitors its own performance metrics, detects self-voice drift across rooms/devices, and maintains an enriched self-attribution history.
* **Stage VI: Adaptive Self-Tracking & Latency Validation (M6–M7).** The live introspection pipeline enables real-time self-model drift detection, device-profile management, and human-competitive interruption latency (measured via second-device threading with actual human audio onset detection).

## 2. Experimental Paradigms
We employ three primary experimental modes to ensure both internal validity and real-world robustness:
* **Hardware-in-the-Loop (HITL):** All critical milestones are validated on real hardware (Mac mini/PC + various microphones/speakers). This forces the system to handle real-world jitter, echo, and acoustic non-linearity.
* **Embodied TTS-to-Mic Loop:** A testing mode that uses real Broca TTS output to validate M2/M3/M4 processing within a closed sensory-motor loop, capturing true acoustic reflections.
* **Bayesian Hyperparameter Optimization:** We use Optuna (TPE) to systematically search for optimal parameters (e.g., onset latency, causal tolerances, similarity thresholds). This ensures that performance claims are not the result of "overfit" manual tuning.
* **Offline Strategy Benchmarking:** To isolate algorithmic performance from hardware variability, we use a "capture once, process many" workflow. We record identical mixed-source artifacts and replay multiple processing strategies (e.g., blunt gain vs. spectral subtraction) against the exact same data to generate statistically significant comparisons.
* **Multi-Artifact Acceptance Harness:** A structured promotion gate that validates M4 scorer performance across multiple captured artifacts simultaneously, ensuring the system generalizes across diverse acoustic conditions rather than overfitting to a single session.

## 3. Core Validation Metrics
* **Self-Voice Suppression Ratio (SVSR):** Measured in dB, tracking how effectively the system zeros out its own predicted audio from the input stream.
* **Word Error Rate (WER) Preservation:** Tracking intelligibility of external human speech during "full-duplex" interaction (when the agent is also speaking).
* **Source Attribution F1:** Frame-level accuracy in labeling audio as `SELF`, `WORLD`, or `BOTH`.
* **M4 Self-Attribution Gain:** Measured as the increase in `SELF` verdicts within ambiguous overlap frames when the speaker-embedding detector is enabled.
* **Interruption Recovery Rate:** Tracking success rate of resuming deferred utterances after mid-speech interruption.
* **Interruption Latency:** M7 measures human-competitive response latency; second-device threading provides actual onset detection vs. intended cue.
* **Convergence Rate:** Reduction in cerebellar prediction error over repeated babbling trials.
* **Multi-Artifact Acceptance:** M4 promotion gate requiring simultaneous pass across ≥3 distinct captured artifacts.
