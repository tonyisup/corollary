# Research Methodology: Project Corollary

**A Developmental and Data-Driven Approach to Introspective AI**

### 1. Developmental Phasing (Infant-to-Agent Trajectory)
Unlike "language-first" models, Corollary adopts a developmental hierarchy. We posit that robust agency requires lower-level sensory-motor grounding before high-level reasoning can be reliably aligned.
*   **Stage I: Pre-Linguistic Attribution (M1).** Learning to distinguish "self-caused" vs. "world-caused" raw acoustic energy.
*   **Stage II: Room-Acoustic Adaptation (M1b).** Using "babbling" (repeated self-generated sounds) to learn a forward model of the environment.
*   **Stage III: Social Contingency & Turn-Taking (M2-M3).** Distinguishing "agentive" vocal others from environmental noise and learning the temporal dynamics of contingent exchange.
*   **Stage IV: Cognitive Reasoning & Consolidation (M4+).** Integrating episodic memory and high-level reasoning resolved through a grounded Semantic Abstraction Graph (SAG).

### 2. Experimental Paradigms
We employ three primary experimental modes to ensure both internal validity and real-world robustness:
*   **Hardware-in-the-Loop (HITL):** All critical milestones are validated on real hardware (Mac mini/PC + various microphones/speakers). This forces the system to handle real-world jitter, echo, and acoustic non-linearity.
*   **Bayesian Hyperparameter Optimization:** We use Optuna (TPE) to systematically search for optimal parameters (e.g., onset latency, causal tolerance, similarity thresholds). This ensures that performance claims are not the result of "overfit" manual tuning.
*   **Offline Strategy Benchmarking:** To isolate algorithmic performance from hardware variability, we use a "capture once, process many" workflow. We record identical mixed-source artifacts and replay multiple processing strategies (e.g., blunt gain vs. spectral subtraction) against the exact same data to generate statistically significant comparisons.

### 3. Core Validation Metrics
*   **Self-Voice Suppression Ratio (SVSR):** Measured in decibels (dB), tracking how effectively the system can zero out its own predicted audio from the input stream.
*   **Word Error Rate (WER) Preservation:** Tracking the intelligibility of external human speech during "full-duplex" interaction (when the agent is also speaking).
*   **Source Attribution F1-Score:** Frame-level accuracy in labeling audio as `SELF`, `WORLD`, or `BOTH`.
*   **Convergence Rate:** Measuring the reduction in cerebellar prediction error over repeated babbling trials.

### 4. Safety & Integrity Protocols
*   **Semantic Abstraction Graph (SAG) Enforcement:** All agent outputs and memory writes must resolve through the SAG. This provides a "type-safe" knowledge layer that prevents the model from hallucinations or semantic drift.
*   **Local-First Privacy:** All early-stage developmental data is captured and processed locally, with PII redaction built into the logging pipeline.
*   **Replay Control Validation:** We specifically test against "identity spoofing" by playing recordings of the agent's own voice back to it from an external source, ensuring the `ThalamicAttributor` correctly labels this as `WORLD` (non-self-caused) based on motor-timing mismatch.
