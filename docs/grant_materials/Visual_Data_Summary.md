# Visual Data Summary for Grant Proposals

The following figures are recommended for inclusion in grant applications to demonstrate technical maturity and empirical validation.

### Figure 1: Sensory-Motor Convergence (Cerebellar Learning)
*   **Source Data:** `docs/babbling_convergence.png` (and M1b logs).
*   **Description:** A line chart showing the reduction in "Prediction Error" (y-axis) over "Babbling Iterations" (x-axis).
*   **Key Message:** "Corollary successfully learns its physical environment through active acoustic exploration, reaching stable adaptation within [N] trials."

### Figure 2: Source Attribution Matrix (M1c Validation)
*   **Source Data:** `docs/m1c_dual_input_results_2026-04-28.md`.
*   **Description:** A 3x3 confusion matrix or a set of bar charts showing 100% accuracy in labeling `SELF`, `WORLD`, and `BOTH` (overlapping) speech across multiple hardware seeds.
*   **Key Message:** "Our geometry-aware fusion rule eliminates identity spoofing and provides robust source separation, even under heavy overlap."

### Figure 3: Speech-Aware Suppression Performance (M2)
*   **Source Data:** `data/m2_baseline/m2_offline_room_eval_6cap_20260429_101059/offline_summary.json`.
*   **Description:** A "Waterfall" or "Delta" chart showing the change in Word Error Rate (WER) across different strategies. Compare "Baseline (No Processing)" vs. "Conservative Soft-Gate" vs. "Aggressive Subtraction."
*   **Key Message:** "Project Corollary preserves the intelligibility of human speech during full-duplex interaction, maintaining a 10% WER ceiling while active."

### Figure 4: Agentive Contingency (M3 Proto-Turn-Taking)
*   **Source Data:** Output from `scripts/m3_turn_taking_demo.py`.
*   **Description:** A temporal timing diagram (multi-track).
    *   Track 1: External Agent Voice (On/Off)
    *   Track 2: Corollary Intent to Speak (On/Off)
    *   Track 3: Corollary Actual Audio Output
*   **Key Message:** "The Agentive Controller demonstrates natural social contingency, automatically deferring its own output when external vocal agents are detected."

### Figure 5: Semantic Abstraction Graph (SAG) Grounding
*   **Source Data:** `docs/sag.md`.
*   **Description:** A simplified node-edge diagram showing a "Self-Concept" subtree (e.g., `Corollary.voice_id`, `Corollary.last_statement`) and its relation to "World-Concepts."
*   **Key Message:** "The SAG provides a grounded, immutable knowledge layer that protects the agent from semantic drift and adversarial prompt injection."
