# Agent 1 - Onset detection + probe design - Gemini 3.1 Flash Lite

For robust onset detection in reverberant environments for self-auditioning systems, you must prioritize methods that explicitly model or separate the direct-path arrival from subsequent room reflections.

### Best Detection Methods & Probe Families

Based on your requirement for low-latency, real-world hardware integration, prioritize these methods:

1.  **Covariance-based Onset Detection:** For hardware setups with repeated probe playback, leveraging the **unimodal covariance of a pair of consecutively measured RIRs** (Prawda et al., 2025) is significantly more robust than thresholding single-measurement energy. It mitigates the noise floor issues inherent in typical small-room reverberant measurements.
2.  **Spectral Peak Classification (Transient/Non-Transient):** Techniques that classify spectral peaks (Roebel, 2011) help separate true signal onsets from reverberant noise. By detecting "synchronized events" in subbands rather than just total energy, you improve reliability in non-ideal rooms.
3.  **Maximum Filter Trajectory Tracking:** For more complex audio environments (e.g., if background sound exists during probe playback), using a **maximum filter** to track spectral trajectories (Böck & Widmer, 2013) suppresses noise floor fluctuations and "vibrato-like" artifacts that typically cause false triggers in simple onset detectors.

### Tradeoffs: Probes

| Probe Type | Advantages | Tradeoffs |
| :--- | :--- | :--- |
| **Vowel-like / Chirps** | Superior frequency coverage for RIR estimation; vowel-like probes are more "natural" if emitted into the same space as speech. | Chirps (linear/log) require deconvolution and can be masked by non-stationary noise. |
| **Clicks / Impulses** | Fastest, lowest latency (no deconvolution needed). | Very low SNR; highly susceptible to noise/reverberation. Not recommended for real hardware without heavy averaging. |
| **Pseudo-random (MLS/IRS)** | High immunity to noise; deterministic and repeatable. | Not robust to time-variant systems (e.g., moving mics/speakers). IRS is preferred over MLS for even-order nonlinearity rejection. |

### Concrete Recommendations for M1A Path

1.  **Implement Pairwise Measurement:** If your system allows, trigger two identical probes in rapid succession. Use the **unimodal regression of their covariance** to detect the true onset time. This is your strongest tool against reverberation jitter.
2.  **Adopt Vowel-like Probes for Self-Attribution:** Given your initial tests, vowel-like signals appear more promising. They mimic the spectral structure of speech, making them easier to mask or distinguish within a system that also processes human speech.
3.  **Subband Analysis:** Do not operate on wideband signals alone. Decompose incoming signals into auditory subbands (Gammatone or Gammachirp). Onset detection is significantly more stable when you can observe synchronous transients across a set of harmonically related bands rather than relying on total signal energy.
4.  **Actionable Implementation Order:**
    *   **Phase 1:** Implement a simple chirp-based deconvolution to establish an baseline "ground truth" RIR in your room.
    *   **Phase 2:** Move to a pair-wise measurement strategy using vowel-like probes and evaluate onset stability using covariance.
    *   **Phase 3:** Integrate a maximum-filter-based spectral flux detector (SuperFlux-style) if you encounter false-positive trigger rates in the presence of noise/interference.

**Most Actionable References:**
*   *Prawda et al. (2025)*: "Cropping room impulse responses using unimodal regression of their covariance" (Best for robust onset timing in reverberation).
*   *Böck & Widmer (2013)*: "Maximum filter vibrato suppression for onset detection" (Best for real-time trajectory tracking to reduce false triggers).
*   *Roebel (2011)*: "Onset detection by means of transient peak classification" (Provides the necessary theory for subband-based onset detection).

# Agent 2 - Echo path alignment + double-talk - Gemini 3.1 Flash Lite

This research context prioritizes practical, low-latency, and robust alignment in small-room environments, shifting focus from "words" to "primitives" and causal attribution.

### Key Research Findings & Categorization

| Title | Relevance Note | Category |
| :--- | :--- | :--- |
| **[Low Complexity Echo Delay Estimator Based on Binarized Feature Matching](https://www.isca-archive.org/interspeech_2024/gao24b_interspeech.pdf)** | Uses binarized features to track variable delays with ultra-low complexity; highly transferable to Python prototypes. | Alignment, Echo Handling |
| **[An Adaptive Filter Bank Based Neural Network Approach for Time Delay Estimation](https://arxiv.org/pdf/2502.06098)** | Employs overlapping adaptive filter banks to classify delays, robust to non-linear distortion. | Probe Design, Alignment |
| **[DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining](https://arxiv.org/pdf/2603.08216)** | Trains on dual-channel conversational audio to model turn-taking and overlap/backchannels without explicit labels. | Double-talk, Attribution |
| **[JAL-Turn: Joint Acoustic-Linguistic Modeling...](https://www.arxiv.org/pdf/2603.26515)** | Uses joint acoustic-linguistic features for parallel, low-latency (12ms) turn-taking predictions. | Double-talk, Latency |
| **[EchoFree: Towards Ultra Lightweight and Efficient Neural AEC](https://arxiv.org/html/2508.06271v1)** | Combines traditional linear filtering with a lightweight neural post-filter operating on Bark-scale features. | Adaptive Filtering, Echo Handling |

---

### Implementation Insights for your Prototype

*   **Robust Alignment (Delay Estimation):** Move beyond simple cross-correlation. The **Gao et al. (Tencent)** paper on Binarized Feature Matching (BFM) is exceptionally lightweight for Python prototyping. It handles software jitter and variable path delays efficiently.
*   **Echo/Double-Talk Distinction:** The **DualTurn** paradigm suggests that generative pre-training on dual-channel audio (monitoring both mic and reference) is superior to reactive VAD-based triggers. This implicitly handles overlapping speech, as the model "anticipates" speaker turns rather than waiting for silences.
*   **Latency/Implementation:** **JAL-Turn** offers a blueprint for synchronous, parallel turn-taking prediction, achieving extremely low latency (12ms) by sharing frozen encoders (e.g., SenseVoice) with your primary task, preventing redundant compute.
*   **Causal Self/World Attribution:** While the research on reversing the "Shoebox Image Source Method" (Sprunck et al.) is mathematically dense, it highlights that **first-order reflections** are key to recovering geometry/path. Your primitive probes should specifically target these early arrivals to attribute sound sources in small rooms.

### Top 3 “Most Actionable Now”

1.  **Integrate Binarized Feature Matching (BFM):** Implement the BFM-SP-10ms (from the Tencent Interspeech 2024 paper) for your delay alignment module. It provides the most stable, low-complexity path to handling variable buffer jitters in a real-time Python implementation.
2.  **Adopt Joint Acoustic-Linguistic Features (JAL-Turn Style):** Instead of waiting for silence, start feeding a stream of joint features into your model. By combining early acoustic primitives (Bark-scale spectral features) with your primitive probes, you can predict "shift" vs. "hold" states with significantly lower latency than text-first methods.
3.  **Implement a Two-Stage Hybrid AEC (EchoFree approach):** For your prototype, pair a classic linear filtering module (e.g., partitioned-block frequency domain adaptive filter) with a light neural network post-filter trained on Bark-scale features. This ensures you maintain the robust linear baseline for strong echo paths while using the lightweight neural part to catch non-linearities and residuals.

# Agent 3 - Hearing-aid feedback suppression + adaptive filtering - Gemini 3.1 Flash Lite

This research summary covers techniques for feedback cancellation (AFC) and adaptive filtering, focusing on low-latency, real-time applicability relevant to your M1A/M1B prototype.

### Best Candidate Adaptive Methods

1.  **Partitioned Block Frequency Domain Adaptive Filtering (PBFDAF/PEFBNLMS):**
    *   **Relevance:** Standard NLMS for high-tap-length acoustic paths is computationally prohibitive. PBFDAF partitions the impulse response, offering massive computational savings while maintaining the convergence properties of time-domain NLMS. 
    *   **Implementation:** Crucial for managing long reverberant paths in small rooms without high CPU load.

2.  **Hybrid Approaches (DSP + Neural Post-Filter):**
    *   **Relevance:** Purely linear filters (like NLMS) struggle with nonlinearities (loudspeaker distortion) and require substantial decorrelation. Hybrid frameworks (e.g., EchoFree) use a linear front-end for echo cancellation and a lightweight NN post-filter for residual suppression and nonlinear artifacts.
    *   **Benefit:** Improves performance close to stable gain limits where linear adaptive filters typically fail.

3.  **Convex Combinations of Adaptive Filters:**
    *   **Relevance:** A convex combination (mixing) of two adaptive filters—one with a fast, unstable step-size (for tracking path changes) and one with a slow, stable step-size (for low steady-state error)—is an excellent strategy to balance the "fast adaptation vs. stability" trade-off without complex state-switching logic.

---

### Critical Insights & Failure Modes

*   **The "Bias" Problem (Prediction Error Method - PEM):** Adaptive filters driven by correlated input (speech) exhibit biased estimation of the feedback path, leading to poor cancellation.
    *   **Fix:** Use the Prediction Error Method (PEM) to pre-whiten input signals before adaptive filter update.
*   **Alignment/Latency:** Hard-coded alignment via cross-correlation is fragile in dynamic environments. 
    *   **Fix:** Incorporate "soft alignment" modules (e.g., self-attention) or flexible tap-length adjustment.
*   **Howling Recovery:** Abrupt changes in acoustic paths (e.g., placing a hand near the mic) often lead to "howling." 
    *   **Tradeoff:** A fixed-rate adaptive filter will either be too slow to stop the howling or too fast, leading to sound quality degradation. Using a "switched" or hybrid update rule based on a stability detector (monitoring the feedback loop gain or residual energy) is mandatory.

---

### Actionable Implementation Ideas for Corollary

1.  **Adopt a PBFDAF Core:** Instead of basic time-domain NLMS, implement PBFDAF. It provides the necessary low latency and computational efficiency for small-room acoustics.
2.  **Integrate Soft-Clipping/Stability Detector:** Add a stability detector that scales the adaptation step-size ($\mu$) or temporarily freezes adaptation when high residual energy is detected (impending howling).
3.  **Hybridize with Vowel-Probes:** Your observation regarding vowel-like probes is strong. A robust implementation would use periodic, inaudible, vowel-based "probes" to keep the adaptive filter converged during speech pauses, ensuring the system is ready when non-self signals arrive, without relying on speech activity detection (SAD) which is prone to error.

### Top 3 “Most Actionable Now”

1.  **PBFDAF Architecture:** Replace current time-domain filters with a partitioned frequency-domain block approach to gain headroom for more complex algorithms.
2.  **PEM Pre-whitening:** Implement pre-filters on the input path to the AFC to ensure the filter converges to the actual feedback path, not the correlation of the speech signal itself.
3.  **Stability-Aware Step-size:** Replace constant step-size adaptation with a variable step-size method controlled by a signal-energy-based stability detector to suppress onset howling rapidly.

---

### Research References

| Title | Relevance | Category |
| :--- | :--- | :--- |
| [L3C-DeepMFC: Low-Latency...] | Proposes low-complexity, 4ms latency T-F domain feedback control. | Implementation / Low Latency |
| [A Switched Algorithm for Adaptive Feedback Cancellation Using Pre-Filters] | Uses PEM and soft-clipping for faster recovery from howling. | Adaptive Filtering / Echo Handling |
| [Iterated Partitioned Block Frequency–Domain Adaptive Filtering] | Classic, rigorous look at PBFDAF for low-latency, long-echo path modeling. | Implementation / Adaptive Filtering |

# Agent 4 - Voiced/vowel detection + causal self/non-self attribution - Gemini 3.1 Flash Lite

For your Corollary system, the core challenge is distinguishing self-originated vocalizations from their acoustic environment counterparts (echoes, playback, or replayed prior sounds). Below are recommended methods for voiced/vowel detection and causal attribution.

### 1. Robust Voiced/Vowel Detection
To achieve robust performance in noisy and reverberant environments, move away from simple energy or threshold-based approaches.

*   **Periodicity & Harmonic Decomposition:** Use techniques that decompose signals into harmonic components.
    *   **Method:** **Variational Mode Decomposition (VMD)** combined with **Non-Local Means (NLM)** denoising (Kumar et al., 2020). VMD effectively isolates harmonic content characteristic of vowels from non-periodic noise, and NLM further suppresses residual ambient interference.
    *   **Actionable:** Implement VMD as a pre-processing step to isolate vowel-like frequency bands (formants) before further analysis.
*   **Deep Learning for Joint Estimation:** Modern deep learning models for speech enhancement and F0 estimation are highly robust.
    *   **Method:** **DC-CRN (Densely-Connected Convolutional Recurrent Network)** (Zhang & Wang, 2023). This architecture is specifically designed for multi-task learning (F0 estimation + voicing detection) and performs well under low SNR conditions and severe reverberation by operating in the complex domain.
    *   **Actionable:** Incorporate a lightweight, real-time variant of DC-CRN (e.g., DTLN-based approaches) as an "upstream" module to provide clean features for your selfhood classifier.

### 2. Causal Self vs. Non-Self Attribution
Attribution must rely on causal priors, such as motor-sensory integration, rather than spectral similarity alone.

*   **Motor Command Alignment (Efference Copy):** Leverage the "Efference Copy" (or Corollary Discharge) hypothesis, which states the motor system sends a copy of motor plans to sensory cortex to predict impending consequences.
    *   **Approach:** Use the temporal gap between the motor command (the "intent to speak") and the resulting sound as a causal anchor.
    *   **Refining Reliability:** Ensure your system explicitly tracks "motor-evoked activity" before sound onset. If the sound arriving at the microphone aligns temporally and spectrally with an internal motor-driven prediction, classify it as "Self." If it occurs without a corresponding motor trigger, categorize it as external/non-self.
*   **Action-Induced Sensitivity:** The brain does not merely suppress self-sounds; it modulates sensitivity to them. 
    *   **Implementation:** Consider building a "predictive internal model." If the incoming sound matches the expected auditory consequences of your current motor command (within a specific time window and spectral range), you can confidently assign "Self" status, even in reverberant environments.

### 3. Vowel-like Probes for Embodied Self-Signal
To make vowel-like probes reliable:

1.  **Temporal Specificity:** Treat probes as "chirps" or vowels with known phase-coherence signatures. Because the system controls these probes, they can be designed to have unique, non-natural acoustic signatures (e.g., non-human spectral characteristics) that simplify the attribution task.
2.  **Multimodal Cues:** If hardware permits, correlate sound emission with physical movement (e.g., tactile/proprioceptive feedback from the speaker housing).
3.  **Causal Gate Ablation:** Utilize the "Forward-Only Gate Ablation" (FoG-A) logic (Zhang et al., 2026) to determine which layers or frequency components of your signal drive the *causal generation* of the sound, and use this to distinguish self-produced vs. replayed playback.

### Summary: Top 3 Actionable Recommendations
1.  **Adopt a Densely-Connected Convolutional Recurrent Network (DC-CRN)** as your front-end for robust voicing and harmonic detection.
2.  **Implement a causal feedback loop:** The system must "know" it just sent a command. Use this motor-intent signal as a hard gated-prior for attribution, rather than waiting for post-hoc analysis.
3.  **Probe design:** Use vowel-like probes with specific spectral "watermarks" (invariants that occur only when self-emitted) to facilitate quick recognition in complex, reverberant, or overlapping sound environments.



 Excellent search results. I now have strong coverage across all the relevant categories. Let me synthesize this into a structured, actionable report.

---

# Agent 1.A - Onset Detection + Probe Design: Curated Research Report - Claude Sonnet 4.6

Below are 9 directly relevant results, organized by actionability. Each entry includes categories, relevance notes, and implementation flags.

---

## Results

---

### 1. Reverberation-Robust Localization of Speakers Using Distinct Speech Onsets and Multi-channel Cross-Correlations
**Link:** https://arxiv.org/abs/2604.01524 (Lin et al., Apr 2026)

**Relevance:** Derives onset detection analytically from speech signal and RIR models — not empirically tuned heuristics. The "Onset-MCCC" method encodes speech onsets per subband via an auditory filterbank, then cross-correlates to isolate direct-path arrivals and suppress reflections. Tested in real reverberant rooms at T60 ≈ 0.65s.

**Categories:**
- ✅ **Onset detection** (primary focus — direct-path extraction from reverberant onset)
- ✅ **Playback/mic alignment** (TDOA via encoded onsets)
- ✅ **Causal self/non-self attribution** (distinguishes direct-path from reflection components)
- ✅ **Voiced/vowel detection** (speech onset = energy envelope rise; subband filterbank captures formant structure)

**Implementation ideas:** The recursive averaging / direct-to-reflection ratio (DRR) onset encoding is pure DSP — implementable with `scipy.signal` filterbanks and short-time energy tracking. No ML required.

**Latency/hardware:** Designed for streaming. The subband STFT onset detection variant is frame-by-frame causal. Works single-channel or multi-channel.

---

### 2. Cropping Room Impulse Responses Using Unimodal Regression of Their Covariance
**Link:** https://eprints.whiterose.ac.uk/id/eprint/232167/1/081601_1_10.0038960.pdf (JASA Express Letters, Aug 2025)

**Relevance:** Directly solves the problem of finding reliable onset and truncation points in measured RIRs under noisy conditions — without assuming single-sloped decay or known SNR. Achieves onset errors below 8% even at very low SNRs by exploiting covariance between repeated probe captures. Outperforms threshold, mean-energy, and max-peak methods.

**Categories:**
- ✅ **Onset detection** (primary: robust RIR onset from repeated probes)
- ✅ **Playback/mic alignment** (RIR onset = the alignment anchor)
- ✅ **Probe design** (implicit: requires paired/repeated captures, compatible with any wideband probe)

**Implementation ideas:** Compute short-time covariance between two consecutive probe captures of the same signal. Apply unimodal regression (isotonic regression in scipy) to smooth the curve. Find onset as argmin above noise floor threshold. Directly translatable to Python with `numpy` + `scipy.stats`.

**Latency/hardware:** Offline post-processing of probe pairs. Not causal during probe, but the covariance onset estimate can be computed within ~10ms of the second capture completing.

---

### 3. Comparison of Different Impulse Response Measurement Techniques (Stan et al., JAES 2002)
**Link:** https://people.montefiore.uliege.be/stan/ArticleJAES.pdf

**Relevance:** The canonical empirical comparison of MLS, IRS, Time-Stretched Pulses, and Log Sine Sweep (Farina sweep/chirp) for room measurement. Definitive finding: **MLS/IRS win under non-white background noise; log sine sweep wins in quiet environments** and handles nonlinear loudspeakers better (distortion products separate in time domain).

**Categories:**
- ✅ **Probe design** (comprehensive tradeoff table for all major probe families)
- ✅ **Playback/mic alignment** (deconvolution alignment for each method)
- ✅ **Echo/double-talk handling** (MLS collapses under nonlinearity; sweep separates harmonics)

**Implementation ideas:** Use log sine sweep (chirp) when your room is low-ambient-noise and your speaker is nonlinear (most consumer hardware). Use MLS/IRS when background noise dominates. Both are straightforward in Python via `scipy.signal.chirp` + FFT deconvolution, or `pyAudioAnalysis`.

**Latency/hardware:** Offline measurement tool. Sweep duration of 100ms–2s is typical. Sub-100ms short chirps reduce time resolution tradeoffs.

---

### 4. Evaluation of Zadoff-Chu, Kasami, and Chirp-Based Encoding Schemes for Acoustic Local Positioning Systems
**Link:** https://dehesa.unex.es/bitstreams/ffa8e21f-bbbc-495a-9e8f-0c7bc33b1d58/download (IEEE TIM, 2019)

**Relevance:** Real-hardware TOA accuracy comparison of orthogonal chirp, Kasami codes, and Zadoff-Chu (ZC) sequences under multipath, near-far effects, and Doppler — all conditions your self-audition system faces. Key finding: **ZC-Chirp achieves the best TOA accuracy** with only 67 symbol length (vs 255–257 for Kasami/ZC-QPSK), making it latency-friendly.

**Categories:**
- ✅ **Probe design** (direct comparison of chirp vs. spread-spectrum probes for acoustic TOA)
- ✅ **Playback/mic alignment** (TOA = alignment anchor)
- ✅ **Onset detection** (correlation peak detection is the onset event)

**Implementation ideas:** Orthogonal chirp (OCDM-style) probe: generate chirp with `scipy.signal.chirp`, cross-correlate at receiver with matched filter. Short burst (~25ms) with good sidelobe suppression. Python: `np.correlate` or FFT-based matched filter.

**Latency/hardware:** Tested in real indoor rooms with ultrasonic transducers. Latency dominated by probe burst duration (~25ms), not processing. The correlation-peak detection is sub-ms. Directly transferable to audio-band speaker+mic.

---

### 5. Room Acoustics and Microphone Characteristics Show Systematic Impact on Sound Event Recognition (Bibbó et al., INTER-NOISE 2025)
**Link:** https://openresearch.surrey.ac.uk/esploro/fulltext/conferencePaper/Room-Acoustics-and-Microphone-Characteristics-Show/991013165002346

**Relevance:** Real-hardware study across 4 rooms and 4 microphones quantifying how RT60 and mic directivity affect detection of impulsive vs. sustained events. Critical finding for probe choice: **high RT60 degrades impulsive event detection by ~50%, but sustained events stay above 90%.** Directional mics improve effective DRR.

**Categories:**
- ✅ **Probe design** (empirical evidence: impulsive ≈ chirp-like probes degrade under reverberation; sustained ≈ vowel-like probes are resilient)
- ✅ **Voiced/vowel detection** (sustained events like speech tolerate high RT60 far better)
- ✅ **Echo/double-talk handling** (overlapping events reduce accuracy ~20%)

**Implementation ideas:** This validates your empirical hardware finding that vowel-like probes may outperform chirps in real rooms. For M1A: if RT60 > ~0.4s, sustained probes will outperform short transient probes for repeatable detection.

**Latency/hardware:** Measured at 16kHz with real speaker/mic in rooms with RT60 = 0.27–0.78s — closely matching your small room scenario.

---

### 6. Why Not Put a Microphone Near the Loudspeaker? A New Paradigm for AEC (Zhao & Wang, arXiv Nov 2025)
**Link:** https://arxiv.org/abs/2511.03244

**Relevance:** Proposes using an **auxiliary reference mic near the speaker** to capture the nonlinearly distorted playback signal, then Wiener-filtering it to suppress near-end speech before feeding a linear AEC stage. Directly addresses the self vs. non-self attribution and nonlinear echo problem with a hardware-level design choice.

**Categories:**
- ✅ **Causal self/non-self attribution** (reference mic captures self-output; neural filter suppresses external input bleed-through)
- ✅ **Echo/double-talk handling** (Wiener mask for near-end suppression in the reference channel)
- ✅ **Adaptive filtering** (linear AEC + DNN residual suppression cascade)
- ✅ **Playback/mic alignment** (reference mic removes acoustic path uncertainty)

**Implementation ideas:** For M1A: add a second mic physically co-located with the speaker cone. Use it as the AEC reference. Wiener filter = `scipy.signal.wiener` or a short NLMS filter on the reference to estimate cross-contamination. Collapses alignment jitter by measuring at the source.

**Latency/hardware:** Real-time target — small DNN post-filter on top of a linear filter. Needs two physical mics. Highly practical for a prototype rig.

---

### 7. Deep Model with Built-in Self-Attention Alignment for AEC (Indenbom et al. / Microsoft, arXiv 2022)
**Link:** https://export.arxiv.org/pdf/2208.11308v1.pdf

**Relevance:** Proposes **Align-CRUSE**, which replaces hard cross-correlation delay estimation with soft attention-based alignment in latent space — handling hardware-induced timing jitter of up to 800ms without a separate alignment block. Directly relevant to your timing jitter problem.

**Categories:**
- ✅ **Playback/mic alignment** (soft attention alignment; handles large unknown delays)
- ✅ **Adaptive filtering** (real-time AEC with built-in alignment)
- ✅ **Echo/double-talk handling** (single-talk + double-talk evaluation)

**Implementation ideas:** The alignment module is a single cross-attention layer between reference and mic streams in latent space. Lighter-weight than full Align-CRUSE: you could use a simple cross-correlation with subframe interpolation (`np.correlate` + parabolic interpolation) as a low-latency proxy, and fall back to soft-attention only when jitter is high.

**Latency/hardware:** Designed for real-time 16kHz telephony. Inference time < 20ms on CPU. The alignment technique works whether or not there's a fixed known delay.

---

### 8. EchoFree: Ultra Lightweight Neural Acoustic Echo Cancellation (Li et al., arXiv 2025)
**Link:** https://arxiv.org/html/2508.06271v1

**Relevance:** 278K parameter hybrid model (linear AEC + neural Bark-scale post-filter) that runs at 30 MMACs/s — comparable to M1 Neural Engine budget. Beats heavier models, demonstrating that the linear front-end + small NN post-filter architecture is the sweet spot for real-time edge deployment.

**Categories:**
- ✅ **Adaptive filtering** (linear AEC front-end is a standard NLMS/RLS filter)
- ✅ **Echo/double-talk handling** (end-to-end suppression including nonlinear echo)
- ✅ **Implementation ideas** (Bark-scale features + gated conv LSTM; Python-portable)

**Implementation ideas:** Bark-scale feature extraction for your post-filter is trivial in Python (`librosa.filters.mel` with Bark spacing). The NLMS linear front-end can be implemented in ~20 lines of numpy. This architecture is your M1A prototype AEC path.

**Latency/hardware:** Targets edge devices. 30 MMACs/s runs in real-time on CPU, easily on M1. Assumes 16kHz.

---

### 9. How Much to Dereverberate? Low-Latency Single-Channel SE in Distant Microphone Scenarios (Venkatesh et al. / L-Acoustics, arXiv 2025)
**Link:** https://arxiv.org/pdf/2505.01338

**Relevance:** Empirically shows that **preserving early reflections (≥30ms offset) improves dereverberation quality** in both small and large rooms. Relevant because your probe's reverberant tail is not pure noise — early reflections carry direct-path correlated energy that aids alignment. Suppressing them too aggressively hurts.

**Categories:**
- ✅ **Probe design** (informs how much of the reverberant tail to keep/discard)
- ✅ **Playback/mic alignment** (early reflections are correlated with direct path — useful for alignment)
- ✅ **Echo/double-talk handling** (late reverberation is the problem; early energy is an ally)

**Implementation ideas:** When designing your probe detection window, don't truncate at the direct-path peak — keep a 30–50ms window of early reflections as part of your "self signal" fingerprint. This improves matched-filter peak sharpness.

**Latency/hardware:** Real-time, single-channel, 16kHz. Room volumes 10–1000 m³ tested.

---

## Probe Family Tradeoff Summary

| Probe Type | Onset Clarity | Reverb Robustness | Alignment SNR | Nonlinear Tolerance | Low-Latency Viable |
|---|---|---|---|---|---|
| **Log sine sweep (chirp)** | Sharp (correlation peak) | Moderate — peak smears at high T60 | High (wideband) | ✅ Excellent (distortion separates) | Moderate (needs 50–200ms burst) |
| **Short linear chirp** | Sharp | Poor at high T60 | Moderate | Moderate | ✅ Best (10–40ms) |
| **MLS** | Sharp | Poor under noise/nonlinearity | High | ❌ Fragile | ✅ Good |
| **Orthogonal chirp (ZC-Chirp)** | Very sharp (OCDM sidelobes low) | Good | Best TOA accuracy | ✅ Good | ✅ ~25ms burst |
| **Vowel-like (sustained tonal)** | Soft (gradual onset) | ✅ Excellent — resilient to T60 | Moderate (narrowband) | Good | ✅ |
| **Click/impulse** | Sharpest possible | ❌ Very poor — smeared completely | Low (energy) | Moderate | ✅ Sub-ms |

**The key tension your hardware tests are surfacing:** chirps have sharp correlation peaks in theory (great in simulation) but real loudspeakers introduce nonlinearity + your room smears the onset. Vowel-like probes trade peak sharpness for temporal stability — their energy persists across the reverberant tail in a predictable, trackable way.

---

## Top 3 Most Actionable for M1A Right Now

### 🥇 #1: Covariance-based RIR onset detection (Paper 2)
**Why now:** Your variability problem is fundamentally an onset detection problem. Measure the same probe twice in rapid succession, compute short-time covariance, find the onset as argmin above noise threshold. Directly solves "where does my probe actually arrive" without assuming anything about room geometry or SNR. Pure numpy/scipy — 2 hours to prototype.

### 🥈 #2: Sustained probe for repeatable detection in your room (Papers 5 + Lin 2026)
**Why now:** Your hardware tests showing vowel-like probes outperforming chirps is consistent with the empirical finding that sustained events are >90% detectable at RT60 up to 0.78s while impulsive events drop 50%. Design your vowel probe with: a clear energy onset edge (for TDOA anchoring), a stable 50–100ms body (for correlation), and an envelope tracker using direct-to-reflection ratio gating. The Lin 2026 onset encoding method gives you the math.

### 🥉 #3: Auxiliary reference mic near speaker + Wiener masking (Paper 6)
**Why now:** This is the cleanest architectural fix for self/non-self attribution and alignment jitter simultaneously. Place a cheap second mic at the speaker, use it as your AEC reference. You collapse the RIR uncertainty to a near-zero-delay system. Wiener mask the reference channel for near-end bleed. This is a 1-weekend hardware change with huge downstream simplification — your alignment becomes trivial (delay ~0), and causal attribution becomes: "if it's in the reference channel, it's self."


# Agent 1.B - Onset Detection + Probe Design: Curated Research Report - Elephant Alpha

We need to find recent research on acoustic onset detection in reverberant rooms for short playback probes captured by nearby microphones, and compare methods for detecting various probe signals (chirps, clicks, tones, vowels, MLS, Golay codes) under room reflections and timing jitter. Also need work on probe design for room/audio alignment in low-latency systems. Let's search.Based on the search results, I can now provide a comprehensive response to the user's query about real-time speaker + mic self-audition in small rooms, focusing on onset detection, probe design, and playback/mic alignment.

## Results Summary

### 1. **Onset Detection in Reverberant Rooms**

**Key Paper:** "Reverberation-Robust Localization of Speakers Using Distinct Speech Onsets and Multi-channel Cross-Correlations" (arXiv:2604.01524v1)

**Relevance:** Proposes a novel speech onset detection approach specifically for reverberant environments using multi-channel cross-correlation. The method extracts direct-path components while suppressing random reflections, which is crucial for robust onset detection in small rooms with echoes.

**Key Insight:** The paper demonstrates that speech onsets (sudden increases in speech envelope) provide reliable localization cues even in reverberation up to T60 ≈ 0.65s, making them superior to continuous speech signals for real-time alignment.

**Implementation Idea:** Use envelope-based onset detection with multi-channel cross-correlation (MCCC) to identify the direct path component, then apply temporal masking to suppress reflected energy.

---

### 2. **Probe Signal Design for Room Alignment**

**Key Paper:** "Blind Localization of Early Room Reflections with Arbitrary Microphone Array" (arXiv:2409.15484v1)

**Relevance:** Focuses on designing probe signals (chirps, MLS, sweeps) for room impulse response estimation and alignment. The FF-PHALCOR method uses frequency-focused phase alignment to separate early reflections from reverberation.

**Key Insight:** Early reflections contain critical room geometry information. The choice between chirp, MLS, or tone probes depends on room size and noise characteristics:
- **Chirps**: Better for small rooms, higher energy density
- **MLS/Pseudorandom**: Better for larger spaces with more reflections
- **Tones/Vowels**: More robust in noisy real-world conditions

**Probe Family Recommendations:**
- **For initial hardware testing**: Use logarithmic sweeps (20Hz-20kHz) or MLS (length 2.5s) for comprehensive frequency coverage
- **For real-world deployment**: Vowel-like probes (formant-based) show better repeatability due to their structured spectral content
- **For low-latency**: Short chirps (10-40ms) with zero-lag correlation

---

### 3. **Tradeoffs: Chirp vs. Click vs. Tone vs. Voiced Probes**

**Key Findings Across Papers:**

| Probe Type | Advantages | Disadvantages | Best For |
|------------|------------|---------------|----------|
| **Chirp/Sweep** | High energy, precise timing, broadband | Sensitive to noise, requires post-processing | Initial calibration, small rooms |
| **MLS** | Good S/N ratio, periodic correlation | High crest factor, artifacts | Medium-sized rooms, offline analysis |
| **Tone/Vowel** | Robust to noise, matches speech production | Lower time resolution | Real-world deployment, noisy environments |
| **Click/Impulse** | Simplest, direct path dominance | Poor S/N, requires averaging | Controlled lab conditions |

**Key Paper:** "On the Phonetic Information in Ultrasonic Microphone Signals" (Livescu et al., 2009) shows that consonant-vowel-consonant sequences provide better phonetic discrimination than pure tones, suggesting that speech-like probes may be more robust for self-attribution.

---

### 4. **Playback-to-Mic Alignment Methods**

**Key Papers:** 
- "Compression of Room Impulse Responses for Compact Storage and Fast Low-Latency Convolution" (Jälmby et al., 2024)
- "Reverberation-Robust Self-Calibration and Synchronization" (PMC10781177)

**Core Methods:**
1. **Time-Domain Cross-Correlation**: Traditional but computationally intensive for long IRs
2. **Frequency-Domain GCC-PHAT**: Generalized Cross-Correlation with Phase Transform - robust to reverberation
3. **Low-Rank Approximation**: Compresses IRs for fast convolution while preserving alignment accuracy
4. **Onset-Based Alignment**: Uses speech onset edges rather than full signal correlation

**Implementation Strategy:**
- Use GCC-PHAT for initial coarse alignment
- Refine with onset-based correlation for sample-accurate timing
- Apply low-rank compression (if needed) for real-time convolution

---

### 5. **Echo/Reverberation/Overlap Handling**

**Key Insight from Multiple Papers:** Echoes are not just noise—they contain structural information about the room. The "LIME" approach (LiveProbe paper) treats early reflections as informative rather than disruptive.

**Strategies:**
- **Direct-Path Dominance Test**: Extract time-frequency bins dominated by direct path (onset energy)
- **Reflection Modeling**: Use geometric acoustic models to predict and subtract known reflections
- **Multi-Channel Blind Deconvolution**: Separate direct and reflected paths using array processing
- **Adaptive Filtering with Constraints**: Limit filter length to capture early reflections (<5ms) while ignoring late reverberation

---

### 6. **Causal Self/Non-Self Attribution**

**Key Paper:** "User-Centric Approach for Deep Residual-Echo Suppression in Double-Talk" (Israel Cohen et al., 2024)

**Core Concept:** Maintain a "user operation point" (UOP) that defines acceptable tradeoffs between:
- **RESL** (Residual Echo Suppression Level)
- **DSML** (Desired Speech Maintained Level)

This enables real-time attribution: the system can distinguish between "self" (desired speech) and "non-self" (echo/reverberation) based on user-defined quality metrics rather than fixed thresholds.

**Implementation:** Deep neural networks with embedded cross-attention alignment can learn speaker-specific characteristics, enabling attribution without explicit enrollment.

---

### 7. **Low-Latency Real-Time Considerations**

**Critical Findings:**
- **Latency Budget**: For real-time systems, total pipeline latency must stay <10-20ms
- **Buffer Size**: 256-512 samples (at 16kHz = 16-32ms) is practical for onsets
- **Computational Load**: GCC-PHAT + onset detection can run in <1ms on modern CPUs
- **Memory**: Low-rank compression can reduce IR storage by 10-50x with minimal accuracy loss

**Hardware Assumptions:**
- 16-bit, 16kHz or 44.1kHz sampling (standard for embedded systems)
- Multi-channel ring buffer for simultaneous recording/playback
- FFT acceleration available (most DSPs have hardware FFT)

---

## **Recommended First Hardware-Facing M1A Path**

### **Phase 1: Baseline Characterization (Weeks 1-2)**
1. **Probe Design**: Start with **logarithmic sine sweeps** (20Hz-20kHz, 100ms duration) as they provide the best frequency coverage with moderate energy
2. **Detection Method**: Implement **GCC-PHAT** with multi-channel cross-correlation
3. **Evaluation**: Measure alignment error in T60 = 0.3s-0.7s rooms

### **Phase 2: Real-World Testing (Weeks 3-4)**
1. **Vowel-Like Probes**: Test with synthesized vowel formants (a, e, i, o, u) to assess robustness
2. **Onset Detection**: Compare energy-based onset vs. spectral flux onset for different room sizes
3. **Self-Attribution**: Use the UOP framework to define acceptable echo suppression levels

### **Phase 3: Optimization (Weeks 5-6)**
1. **Hybrid Approach**: Combine chirp (for calibration) + vowel probes (for real-time)
2. **Low-Rank Compression**: Implement N-D tensor approximation for fast convolution
3. **Latency Tuning**: Optimize buffer sizes for your specific hardware constraints

### **Key Decision Points:**
- **For small rooms (<20m³, T60 < 0.5s)**: Chirps + GCC-PHAT provide sufficient accuracy
- **For medium rooms (20-50m³, T60 0.5-1s)**: Vowel-like probes with multi-channel onset detection
- **For noisy environments**: Prioritize speech-like probes (vowels/CSH) over pure tones

### **Implementation Priorities:**
1. **Robust onset detection** (highest priority for alignment)
2. **Probe design flexibility** (ability to switch between chirp/vowel/MLS)
3. **Low-latency cross-correlation** (GCC-PHAT or equivalent)
4. **Real-time UOP tracking** (user-centric quality metrics)

The evidence suggests that **vowel-like probes combined with onset-based alignment** offer the best balance of robustness and accuracy for real-world small-room applications, while chirps remain valuable for initial calibration and offline analysis.
