#!/usr/bin/env python3
"""M4 Threshold Sweep — no hardware required.

Tests ECAPA-TDNN embedding discrimination across a range of thresholds
using synthetic speech-like signals. Finds the minimum threshold that
correctly accepts self and rejects non-self.

Usage:
    .venv/bin/python scripts/m4_threshold_sweep_synthetic.py
"""

import sys, os, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector

SR = 16000

def make_speech_like(seed, duration=1.0):
    """Generate a speech-like signal with formant-like structure."""
    np.random.seed(seed)
    n = int(SR * duration)
    t = np.linspace(0, duration, n)
    f0 = 100 + (seed % 7) * 25  # 100-250 Hz range
    signal = np.zeros(n)
    for harmonic in range(1, 8):
        amp = 1.0 / (harmonic ** 0.7)
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)
    # Formant modulation
    signal *= (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * (3 + seed * 0.5) * t)))
    # Breathiness
    signal += np.random.randn(n) * 0.03
    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


def main():
    print("=" * 70)
    print("M4 THRESHOLD SWEEP (synthetic speech-like signals)")
    print("=" * 70)

    identifier = SpeakerIdentifier()
    identifier._load_model()
    model_type = "mock" if identifier._model == "mock" else "ecapa-tdnn"
    print(f"Model: {model_type}")
    print()

    # Generate self template (speaker 0)
    self_pcm = make_speech_like(seed=0, duration=1.0)
    print(f"Self template: {len(self_pcm)} samples ({len(self_pcm)/SR*1000:.0f}ms)")

    # Generate test speakers (0=self, 1-9=other)
    num_others = 9
    test_pcms = {}
    for i in range(num_others + 1):
        test_pcms[i] = make_speech_like(seed=i, duration=1.0)

    # Pre-compute embeddings
    print("Computing embeddings...")
    embeddings = {}
    for i, pcm in test_pcms.items():
        embeddings[i] = identifier.get_embedding(pcm)
    print(f"  Embedding dim: {embeddings[0].shape}")
    print()

    # Compute full similarity matrix
    print("Similarity matrix (cosine):")
    print(f"      ", end="")
    for j in range(num_others + 1):
        print(f"  spk{j:02d}", end="")
    print()

    sim_matrix = np.zeros((num_others + 1, num_others + 1))
    for i in range(num_others + 1):
        print(f"  spk{i:02d}", end="")
        for j in range(num_others + 1):
            n1 = np.linalg.norm(embeddings[i])
            n2 = np.linalg.norm(embeddings[j])
            sim = float(np.dot(embeddings[i], embeddings[j]) / (n1 * n2 + 1e-10))
            sim_matrix[i][j] = sim
            print(f" {sim:6.3f}", end="")
        print()
    print()

    # Self-similarity stats
    self_sims = [sim_matrix[0][j] for j in range(1, num_others + 1)]
    print(f"  Self vs others: min={min(self_sims):.3f}, max={max(self_sims):.3f}, "
          f"mean={np.mean(self_sims):.3f}, std={np.std(self_sims):.3f}")

    cross_sims = []
    for i in range(1, num_others + 1):
        for j in range(i + 1, num_others + 1):
            cross_sims.append(sim_matrix[i][j])
    if cross_sims:
        print(f"  Others vs others: min={min(cross_sims):.3f}, max={max(cross_sims):.3f}, "
              f"mean={np.mean(cross_sims):.3f}, std={np.std(cross_sims):.3f}")
    print()

    # Threshold sweep
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                  0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    print(f"{'Threshold':>10} | {'Self Accept':>11} | {'False Pos':>9} | {'False Neg':>9} | {'Accuracy':>9} | {'Status'}")
    print("-" * 75)

    results = []
    best_threshold = None
    best_accuracy = 0

    for th in thresholds:
        # Test self (speaker 0 should be accepted)
        self_sim = sim_matrix[0][0]  # always 1.0
        self_accepted = self_sim >= th

        # Test others (speakers 1-9 should be rejected)
        false_positives = 0
        for i in range(1, num_others + 1):
            sim_to_self = sim_matrix[0][i]
            if sim_to_self >= th:
                false_positives += 1

        false_negatives = 0 if self_accepted else 1
        total_tests = 1 + num_others  # 1 self + N others
        correct = (1 if self_accepted else 0) + (num_others - false_positives)
        accuracy = correct / total_tests

        status = ""
        if self_accepted and false_positives == 0:
            status = "PERFECT"
            if best_threshold is None:
                best_threshold = th
        elif self_accepted and false_positives <= 2:
            status = "GOOD"
            if best_threshold is None:
                best_threshold = th
        elif not self_accepted:
            status = "TOO HIGH"
        else:
            status = "LOOSE"

        results.append({
            "threshold": th,
            "self_accepted": self_accepted,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "accuracy": accuracy,
            "status": status,
        })

        marker = " <--" if status in ("PERFECT", "GOOD") and best_threshold == th else ""
        print(f"{th:>10.2f} | {str(self_accepted):>11} | {false_positives:>9} | "
              f"{false_negatives:>9} | {accuracy:>8.1%} | {status}{marker}")

    print()
    if best_threshold is not None:
        print(f"RECOMMENDED THRESHOLD: {best_threshold} (first perfect/good)")
    else:
        print("WARNING: No threshold achieved perfect separation!")
        # Find best accuracy
        best = max(results, key=lambda r: r["accuracy"])
        print(f"  Best accuracy: {best['accuracy']:.1%} at threshold={best['threshold']}")

    # Also test with varying buffer sizes (simulated)
    print()
    print("=" * 70)
    print("BUFFER SIZE IMPACT (at recommended threshold)")
    print("=" * 70)

    if best_threshold is not None:
        # Test with truncated signals to simulate shorter buffers
        buffer_durations = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        print(f"{'Duration(ms)':>12} | {'Self Accept':>11} | {'FP rate':>8} | {'Self Conf':>9}")
        print("-" * 50)

        for dur in buffer_durations:
            n_samples = int(SR * dur)
            self_pcm_trunc = self_pcm[:n_samples]

            # Re-register with truncated
            det = SelfVoiceDetector(identifier, threshold=best_threshold)
            det.register_self(self_pcm_trunc)

            # Test self
            is_self_self, conf_self = det.is_self(self_pcm_trunc)

            # Test others (truncated)
            fp = 0
            for i in range(1, num_others + 1):
                other_trunc = test_pcms[i][:n_samples]
                is_self_other, _ = det.is_self(other_trunc)
                if is_self_other:
                    fp += 1

            print(f"{dur*1000:>12.0f} | {str(is_self_self):>11} | {fp/num_others:>8.1%} | {conf_self:>9.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model_type,
        "num_test_speakers": num_others + 1,
        "recommended_threshold": best_threshold,
        "similarity_matrix": sim_matrix.tolist(),
        "threshold_results": results,
    }

    out_path = "data/m4_threshold_sweep_synthetic.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
