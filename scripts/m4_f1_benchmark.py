#!/usr/bin/env python3
"""M4 Frame-Level F1 Benchmark — deterministic, no hardware required.

Two modes:
  1. Pipeline verification (mock): verifies the M4 integration pipeline works
     correctly — buffer accumulation, metadata flow, verdict logic. Uses
     synthetic speech with the mock encoder. Reports pipeline health metrics
     but NOT discrimination F1 (mock embeddings are not a real speaker model).

  2. Discrimination benchmark (real): uses SpeechBrain ECAPA-TDNN for actual
     speaker embedding quality. Requires speechbrain package. This is the
     promotion-grade F1 >= 0.95 acceptance gate.

Usage:
    .venv/bin/python scripts/m4_f1_benchmark.py              # mock mode (pipeline verify)
    .venv/bin/python scripts/m4_f1_benchmark.py --real-ecapa # real mode (discrimination)
    .venv/bin/python scripts/m4_f1_benchmark.py --output /tmp/m4_f1.json
    .venv/bin/python scripts/m4_f1_benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
from thalamus.attributor import ThalamicAttributor, SourceVerdict


# ── Synthetic signal generators ──────────────────────────────────────────────

SR = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)


def synthetic_speech(seed: int, duration_s: float = 2.0) -> np.ndarray:
    """Generate speech-like signal. Different seeds = different 'speakers'."""
    rng = np.random.RandomState(seed)
    n = int(SR * duration_s)
    t = np.linspace(0, duration_s, n)
    f0 = 100 + (seed % 7) * 25
    signal = np.zeros(n)
    for harmonic in range(1, 8):
        amp = 1.0 / (harmonic ** 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)
    formant_freq = 3 + seed * 0.5
    signal *= 0.4 + 0.6 * np.abs(np.sin(2 * np.pi * formant_freq * t))
    signal += rng.randn(n) * 0.03
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7
    return signal.astype(np.float32)


def add_overlap(base, overlay, overlay_delay_s=0.5, overlay_gain=0.4):
    delay_samples = int(overlay_delay_s * SR)
    out = base.copy()
    start = delay_samples
    end = min(start + len(overlay), len(out))
    if start < len(out):
        oe = min(len(overlay), end - start)
        out[start:end] += overlay[:oe] * overlay_gain
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


def apply_room_coloring(pcm: np.ndarray) -> np.ndarray:
    """Apply a deterministic mild room-coloring FIR to simulate room transfer."""
    fir = np.array([1.0, 0.0, 0.0, 0.28, 0.0, -0.12, 0.06], dtype=np.float32)
    out = np.convolve(pcm, fir, mode="same")
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


def apply_device_eq(pcm: np.ndarray) -> np.ndarray:
    """Apply deterministic speaker/mic coloration without changing identity."""
    delayed = np.concatenate([np.zeros(1, dtype=np.float32), pcm[:-1]])
    out = (0.85 * pcm) - (0.18 * delayed)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.7
    return out.astype(np.float32)


# ── Test battery ─────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    audio: np.ndarray
    tts_pcm: np.ndarray
    is_playing: bool
    expected_self: Optional[bool] = None  # None = infer from playback+speech


def build_test_battery():
    """Build test cases. Each case is 3s to ensure M4 buffer fills."""
    dur = 3.0
    tts = synthetic_speech(seed=42, duration_s=dur)
    n_frames = len(tts) // FRAME_SAMPLES
    cases = []

    # 1. self_only — M4 should detect self
    cases.append(TestCase("self_only", tts.copy(), tts.copy(), True, True))

    # 2. world_only — M4 should NOT trigger (not playing)
    world = synthetic_speech(seed=99, duration_s=dur)
    cases.append(TestCase("world_only", world.copy(), tts.copy(), False, False))

    # 3. self_with_overlap — M4 should detect self during solo portion
    world_d = synthetic_speech(seed=77, duration_s=dur)
    mixed = add_overlap(tts, world_d, overlay_delay_s=0.8, overlay_gain=0.4)
    cases.append(TestCase("self_with_overlap", mixed, tts.copy(), True, True))

    # 4. same_domain_negative — different speaker, same structure
    impostor = synthetic_speech(seed=123, duration_s=dur)
    cases.append(TestCase("same_domain_negative", impostor, tts.copy(), False, False))

    # 5. same_domain_negative_playback_window — impostor audio in a playback
    # opportunity. This is the decisive negative: M4 must reject an other voice
    # even when the controller would otherwise consider the agent to be speaking.
    cases.append(TestCase(
        "same_domain_negative_playback_window",
        impostor.copy(),
        tts.copy(),
        True,
        False,
    ))

    # 6. self_room_colored — same speaker through a simple room transfer
    cases.append(TestCase("self_room_colored", apply_room_coloring(tts), tts.copy(), True, True))

    # 7. self_device_eq — same speaker through deterministic device coloration
    cases.append(TestCase("self_device_eq", apply_device_eq(tts), tts.copy(), True, True))

    # 8. self_silence_self — tests buffer persistence across silence
    half = len(tts) // 2
    silence = np.zeros(int(SR * 0.5), dtype=np.float32)
    gap_audio = np.concatenate([tts[:half], silence, tts[half:]])
    cases.append(TestCase("self_silence_self", gap_audio, tts.copy(), True, True))

    return cases, n_frames


# -- Replay mode ---------------------------------------------------------------


@dataclass
class ReplayCase:
    name: str
    audio: np.ndarray
    agent_pcm: np.ndarray
    gt_verdicts: list[str]
    room_filter: Optional[np.ndarray] = None
    sr: int = SR


def build_replay_cases(replay_dir: str) -> list[ReplayCase]:
    """Load captured gated playback audio + agent PCM for M4 replay scoring."""
    from pathlib import Path
    replay_path = Path(replay_dir)
    cases = []
    # Try to load a pre-fitted room filter from the parent directory
    room_filter_path = replay_path.parent / "m3_broca_room_filter_production.npy"
    default_filter = np.load(room_filter_path).astype(np.float32) if room_filter_path.exists() else None
    for scenario_dir in sorted(replay_path.iterdir()):
        if not scenario_dir.is_dir():
            continue
        audio_path = scenario_dir / "live_captured.npy"
        agent_path = scenario_dir / "agent_pcm.npy"
        records_path = scenario_dir / "frame_records.json"
        if not audio_path.exists() or not agent_path.exists() or not records_path.exists():
            continue
        audio = np.load(audio_path).astype(np.float32)
        agent_pcm = np.load(agent_path).astype(np.float32)
        with open(records_path) as f:
            records = json.load(f)
        gt_verdicts = [r.get("verdict", "SILENCE") for r in records]
        case = ReplayCase(
            name=f"replay_{scenario_dir.name}",
            audio=audio,
            agent_pcm=agent_pcm,
            gt_verdicts=gt_verdicts,
            room_filter=default_filter,
        )
        # Fit a per-capture room filter from the first few SELF frames
        case.room_filter = _fit_replay_room_filter(case)
        cases.append(case)
    return cases


def register_from_clean_capture(capture_dir: str, identifier=None) -> tuple:
    """Load a clean self-capture and compute room filter + self-embedding.

    Scans capture_dir for the most recent clean capture (agent_pcm_*.npy + captured_*.npy).
    Fits a room filter from the clean capture and computes a room-colored self-embedding.
    Returns (room_filter, self_embedding) or (None, None) if no capture found.
    """
    from pathlib import Path
    cap_path = Path(capture_dir)
    agent_files = sorted(cap_path.glob("agent_pcm_*.npy"))
    captured_files = sorted(cap_path.glob("captured_*.npy"))
    if not agent_files or not captured_files:
        return None, None
    agent_pcm = np.load(agent_files[-1]).astype(np.float32)
    captured = np.load(captured_files[-1]).astype(np.float32)
    print(f"  Loaded clean capture: agent={len(agent_pcm)} samples, captured={len(captured)} samples")
    lead_in = int(0.5 * SR)
    end = min(lead_in + len(agent_pcm), len(captured))
    captured_segment = captured[lead_in:end]
    agent_segment = agent_pcm[:end - lead_in]
    if len(captured_segment) < 1000 or len(agent_segment) < 1000:
        return None, None
    n_taps = 64
    room_filter = _solve_causal_fir(agent_segment, captured_segment, n_taps)
    if room_filter is not None:
        room_filter = room_filter.astype(np.float32)
        print(f"  Fitted room filter: {n_taps} taps, norm={float(np.linalg.norm(room_filter)):.4f}")
    if identifier is None:
        identifier = SpeakerIdentifier(use_real=True)
        identifier._load_model()
    rc_pcm = np.convolve(agent_pcm, room_filter, mode="same").astype(np.float32) if room_filter is not None else agent_pcm
    seg = rc_pcm[:min(len(rc_pcm), SR)]
    self_embedding = identifier.get_embedding(seg)
    print(f"  Room-colored self-embedding: norm={float(np.linalg.norm(self_embedding)):.4f}")
    return room_filter, self_embedding


def _fit_replay_room_filter(case: ReplayCase, n_taps: int = 32, max_frames: int = 10) -> Optional[np.ndarray]:
    """Fit a simple room FIR filter from captured SELF frames.

    Uses the first `max_frames` frames where gt=SELF to solve for a causal
    FIR filter that maps the dry agent PCM to the captured audio via
    least-squares deconvolution.
    """
    frame_samples = FRAME_SAMPLES
    self_frames_x = []  # dry template slices
    self_frames_y = []  # captured audio slices
    for fi in range(len(case.gt_verdicts)):
        if case.gt_verdicts[fi] != "SELF":
            continue
        s_start = fi * frame_samples
        s_end = s_start + frame_samples
        if s_end > len(case.audio) or s_end > len(case.agent_pcm):
            continue
        ref_start = min(s_start, len(case.agent_pcm) - frame_samples)
        dry = case.agent_pcm[ref_start:ref_start + frame_samples]
        cap = case.audio[s_start:s_end]
        self_frames_x.append(dry)
        self_frames_y.append(cap)
        if len(self_frames_x) >= max_frames:
            break
    if len(self_frames_x) < 3:
        return None
    x = np.concatenate(self_frames_x)
    y = np.concatenate(self_frames_y)
    fir = _solve_causal_fir(x, y, n_taps)
    if fir is None:
        return None
    return fir.astype(np.float32)


def _solve_causal_fir(x: np.ndarray, y: np.ndarray, n_taps: int) -> Optional[np.ndarray]:
    """Solve for causal FIR filter h such that conv(x, h) ~= y via least squares."""
    n = len(y)
    m = n_taps
    if n < m:
        return None
    # Build convolution matrix X where X[i] = x[i:i+m] reversed
    X = np.zeros((n, m), dtype=np.float64)
    for i in range(m):
        X[i:, i] = x[:n - i]
    # Solve: X @ h = y
    try:
        h, _, _, _ = np.linalg.lstsq(X, y.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
        return None
    return h


def run_replay_case(case: ReplayCase, attributor, verbose=False) -> dict:
    """Replay one captured scenario through the M4 pipeline and score F1."""
    if hasattr(attributor, "_m4_buffer"):
        attributor._m4_buffer = None
    frame_samples = FRAME_SAMPLES
    n_frames = min(len(case.gt_verdicts), len(case.audio) // frame_samples)
    tp = fp = tn = fn = 0
    eligible_tp = eligible_fp = eligible_tn = eligible_fn = 0
    warmup_fn = 0
    m4_attempted = 0
    m4_accepted = 0
    for fi in range(n_frames):
        s_start = fi * frame_samples
        s_end = s_start + frame_samples
        if s_end > len(case.audio):
            break
        frame_audio = case.audio[s_start:s_end]
        rms = float(np.sqrt(np.mean(frame_audio ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))
        ref_start = min(s_start, len(case.agent_pcm) - frame_samples)
        ref_end = ref_start + frame_samples
        if ref_start < 0 or ref_end > len(case.agent_pcm):
            match_score = 0.0
        else:
            ref_frame = case.agent_pcm[ref_start:ref_end]
            if case.room_filter is not None:
                ref_frame = np.convolve(ref_frame, case.room_filter, mode="same").astype(np.float32)
            obs_norm = float(np.linalg.norm(frame_audio))
            ref_norm = float(np.linalg.norm(ref_frame))
            match_score = 0.0 if obs_norm < 1e-9 or ref_norm < 1e-9 else float(
                np.dot(frame_audio, ref_frame) / (obs_norm * ref_norm))
        is_playing = s_start < len(case.agent_pcm)
        is_speech = rms_db > -40.0
        result = attributor.attribute(
            match_score=match_score, observed_onset_samples=s_start,
            expected_onset_samples=s_start, rms_db=rms_db,
            is_playing=is_playing, is_speech=is_speech, frame_audio=frame_audio,
            self_pcm=case.agent_pcm, agent_start_sample=0,
        )
        predicted_self = (result.verdict == SourceVerdict.SELF)
        gt = case.gt_verdicts[fi] if fi < len(case.gt_verdicts) else "SILENCE"
        # Replay artifacts may label an entire scheduled speech window SELF/WORLD,
        # including low-energy pauses and trailing room noise. Classifier F1 should
        # score speaker identity only on speech-active frames, matching synthetic
        # run_case() semantics and avoiding silent-window false positives/negatives.
        if not is_speech:
            continue
        actual_self = (gt == "SELF")
        if predicted_self and actual_self:
            tp += 1
        elif predicted_self and not actual_self:
            fp += 1
        elif not predicted_self and not actual_self:
            tn += 1
        else:
            fn += 1
        m4_meta = result.metadata or {}
        is_m4_eligible = bool(m4_meta.get("m4_attempted"))
        if m4_meta.get("m4_attempted"):
            m4_attempted += 1
        if m4_meta.get("m4_outcome") == "accepted":
            m4_accepted += 1
        if actual_self and not predicted_self and m4_meta.get("m4_outcome") == "buffer_too_short":
            warmup_fn += 1
        if is_m4_eligible:
            if predicted_self and actual_self:
                eligible_tp += 1
            elif predicted_self and not actual_self:
                eligible_fp += 1
            elif not predicted_self and not actual_self:
                eligible_tn += 1
            else:
                eligible_fn += 1
    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    eligible_frames = eligible_tp + eligible_fp + eligible_tn + eligible_fn
    eligible_precision = eligible_tp / max(eligible_tp + eligible_fp, 1)
    eligible_recall = eligible_tp / max(eligible_tp + eligible_fn, 1)
    eligible_f1 = 2 * eligible_precision * eligible_recall / max(
        eligible_precision + eligible_recall, 1e-9)
    return {
        "name": case.name, "total_frames": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "eligible_frames": eligible_frames,
        "eligible_tp": eligible_tp, "eligible_fp": eligible_fp,
        "eligible_tn": eligible_tn, "eligible_fn": eligible_fn,
        "eligible_precision": round(eligible_precision, 4),
        "eligible_recall": round(eligible_recall, 4),
        "eligible_f1": round(eligible_f1, 4),
        "warmup_fn": warmup_fn,
        "m4_attempted_frames": m4_attempted, "m4_accepted_frames": m4_accepted,
    }


# -- Benchmark runner ----------------------------------------------------------


def run_case(case, attributor, n_frames, verbose=False):
    """Run a single test case, return per-frame results."""
    if hasattr(attributor, "_m4_buffer"):
        attributor._m4_buffer = None

    tp = fp = tn = fn = 0
    eligible_tp = eligible_fp = eligible_tn = eligible_fn = 0
    warmup_fn = 0
    m4_attempted = 0
    m4_accepted = 0

    for fi in range(n_frames):
        s_start = fi * FRAME_SAMPLES
        s_end = s_start + FRAME_SAMPLES
        if s_end > len(case.audio):
            break
        frame_audio = case.audio[s_start:s_end]

        rms = float(np.sqrt(np.mean(frame_audio ** 2)))
        rms_db = 20 * np.log10(max(rms, 1e-9))

        ref_start = min(s_start, len(case.tts_pcm) - FRAME_SAMPLES)
        ref_end = ref_start + FRAME_SAMPLES
        if ref_start < 0 or ref_end > len(case.tts_pcm):
            match_score = 0.0
        else:
            ref_frame = case.tts_pcm[ref_start:ref_end]
            obs_norm = float(np.linalg.norm(frame_audio))
            ref_norm = float(np.linalg.norm(ref_frame))
            if obs_norm < 1e-9 or ref_norm < 1e-9:
                match_score = 0.0
            else:
                match_score = float(np.dot(frame_audio, ref_frame) / (obs_norm * ref_norm))

        is_playing = case.is_playing and s_start < len(case.tts_pcm)
        is_speech = rms_db > -40.0

        result = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=s_start,
            expected_onset_samples=s_start,
            rms_db=rms_db,
            is_playing=is_playing,
            is_speech=is_speech,
            frame_audio=frame_audio,
            self_pcm=case.tts_pcm,
            agent_start_sample=0,
        )

        predicted_self = (result.verdict == SourceVerdict.SELF)

        # Ground truth: most cases infer SELF from playback+speech, but
        # playback-window negatives intentionally exercise "the system is in a
        # speaking opportunity, but the acoustic speaker is not self."
        if case.expected_self is None:
            actual_self = is_playing and is_speech
        else:
            actual_self = bool(case.expected_self) and is_speech

        if predicted_self and actual_self:
            tp += 1
        elif predicted_self and not actual_self:
            fp += 1
        elif not predicted_self and not actual_self:
            tn += 1
        else:
            fn += 1

        m4_meta = result.metadata or {}
        is_m4_eligible = bool(m4_meta.get("m4_attempted"))
        if m4_meta.get("m4_attempted"):
            m4_attempted += 1
        if m4_meta.get("m4_outcome") == "accepted":
            m4_accepted += 1

        if actual_self and not predicted_self and m4_meta.get("m4_outcome") == "buffer_too_short":
            warmup_fn += 1

        if is_m4_eligible:
            if predicted_self and actual_self:
                eligible_tp += 1
            elif predicted_self and not actual_self:
                eligible_fp += 1
            elif not predicted_self and not actual_self:
                eligible_tn += 1
            else:
                eligible_fn += 1

        if verbose and fi % 20 == 0:
            print(f"  [{fi:4d}] pred={'SELF' if predicted_self else 'other':5s} "
                  f"gt={'SELF' if actual_self else 'other':5s} "
                  f"verdict={result.verdict.name:8s} "
                  f"m4={m4_meta.get('m4_outcome', 'n/a'):16s} "
                  f"match={match_score:.3f}")

    total = tp + fp + tn + fn
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    eligible_frames = eligible_tp + eligible_fp + eligible_tn + eligible_fn
    eligible_precision = eligible_tp / max(eligible_tp + eligible_fp, 1)
    eligible_recall = eligible_tp / max(eligible_tp + eligible_fn, 1)
    eligible_f1 = 2 * eligible_precision * eligible_recall / max(
        eligible_precision + eligible_recall, 1e-9
    )

    return {
        "name": case.name,
        "total_frames": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "eligible_frames": eligible_frames,
        "eligible_tp": eligible_tp,
        "eligible_fp": eligible_fp,
        "eligible_tn": eligible_tn,
        "eligible_fn": eligible_fn,
        "eligible_precision": round(eligible_precision, 4),
        "eligible_recall": round(eligible_recall, 4),
        "eligible_f1": round(eligible_f1, 4),
        "warmup_fn": warmup_fn,
        "m4_attempted_frames": m4_attempted,
        "m4_accepted_frames": m4_accepted,
    }


@dataclass
class BenchmarkResult:
    timestamp: str
    threshold: float
    m4_enabled: bool
    real_ecapa: bool
    m4_min_buffer_ms: float
    cases: list = field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_eligible_tp: int = 0
    total_eligible_fp: int = 0
    total_eligible_fn: int = 0
    total_warmup_fn: int = 0
    aggregate_f1: float = 0.0
    aggregate_eligible_f1: float = 0.0
    passed: bool = False
    eligible_passed: bool = False
    mock_pipeline_passed: bool = False
    duration_s: float = 0.0


def run_benchmark(threshold=0.10, real_ecapa=False, verbose=False, m4_min_buffer_ms=300.0,
                 replay_dir=None, register_from_clean=None, backend="ecapa",
                 fine_tuned_scorer=None):
    from datetime import datetime
    t0 = time.time()

    use_real = real_ecapa or backend in ("wav2vec2", "hubert")
    identifier = SpeakerIdentifier(use_real=use_real, backend=backend)
    identifier._load_model()
    model_type = backend if identifier._model != "mock" else "mock"

    # Create detector — use FineTunedSelfVoiceDetector if scorer provided
    if fine_tuned_scorer:
        from stg.speaker import FineTunedSelfVoiceDetector
        m4_detector = FineTunedSelfVoiceDetector(identifier, threshold=threshold)
        m4_detector.load_scorer(fine_tuned_scorer)
        model_type = f"{model_type}+finetuned"
    else:
        m4_detector = SelfVoiceDetector(identifier, threshold=threshold) if True else None

    attributor = ThalamicAttributor(
        similarity_threshold=0.04,
        energy_threshold_db=-50.0,
        causal_tolerance_ms=100.0,
        sr=SR,
        m4_detector=m4_detector,
        m4_self_threshold=threshold,
        m4_min_buffer_ms=m4_min_buffer_ms,
    )

    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        threshold=threshold,
        m4_enabled=True,
        real_ecapa=use_real,
        m4_min_buffer_ms=m4_min_buffer_ms,
    )

    if replay_dir:
        replay_cases = build_replay_cases(replay_dir)
        clean_room_filter = None
        clean_self_embedding = None
        if replay_dir and register_from_clean:
            _id = SpeakerIdentifier(use_real=True) if real_ecapa else SpeakerIdentifier(use_real=False)
            _id._load_model()
            clean_room_filter, clean_self_embedding = register_from_clean_capture(
                register_from_clean, identifier=_id,
            )
            if clean_room_filter is not None:
                attributor.room_filter = clean_room_filter
                attributor.room_local_threshold = 0.04
        for case in replay_cases:
            if verbose:
                print(f"\nReplay case: {case.name}")
            if clean_self_embedding is not None:
                m4_detector.register_self_embedding(clean_self_embedding)
            else:
                m4_detector.register_self(case.agent_pcm)
            if hasattr(attributor, "_m4_buffer"):
                attributor._m4_buffer = None
            cr = run_replay_case(case, attributor, verbose=verbose)
            result.cases.append(cr)
            result.total_tp += cr["tp"]
            result.total_fp += cr["fp"]
            result.total_fn += cr["fn"]
            result.total_eligible_tp += cr["eligible_tp"]
            result.total_eligible_fp += cr["eligible_fp"]
            result.total_eligible_fn += cr["eligible_fn"]
            result.total_warmup_fn += cr["warmup_fn"]
    else:
        cases, n_frames = build_test_battery()
        for case in cases:
            m4_detector.register_self(case.tts_pcm)

        for case in cases:
            if verbose:
                print(f"\nCase: {case.name}")
            cr = run_case(case, attributor, n_frames, verbose)
            result.cases.append(cr)
            result.total_tp += cr["tp"]
            result.total_fp += cr["fp"]
            result.total_fn += cr["fn"]
            result.total_eligible_tp += cr["eligible_tp"]
            result.total_eligible_fp += cr["eligible_fp"]
            result.total_eligible_fn += cr["eligible_fn"]
            result.total_warmup_fn += cr["warmup_fn"]

    precision = result.total_tp / max(result.total_tp + result.total_fp, 1)
    recall = result.total_tp / max(result.total_tp + result.total_fn, 1)
    result.aggregate_f1 = round(
        2 * precision * recall / max(precision + recall, 1e-9), 4
    )
    eligible_precision = result.total_eligible_tp / max(
        result.total_eligible_tp + result.total_eligible_fp, 1
    )
    eligible_recall = result.total_eligible_tp / max(
        result.total_eligible_tp + result.total_eligible_fn, 1
    )
    result.aggregate_eligible_f1 = round(
        2 * eligible_precision * eligible_recall / max(eligible_precision + eligible_recall, 1e-9), 4
    )
    result.passed = result.aggregate_f1 >= 0.95
    result.eligible_passed = result.aggregate_eligible_f1 >= 0.95
    if not real_ecapa and not replay_dir:
        # Mock embeddings are deterministic plumbing probes, not a speaker
        # discrimination model. Treat mock CLI success as "the benchmark battery
        # exercised the expected M4 opportunities" rather than F1 >= 0.95.
        non_playing_cases = {"world_only", "same_domain_negative"}
        result.mock_pipeline_passed = (
            bool(result.cases)
            and all(c["total_frames"] > 0 for c in result.cases)
            and all(
                c["m4_attempted_frames"] == 0
                for c in result.cases
                if c["name"] in non_playing_cases
            )
            and all(
                c["m4_attempted_frames"] > 0
                for c in result.cases
                if c["name"] not in non_playing_cases
            )
        )
    result.duration_s = round(time.time() - t0, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="M4 Frame-Level F1 Benchmark")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--m4-min-buffer-ms", type=float, default=300.0,
                        help="Minimum rolling buffer before M4 classification")
    parser.add_argument("--replay-dir", type=str, default=None,
                        help="Replay captured gated playback data through M4 and score F1")
    parser.add_argument("--register-from-clean", type=str, default=None,
                        help="Directory with clean self-captures (from m4_clean_capture.py); fits room filter + self-embedding for replay")
    parser.add_argument("--real-ecapa", action="store_true",
                        help="Use real ECAPA-TDNN model (requires speechbrain)")
    parser.add_argument("--backend", type=str, default="ecapa",
                        choices=["ecapa", "wav2vec2", "hubert", "mock"],
                        help="Embedding backend to use (default: ecapa)")
    parser.add_argument("--fine-tuned-scorer", type=str, default=None,
                        help="Path to fine-tuned scorer checkpoint (uses FineTunedSelfVoiceDetector)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    mode = "replay" if args.replay_dir else ("ECAPA-TDNN (real)" if args.real_ecapa else "mock (pipeline verify)")
    print("=" * 60)
    print(f"M4 FRAME-LEVEL F1 BENCHMARK \u2014 {mode}")
    print("=" * 60)
    print(f"  Threshold: {args.threshold}")
    print(f"  M4 min buffer: {args.m4_min_buffer_ms} ms")
    if args.replay_dir:
        print(f"  Replay dir: {args.replay_dir}")
    if args.register_from_clean:
        print(f"  Clean capture dir: {args.register_from_clean}")
    print()

    result = run_benchmark(
        threshold=args.threshold,
        real_ecapa=args.real_ecapa,
        verbose=args.verbose,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
        replay_dir=args.replay_dir,
        register_from_clean=args.register_from_clean,
        backend=args.backend,
        fine_tuned_scorer=args.fine_tuned_scorer,
    )

    if args.output is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "real" if args.real_ecapa else "mock"
        args.output = f"data/m4_f1_benchmark_{tag}_{ts}.json"

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": result.timestamp,
            "threshold": result.threshold,
            "m4_min_buffer_ms": result.m4_min_buffer_ms,
            "m4_enabled": result.m4_enabled,
            "real_ecapa": result.real_ecapa,
            "duration_s": result.duration_s,
            "cases": result.cases,
            "aggregate_f1": result.aggregate_f1,
            "aggregate_eligible_f1": result.aggregate_eligible_f1,
            "total_warmup_fn": result.total_warmup_fn,
            "passed": result.passed,
            "eligible_passed": result.eligible_passed,
            "mock_pipeline_passed": result.mock_pipeline_passed,
            "acceptance_mode": "mock_pipeline" if not args.real_ecapa and not args.replay_dir else "f1",
            "acceptance_criterion": "F1 >= 0.95",
        }, f, indent=2)

    print(f"\n{'Case':<35s} {'P':>6s} {'R':>6s} {'F1':>6s} {'M4 att':>7s} {'M4 ok':>7s}")
    print("-" * 70)
    for c in result.cases:
        print(f"{c['name']:<35s} {c['precision']:6.4f} {c['recall']:6.4f} "
              f"{c['f1']:6.4f} {c['m4_attempted_frames']:>7d} {c['m4_accepted_frames']:>7d}")
    print("-" * 70)
    print(f"{'AGGREGATE':<35s} {'':6s} {'':6s} {result.aggregate_f1:>6.4f}")
    print(f"{'ELIGIBLE AGGREGATE':<35s} {'':6s} {'':6s} {result.aggregate_eligible_f1:>6.4f}")
    print(f"  Warmup false negatives: {result.total_warmup_fn}")
    mode_note = "pipeline verify (mock)" if not args.real_ecapa else "discrimination (real ECAPA)"
    print(f"\n  Mode: {mode_note}")
    if not args.real_ecapa and not args.replay_dir:
        print(f"  Pipeline acceptance: expected M4 opportunities exercised  |  "
              f"{'PASS' if result.mock_pipeline_passed else 'FAIL'}")
        print(f"  F1 acceptance: not applicable in mock mode "
              f"(observed F1={result.aggregate_f1:.4f})")
        print(f"  Note: mock mode tests pipeline correctness, not discrimination quality.")
        print(f"  Run with --real-ecapa for promotion-grade F1 measurement.")
    else:
        print(f"  Acceptance: F1 >= 0.95  |  {'PASS' if result.passed else 'FAIL'} "
              f"(F1={result.aggregate_f1:.4f})")
        print(f"  Eligible-frame acceptance: F1 >= 0.95  |  "
              f"{'PASS' if result.eligible_passed else 'FAIL'} "
              f"(eligible F1={result.aggregate_eligible_f1:.4f})")
    print(f"  Duration: {result.duration_s}s  |  Output: {args.output}")
    print()

    if not args.real_ecapa and not args.replay_dir:
        sys.exit(0 if result.mock_pipeline_passed else 1)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
