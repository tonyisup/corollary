#!/usr/bin/env python3
"""M3 Live E2E Validation Harness.

Replays pre-recorded captured audio through the speaker and re-captures it
through the mic, running the full attribution + controller pipeline in real-time.
Compares live results against the offline analysis of the same recording.

This tests:
- Room acoustics (speaker → room → mic)
- Real-time capture latency and jitter
- Attribution accuracy with live audio
- Controller state transitions under real conditions

Usage:
    python scripts/m3_live_e2e_harness.py --scenario human_early
    python scripts/m3_live_e2e_harness.py --all
    python scripts/m3_live_e2e_harness.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.dirname(__file__))

from broca.tts import synthesize
from cochlea.capture import SR
from cochlea.vad import VAD_FRAME_SAMPLES, SmoothedVAD
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from pfc.agentive_controller import AgentiveController
from pfc.m2_artifact_replay import frames_from_frame_records
from pfc.m2_turn_bridge import run_turn_taking_bridge
from m2_speech_preservation_harness import run_m2_trial, save_trial_artifacts


# ── Scenario definitions ──────────────────────────────────────────────────────

@dataclass
class LiveScenario:
    name: str
    captured_audio_file: str       # Path to .npy file with pre-recorded captured audio
    agent_pcm_file: str            # Path to .npy file with agent TTS
    agent_text: str
    human_audio_file: str = None   # Path to clean human speech .npy (for dual-speaker)
    human_delay_s: float = 0.0     # Human onset relative to agent_start_s
    expected_behavior: str = "interrupt_defer_resume"
    description: str = ""


DATA_DIR = Path("data/m3_interrupt_v3-mic_adjusted")
HUMAN_SPEECH_DIR = Path("data/m3_human_speech")

SCENARIOS: dict[str, LiveScenario] = {
    "human_early": LiveScenario(
        name="human_early",
        captured_audio_file="data/m3_interrupt_v3-mic_adjusted/human_early/captured.npy",
        agent_pcm_file="data/m3_interrupt_v3-mic_adjusted/human_early/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        human_audio_file="data/m3_human_speech/human_speech_00.npy",
        human_delay_s=0.8,
        expected_behavior="interrupt_defer_resume",
        description="Human starts mid-utterance",
    ),
    "human_late": LiveScenario(
        name="human_late",
        captured_audio_file="data/m3_interrupt_v3-mic_adjusted/human_late/captured.npy",
        agent_pcm_file="data/m3_interrupt_v3-mic_adjusted/human_late/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        human_audio_file="data/m3_human_speech/human_speech_01.npy",
        human_delay_s=1.8,
        expected_behavior="interrupt_defer_resume",
        description="Human starts near end of utterance",
    ),
    "human_before": LiveScenario(
        name="human_before",
        captured_audio_file="data/m3_interrupt_v3-mic_adjusted/human_before/captured.npy",
        agent_pcm_file="data/m3_interrupt_v3-mic_adjusted/human_before/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        human_audio_file="data/m3_human_speech/human_speech_02.npy",
        human_delay_s=-1.4,
        expected_behavior="defer",
        description="Human speaks before agent starts",
    ),
    "human_after": LiveScenario(
        name="human_after",
        captured_audio_file="data/m3_interrupt_v3-mic_adjusted/human_after/captured.npy",
        agent_pcm_file="data/m3_interrupt_v3-mic_adjusted/human_after/agent_pcm.npy",
        agent_text="I was speaking, then paused.",
        human_audio_file="data/m3_human_speech/human_speech_03.npy",
        human_delay_s=3.0,
        expected_behavior="respond",
        description="Human speaks after agent finishes",
    ),
    "overlap_self_first": LiveScenario(
        name="overlap_self_first",
        captured_audio_file="data/m3_interrupt_v3-mic_adjusted/overlap_self_first/captured.npy",
        agent_pcm_file="data/m3_interrupt_v3-mic_adjusted/overlap_self_first/agent_pcm.npy",
        agent_text="I'm Corollary. I started speaking first.",
        human_audio_file="data/m3_human_speech/human_speech_04.npy",
        human_delay_s=0.5,
        expected_behavior="interrupt_defer_resume",
        description="Human and agent overlap from the start",
    ),
}


# ── Live replay + capture ────────────────────────────────────────────────────

def _round_seconds(value: float) -> float:
    """Round seconds for stable JSON/test output while preserving audio timing."""
    return round(float(value), 6)


def plan_controller_gated_tts_segments(
    agent_start_s: float,
    agent_duration_s: float,
    human_start_s: float | None,
    human_end_s: float | None,
    expected_behavior: str,
    resume_delay_s: float = 0.2,
) -> list[dict]:
    """Plan physical TTS playback segments for controller-gated M3 scenarios.

    The live harness has deterministic scenario timing, so this function turns
    the controller policy into an audio-output plan before opening the output
    stream:
    - defer: hold the full utterance until the pre-existing human turn ends
    - interrupt_defer_resume: play until overlap, mute during human speech, then
      resume the remaining source audio
    - respond/other: play the full utterance at the normal agent start

    Segment dictionaries use seconds and contain source/destination offsets:
    {"src_start_s", "dst_start_s", "duration_s"}.
    """
    if agent_duration_s <= 0:
        return []

    if human_start_s is None or human_end_s is None or human_end_s <= human_start_s:
        return [{
            "src_start_s": 0.0,
            "dst_start_s": _round_seconds(agent_start_s),
            "duration_s": _round_seconds(agent_duration_s),
        }]

    agent_end_s = agent_start_s + agent_duration_s

    if expected_behavior == "defer" and human_start_s <= agent_start_s < human_end_s:
        return [{
            "src_start_s": 0.0,
            "dst_start_s": _round_seconds(human_end_s + resume_delay_s),
            "duration_s": _round_seconds(agent_duration_s),
        }]

    if expected_behavior == "interrupt_defer_resume":
        overlap_start_s = max(agent_start_s, human_start_s)
        overlap_end_s = min(agent_end_s, human_end_s)
        if overlap_start_s < overlap_end_s:
            segments: list[dict] = []
            pre_duration_s = max(0.0, overlap_start_s - agent_start_s)
            if pre_duration_s > 0:
                segments.append({
                    "src_start_s": 0.0,
                    "dst_start_s": _round_seconds(agent_start_s),
                    "duration_s": _round_seconds(pre_duration_s),
                })
            remaining_src_s = pre_duration_s
            remaining_duration_s = max(0.0, agent_duration_s - remaining_src_s)
            if remaining_duration_s > 0:
                segments.append({
                    "src_start_s": _round_seconds(remaining_src_s),
                    "dst_start_s": _round_seconds(human_end_s + resume_delay_s),
                    "duration_s": _round_seconds(remaining_duration_s),
                })
            return segments

    return [{
        "src_start_s": 0.0,
        "dst_start_s": _round_seconds(agent_start_s),
        "duration_s": _round_seconds(agent_duration_s),
    }]


def controller_request_time_s(expected_behavior: str, original_agent_start_s: float) -> float:
    """Return when PFC should request speech for post-hoc controller scoring.

    Gated playback may delay the physical TTS segment, especially for `defer`,
    but the controller request still occurs at the original agent intent time.
    """
    return original_agent_start_s if expected_behavior == "defer" else 0.0


def build_controller_gated_tts_buffer(
    tts_output: np.ndarray,
    sr: int,
    total_samples: int,
    segments: list[dict],
) -> np.ndarray:
    """Render planned gated TTS segments into a physical output buffer."""
    buffer = np.zeros(total_samples, dtype=np.float32)
    for seg in segments:
        src_start = max(0, int(round(float(seg["src_start_s"]) * sr)))
        dst_start = max(0, int(round(float(seg["dst_start_s"]) * sr)))
        duration = max(0, int(round(float(seg["duration_s"]) * sr)))
        if duration <= 0 or src_start >= len(tts_output) or dst_start >= total_samples:
            continue
        n = min(duration, len(tts_output) - src_start, total_samples - dst_start)
        if n > 0:
            buffer[dst_start:dst_start + n] = tts_output[src_start:src_start + n]
    return buffer


async def calibrate_noise_floor(
    input_device: int,
    duration_s: float = 2.0,
    margin_db: float = 15.0,
) -> float:
    """Capture room audio and measure the ambient noise floor.
    
    Plays silence through the output (so the room is in its normal state)
    and captures mic input for `duration_s` seconds. Measures the RMS
    energy distribution and returns an energy threshold that is `margin_db`
    above the median noise floor.
    
    This is the "LISTEN" phase — the system calibrates to the room before
    trying to detect speech.
    
    Returns the recommended energy_threshold_db.
    """
    import sounddevice as sd
    
    print(f"\n[CALIBRATING] Listening to room for {duration_s}s...")
    
    sr = int(sd.query_devices(input_device)["default_samplerate"])
    total_samples = int(duration_s * sr)
    captured = np.zeros(total_samples, dtype=np.float32)
    capture_queue = asyncio.Queue()
    samples_captured = [0]
    
    def audio_callback(indata, outdata, frames, time_info, status):
        # Output silence
        outdata[:] = 0
        # Capture mic input
        n = min(frames, total_samples - samples_captured[0])
        if n > 0:
            captured[samples_captured[0]:samples_captured[0] + n] = indata[:n, 0]
            samples_captured[0] += n
        capture_queue.put_nowait(n)
    
    with sd.Stream(
        samplerate=sr,
        blocksize=int(VAD_FRAME_SAMPLES * sr / SR),
        dtype="float32",
        channels=1,
        callback=audio_callback,
        device=(input_device, sd.default.device[1]),
    ):
        await asyncio.sleep(duration_s + 0.3)
    
    # Trim to actual captured length
    captured = captured[:samples_captured[0]]
    
    if len(captured) < VAD_FRAME_SAMPLES:
        print(f"  WARNING: Only captured {len(captured)} samples, using default threshold")
        return -55.0
    
    # Measure RMS energy in 30ms frames
    frame_count = len(captured) // VAD_FRAME_SAMPLES
    rms_values = []
    for i in range(frame_count):
        frame = captured[i * VAD_FRAME_SAMPLES:(i + 1) * VAD_FRAME_SAMPLES]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        rms_values.append(rms_db)
    
    rms_array = np.array(rms_values)
    median_floor = np.median(rms_array)
    p90_floor = np.percentile(rms_array, 90)
    
    # Set threshold above the noise floor
    energy_threshold = median_floor + margin_db
    
    print(f"  Room noise floor: median={median_floor:.1f} dB, p90={p90_floor:.1f} dB")
    print(f"  Energy threshold: {energy_threshold:.1f} dB (median + {margin_db} dB margin)")
    
    return float(energy_threshold)


async def run_live_replay_capture(
    scenario: LiveScenario,
    input_device: int,
    output_device: int,
    human_output_device: int | None = None,
    gated_playback: bool = False,
    gated_resume_delay_s: float = 0.2,
) -> dict:
    """Replay pre-recorded captured audio through speaker(s), re-capture through mic.
    
    If human_output_device is provided, plays TTS through output_device and
    human speech through human_output_device. This separates the acoustic
    sources for more realistic overlap testing.
    
    The human speech timing is extracted from the scenario's metadata
    (speak_windows) and the agent_pcm is used as the clean TTS reference.
    """
    import sounddevice as sd
    from scipy.signal import resample as scipy_resample
    
    # Load pre-recorded audio and metadata
    replay_audio_raw = np.load(scenario.captured_audio_file).astype(np.float32)
    agent_pcm_raw = np.load(scenario.agent_pcm_file).astype(np.float32)
    
    # Read the actual sample rate from metadata
    meta_path = Path(scenario.captured_audio_file).parent / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        input_sr = meta.get("input_sr", SR)
        agent_start_s = meta.get("agent_start_s", 0.5)
        human_delay_s = meta.get("human_delay_s", scenario.human_delay_s)
        speak_windows = meta.get("speak_windows", [])
    else:
        input_sr = SR
        agent_start_s = 0.5
        human_delay_s = scenario.human_delay_s
        speak_windows = []
    
    # Resample captured audio to working rate (SR = 16kHz) for attribution.
    # Note: agent_pcm is always recorded at SR (16kHz) regardless of the
    # recording's input_sr. Only the captured_audio needs resampling.
    if input_sr != SR:
        replay_audio_16k = scipy_resample(
            replay_audio_raw,
            int(len(replay_audio_raw) * SR / input_sr)
        ).astype(np.float32)
    else:
        replay_audio_16k = replay_audio_raw
    
    # Agent PCM is already at SR — use directly
    agent_pcm = agent_pcm_raw.astype(np.float32)
    
    # Determine stream sample rates
    tts_sr = int(sd.query_devices(output_device)["default_samplerate"])
    
    # Resample TTS from SR to output rate
    if tts_sr != SR:
        tts_output = scipy_resample(
            agent_pcm,
            int(len(agent_pcm) * tts_sr / SR)
        ).astype(np.float32)
    else:
        tts_output = agent_pcm.copy()
    
    # If dual-speaker mode, prepare clean human speech output
    dual_output = human_output_device is not None
    if dual_output:
        human_sr = int(sd.query_devices(human_output_device)["default_samplerate"])
        # Load clean human speech if available, otherwise fall back to captured audio
        if scenario.human_audio_file and Path(scenario.human_audio_file).exists():
            human_pcm = np.load(scenario.human_audio_file).astype(np.float32)
            print(f"  Human speech: {scenario.human_audio_file} ({len(human_pcm)} samples)")
        else:
            human_pcm = replay_audio_16k
            print(f"  Human speech: using captured audio (no clean sample)")
        if human_sr != SR:
            human_output = scipy_resample(
                human_pcm,
                int(len(human_pcm) * human_sr / SR)
            ).astype(np.float32)
        else:
            human_output = human_pcm.copy()
    
    # Total duration
    total_duration_s = len(replay_audio_16k) / SR + 1.0
    # Total duration and output buffer
    total_duration_s = len(replay_audio_16k) / SR + 1.0
    total_samples = int(total_duration_s * tts_sr)
    
    # TTS output buffer is rendered after human timing is known so gated mode
    # can defer or split the physical playback around the human turn.
    tts_buffer = np.zeros(total_samples, dtype=np.float32)
    tts_segments: list[dict] = []
    
    # Build human speech output buffer (positioned at correct offset)
    dual_output = human_output_device is not None
    human_buffer = np.zeros(total_samples, dtype=np.float32)
    human_window_start_s = None
    human_window_end_s = None
    
    if dual_output and scenario.human_audio_file and Path(scenario.human_audio_file).exists():
        human_pcm = np.load(scenario.human_audio_file).astype(np.float32)
        # Resample to output rate
        if tts_sr != SR:
            human_pcm_out = scipy_resample(
                human_pcm,
                int(len(human_pcm) * tts_sr / SR)
            ).astype(np.float32)
        else:
            human_pcm_out = human_pcm.copy()
        
        # Position human speech at agent_start_s + human_delay_s
        human_start_s = agent_start_s + human_delay_s
        if human_start_s >= 0:
            human_start_sample = int(human_start_s * tts_sr)
            human_source_start = 0
        else:
            # Human starts before capture/playback time 0. Trim the leading
            # portion so the scheduled utterance is already in progress when
            # the live run starts.
            human_start_sample = 0
            human_source_start = min(int((-human_start_s) * tts_sr), len(human_pcm_out))
        human_end_sample = min(
            human_start_sample + len(human_pcm_out) - human_source_start,
            total_samples,
        )
        human_buffer[human_start_sample:human_end_sample] = human_pcm_out[
            human_source_start:human_source_start + (human_end_sample - human_start_sample)
        ]
        print(f"  Human speech: {len(human_pcm_out)} samples at t={max(0, human_start_s):.2f}s")
        human_window_start_s = human_start_sample / tts_sr
        human_window_end_s = human_end_sample / tts_sr
    elif dual_output:
        print(f"  Human speech: none (no clean sample)")
        dual_output = False

    if gated_playback:
        tts_segments = plan_controller_gated_tts_segments(
            agent_start_s=agent_start_s,
            agent_duration_s=len(tts_output) / tts_sr,
            human_start_s=human_window_start_s,
            human_end_s=human_window_end_s,
            expected_behavior=scenario.expected_behavior,
            resume_delay_s=gated_resume_delay_s,
        )
        required_end_s = max(
            [total_duration_s]
            + [seg["dst_start_s"] + seg["duration_s"] + 0.5 for seg in tts_segments]
            + ([human_window_end_s + 0.5] if human_window_end_s is not None else [])
        )
        if required_end_s > total_duration_s:
            old_human_buffer = human_buffer
            total_duration_s = required_end_s
            total_samples = int(total_duration_s * tts_sr)
            human_buffer = np.zeros(total_samples, dtype=np.float32)
            human_buffer[:len(old_human_buffer)] = old_human_buffer
        tts_buffer = build_controller_gated_tts_buffer(
            tts_output,
            sr=tts_sr,
            total_samples=total_samples,
            segments=tts_segments,
        )
        print(f"  Gated TTS playback: {len(tts_segments)} segment(s)")
        for seg in tts_segments:
            print(
                "    src={src_start_s:.2f}s dst={dst_start_s:.2f}s dur={duration_s:.2f}s".format(**seg)
            )
    else:
        tts_segments = [{
            "src_start_s": 0.0,
            "dst_start_s": _round_seconds(agent_start_s),
            "duration_s": _round_seconds(len(tts_output) / tts_sr),
        }]
        tts_buffer = build_controller_gated_tts_buffer(
            tts_output,
            sr=tts_sr,
            total_samples=total_samples,
            segments=tts_segments,
        )
    
    print(f"  Replay audio: {len(replay_audio_16k)} samples at {SR}Hz ({len(replay_audio_16k)/SR:.2f}s)")
    print(f"  Agent PCM:    {len(agent_pcm)} samples at {SR}Hz ({len(agent_pcm)/SR:.2f}s)")
    print(f"  TTS output:   device {output_device} at {tts_sr}Hz")
    if dual_output:
        print(f"  Human output: device {human_output_device} at {tts_sr}Hz")
    print(f"  Mic input:    device {input_device}")
    print(f"  Duration:     {total_duration_s:.1f}s")
    
    # Capture buffer (at input sample rate)
    input_sr_actual = int(sd.query_devices(input_device)["default_samplerate"])
    capture_samples = int(total_duration_s * input_sr_actual)
    mic_capture = np.zeros(capture_samples, dtype=np.float32)
    
    samples_played = [0]
    samples_captured = [0]
    
    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"  Audio status: {status}")
        
        # Output TTS
        start = samples_played[0]
        end = min(start + frames, total_samples)
        if start < total_samples:
            n = end - start
            outdata[:n, 0] = tts_buffer[start:end]
            if n < frames:
                outdata[n:, 0] = 0
        else:
            outdata[:] = 0
        
        # Capture mic
        cap_start = samples_captured[0]
        cap_end = min(cap_start + frames, capture_samples)
        if cap_start < capture_samples:
            n = cap_end - cap_start
            mic_capture[cap_start:cap_end] = indata[:n, 0]
            samples_captured[0] = cap_end
        
        samples_played[0] = end
    
    if dual_output:
        # Dual-output mode: sd.Stream for mic input + TTS output,
        # plus separate sd.OutputStream for human speech
        human_samples_played = [0]
        
        def human_callback(outdata, frames, time_info, status):
            start = human_samples_played[0]
            end = min(start + frames, total_samples)
            if start < total_samples:
                n = end - start
                outdata[:n, 0] = human_buffer[start:end]
                if n < frames:
                    outdata[n:, 0] = 0
            else:
                outdata[:] = 0
            human_samples_played[0] = end
        
        print(f"  Starting dual-output capture...")
        with sd.Stream(
            samplerate=tts_sr,
            blocksize=int(VAD_FRAME_SAMPLES * tts_sr / SR),
            dtype="float32",
            channels=1,
            callback=audio_callback,
            device=(input_device, output_device),
        ):
            with sd.OutputStream(
                samplerate=tts_sr,
                blocksize=int(VAD_FRAME_SAMPLES * tts_sr / SR),
                dtype="float32",
                channels=1,
                callback=human_callback,
                device=human_output_device,
            ):
                await asyncio.sleep(total_duration_s + 0.5)
    else:
        # Single-output mode (original)
        print(f"  Starting single-output capture...")
        with sd.Stream(
            samplerate=tts_sr,
            blocksize=int(VAD_FRAME_SAMPLES * tts_sr / SR),
            dtype="float32",
            channels=1,
            callback=audio_callback,
            device=(input_device, output_device),
        ):
            await asyncio.sleep(total_duration_s + 0.5)
    
    elapsed = total_duration_s  # Approximate
    
    # Trim capture
    mic_capture = mic_capture[:samples_captured[0]]
    
    # Resample capture to 16kHz for attribution
    if input_sr_actual != SR:
        from scipy.signal import resample as scipy_resample
        live_captured = scipy_resample(
            mic_capture,
            int(len(mic_capture) * SR / input_sr_actual)
        ).astype(np.float32)
    else:
        live_captured = mic_capture
    
    print(f"  Capture complete: {len(live_captured)} samples at {SR}Hz")
    
    return {
        "live_captured": live_captured,
        "agent_pcm": agent_pcm,
        "replay_audio": replay_audio_16k,
        "agent_start_s": tts_segments[0]["dst_start_s"] if tts_segments else agent_start_s,
        "original_agent_start_s": agent_start_s,
        "human_start_s": human_window_start_s,
        "human_end_s": human_window_end_s,
        "tts_segments": tts_segments,
        "gated_playback": gated_playback,
        "capture_time_s": elapsed,
    }


async def run_gated_scenario(scenario: LiveScenario, args: argparse.Namespace) -> dict:
    """Run a single live E2E scenario with controller-gated TTS playback."""
    print(f"\n{'='*60}")
    print(f"Scenario (gated): {scenario.name}")
    print("  Controller-gated physical playback enabled.")
    print(f"{'='*60}")
    return await run_scenario(scenario, args, gated_playback=True)


async def run_scenario(
    scenario: LiveScenario,
    args: argparse.Namespace,
    gated_playback: bool = False,
) -> dict:
    """Run a single live E2E scenario."""


def run_attribution_pipeline(
    captured_audio: np.ndarray,
    agent_pcm: np.ndarray,
    attributor: ThalamicAttributor,
    vad: SmoothedVAD,
    agent_start_s: float = 0.0,
) -> list[dict]:
    """Run the full attribution pipeline on captured audio.
    
    Returns a list of frame records compatible with frames_from_frame_records().
    """
    frame_records = []
    frame_step = VAD_FRAME_SAMPLES
    num_frames = max(0, (len(captured_audio) - frame_step) // frame_step)
    agent_start_sample = int(agent_start_s * SR)
    
    # Compute global self score
    from scripts.m2_speech_preservation_harness import normalized_xcorr
    global_score, _ = normalized_xcorr(captured_audio, agent_pcm)
    
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_step
        if end > len(captured_audio):
            break
        frame_audio = captured_audio[start:end]
        
        # VAD
        is_speech = vad.process_frame(frame_audio)
        
        # RMS
        rms = np.sqrt(np.mean(frame_audio ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        
        # Self-match score
        is_playing = start >= agent_start_sample and end <= agent_start_sample + len(agent_pcm)
        match_score = global_score if is_playing else 0.0
        
        # Attribution
        result = attributor.attribute(
            match_score=match_score,
            observed_onset_samples=start,
            expected_onset_samples=agent_start_sample,
            rms_db=rms_db,
            is_playing=is_playing,
            self_pcm=agent_pcm,
            agent_start_sample=0,
            frame_audio=frame_audio,
        )
        
        frame_records.append({
            "frame_index": i,
            "start_s": start / SR,
            "verdict": result.verdict.name,
            "confidence": result.confidence,
            "self_match_score": result.self_match_score,
            "self_match_score_per_frame": result.self_match_score_per_frame,
            "is_in_motor_window": result.is_in_motor_window,
            "causal_score": result.causal_score,
            "energy_db": result.energy_db,
            "is_speech": is_speech,
            "rms_db": rms_db,
        })
    
    return frame_records


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class LiveMetrics:
    scenario: str
    expected_behavior: str = "interrupt_defer_resume"
    total_frames: int = 0
    false_interrupts: int = 0
    false_defers: int = 0
    missed_interrupts: int = 0
    true_interrupts: int = 0
    successful_resumes: int = 0
    total_resumes_expected: int = 0
    state_transitions: list = field(default_factory=list)
    latency_ms: float = 0.0
    notes: list = field(default_factory=list)
    
    def summary(self) -> dict:
        return {
            "scenario": self.scenario,
            "expected_behavior": self.expected_behavior,
            "total_frames": self.total_frames,
            "false_interrupts": self.false_interrupts,
            "false_defers": self.false_defers,
            "missed_interrupts": self.missed_interrupts,
            "true_interrupts": self.true_interrupts,
            "successful_resumes": self.successful_resumes,
            "total_resumes_expected": self.total_resumes_expected,
            "resume_rate": (
                f"{self.successful_resumes}/{self.total_resumes_expected}"
                if self.total_resumes_expected else "n/a"
            ),
            "state_transitions": self.state_transitions,
            "latency_ms": round(self.latency_ms, 1),
            "notes": self.notes,
        }


def annotate_metric_mode(summary: dict, gated_playback: bool) -> dict:
    """Mark whether metrics should be interpreted as reflex or gated-suppression.

    In gated mode, physical playback is intentionally scheduled to avoid human
    speech windows, so legacy interrupt/resume fields are diagnostic only: a
    successful suppression run can legitimately report MI>0/TI=0/resume=0/N.
    """
    annotated = dict(summary)
    notes = list(annotated.get("notes") or [])
    if gated_playback:
        annotated["metric_mode"] = "gated_suppression"
        annotated["reflex_metrics_are_diagnostic"] = True
        notes.append(
            "Gated playback mode: legacy interrupt/resume metrics are diagnostic; "
            "use suppression analyzer fields as pass/fail criteria."
        )
    else:
        annotated["metric_mode"] = "reflex_interrupt_resume"
        annotated["reflex_metrics_are_diagnostic"] = False
    annotated["notes"] = notes
    return annotated


def compute_live_metrics(
    scenario: LiveScenario,
    frame_records: list[dict],
    decisions: list,
    accepted_initial_output: bool,
    resumed_output: str | None,
    transitions: list[str],
    latency_ms: float,
) -> LiveMetrics:
    """Compute behavior-aware M3 live metrics from controller outputs.

    Only `interrupt_defer_resume` scenarios are scored as overlap interrupt +
    resume tasks. `defer` and `respond` are distinct conversation phases and
    should not be counted as ordinary overlap-interrupt/resume successes.
    """
    metrics = LiveMetrics(scenario=scenario.name, expected_behavior=scenario.expected_behavior)
    metrics.total_frames = len(decisions)
    metrics.state_transitions = transitions
    metrics.latency_ms = latency_ms

    expects_interrupt_resume = scenario.expected_behavior == "interrupt_defer_resume"
    expects_defer = scenario.expected_behavior == "defer"
    expects_respond = scenario.expected_behavior == "respond"

    saw_expected_overlap = False
    saw_interrupt = False
    for i, d in enumerate(decisions):
        if i >= len(frame_records):
            continue
        fr = frame_records[i]
        is_speech = fr.get("is_speech", False)
        is_motor = fr.get("is_in_motor_window", False)
        verdict = fr.get("verdict", "")

        gt_human = fr.get("gt_human_speaking")
        if gt_human is None:
            gt_human = verdict == "BOTH" and is_speech and is_motor
        gt_human_during = bool(gt_human and is_motor)
        if gt_human_during:
            saw_expected_overlap = True

        if d.should_interrupt:
            saw_interrupt = True
            if expects_interrupt_resume and gt_human_during:
                metrics.true_interrupts += 1
            elif expects_defer:
                metrics.notes.append(
                    "Observed interrupt after initial defer; ignored because fixed-playback harness still emits TTS"
                )
            else:
                metrics.false_interrupts += 1

    if expects_interrupt_resume and saw_expected_overlap and not saw_interrupt:
        metrics.missed_interrupts += 1

    if expects_defer and accepted_initial_output:
        metrics.false_defers += 1
        metrics.notes.append("Expected pre-agent defer, but initial output was accepted")

    if expects_respond and saw_interrupt:
        metrics.notes.append("Expected post-agent respond/listen behavior, but controller interrupted")

    metrics.total_resumes_expected = 1 if expects_interrupt_resume else 0
    metrics.successful_resumes = 1 if (expects_interrupt_resume and resumed_output) else 0
    if expects_respond and resumed_output:
        metrics.notes.append("Unexpected resumed output for non-interrupt scenario")

    return metrics


# ── Main harness ──────────────────────────────────────────────────────────────

async def run_scenario(
    scenario: LiveScenario,
    args: argparse.Namespace,
    gated_playback: bool = False,
) -> dict:
    """Run a single live E2E scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"{'='*60}")
    
    # 1. Run live replay + capture
    print(f"\n[1/3] Running live replay + capture...")
    t0 = time.time()
    capture_result = await run_live_replay_capture(
        scenario, args.input_device, args.output_device,
        human_output_device=args.human_output_device,
        gated_playback=gated_playback,
        gated_resume_delay_s=getattr(args, "gated_resume_delay_ms", 200.0) / 1000.0,
    )
    capture_time = (time.time() - t0) * 1000
    
    live_captured = capture_result["live_captured"]
    agent_pcm = capture_result["agent_pcm"]
    agent_start_s = capture_result.get("agent_start_s", 0.0)
    original_agent_start_s = capture_result.get("original_agent_start_s", agent_start_s)
    human_start_s = capture_result.get("human_start_s")
    human_end_s = capture_result.get("human_end_s")
    tts_segments = capture_result.get("tts_segments", [])
    
    # 1.5. Calibrate noise floor (LISTEN phase)
    if args.calibrate:
        energy_threshold_db = await calibrate_noise_floor(
            args.input_device,
            duration_s=args.calibrate_duration,
            margin_db=args.calibrate_margin_db,
        )
    else:
        energy_threshold_db = -55.0
    
    # 2. Run attribution pipeline
    print(f"\n[2/3] Running attribution pipeline...")

    # Load pre-fitted room filter for room-local scoring (production-causal path)
    room_filter = None
    room_local_threshold = args.room_local_threshold
    if args.room_filter:
        room_filter = np.load(args.room_filter)
        print(f"  Room filter loaded: {args.room_filter} ({len(room_filter)} taps)")
        # Auto-detect threshold from companion calibration JSON
        if room_local_threshold is None:
            cal_path = Path(args.room_filter).with_suffix(".json")
            if not cal_path.exists():
                cal_path = Path(str(args.room_filter).replace(".npy", ".json"))
            if cal_path.exists():
                cal = json.loads(cal_path.read_text())
                rec = cal.get("recommended_threshold", {})
                if isinstance(rec, dict):
                    room_local_threshold = rec.get("threshold", 0.04)
                elif isinstance(rec, (int, float)):
                    room_local_threshold = float(rec)
                else:
                    room_local_threshold = 0.04
                print(f"  Auto threshold from {cal_path.name}: {room_local_threshold:.3f}")
            if room_local_threshold is None:
                room_local_threshold = 0.04
                print(f"  Using default threshold: {room_local_threshold:.3f}")

    # Build M4 speaker-embedding detector from the agent's self PCM
    m4_detector = None
    if args.m4_enable:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
            from stg.speaker import SpeakerIdentifier, SelfVoiceDetector
            identifier = SpeakerIdentifier(use_real=False)  # mock mode for validation
            m4_detector = SelfVoiceDetector(identifier, threshold=args.m4_self_threshold)
            m4_detector.register_self(agent_pcm.astype(np.float32))
            print(f"  M4 detector enabled (mock mode, threshold={args.m4_self_threshold})")
        except Exception as e:
            print(f"  WARNING: M4 detector failed to initialize: {e}")
            m4_detector = None

    attributor = ThalamicAttributor(
        energy_threshold_db=energy_threshold_db,
        similarity_threshold=0.05,
        causal_tolerance_ms=175.0,
        room_filter=room_filter,
        room_local_threshold=room_local_threshold if room_filter is not None else None,
        m4_detector=m4_detector,
        m4_self_threshold=args.m4_self_threshold,
        m4_min_buffer_ms=args.m4_min_buffer_ms,
    )
    vad = SmoothedVAD()
    
    t0 = time.time()
    frame_records = run_attribution_pipeline(
        live_captured, agent_pcm, attributor, vad,
        agent_start_s=agent_start_s,
    )
    attr_time = (time.time() - t0) * 1000
    
    print(f"  Attribution time: {attr_time:.0f}ms")
    print(f"  Frames: {len(frame_records)}")
    
    # Count verdicts
    verdict_counts = {}
    for fr in frame_records:
        v = fr["verdict"].name if hasattr(fr["verdict"], "name") else str(fr["verdict"])
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print(f"  Verdicts: {dict(verdict_counts)}")

    # Annotate records with scheduled clean-human playback ground truth when
    # dual-speaker mode is active. This avoids treating every BOTH frame as
    # human speech; BOTH can also arise from self-playback artifacts.
    if human_start_s is not None and human_end_s is not None:
        for fr in frame_records:
            t = fr.get("start_s", 0.0)
            fr["gt_human_speaking"] = human_start_s <= t < human_end_s
    else:
        for fr in frame_records:
            fr["gt_human_speaking"] = None
    
    # 3. Run controller
    print(f"\n[3/3] Running controller...")
    controller = AgentiveController(
        listen_threshold_ms=args.listen_threshold_ms,
        cooldown_ms=args.cooldown_ms,
        both_external_min_frames=args.both_external_min_frames,
        both_external_energy_ratio_threshold=args.both_external_energy_ratio,
    )
    
    # Create frames directly from attribution results (preserving self_match_score_per_frame etc.)
    from pfc.m2_artifact_replay import M2Frame, attribution as attr_fn
    frames = []
    for fr in frame_records:
        # Use the full attribution result if available, otherwise create a basic one
        if 'self_match_score_per_frame' in fr:
            # Reconstruct the AttributionResult from the record
            verdict = SourceVerdict[fr.get('verdict', 'SILENCE')]
            attr = attr_fn(
                verdict,
                bool(fr.get('is_speech', False)),
                float(fr.get('energy_db', -30.0)),
                is_in_motor_window=fr.get('is_in_motor_window', None),
            )
            # Override with the actual values from attribution
            object.__setattr__(attr, 'self_match_score_per_frame', fr.get('self_match_score_per_frame'))
            object.__setattr__(attr, 'self_match_score', fr.get('self_match_score', 0.0))
            object.__setattr__(attr, 'causal_score', fr.get('causal_score', 0.0))
            object.__setattr__(attr, 'confidence', fr.get('confidence', 0.9))
        else:
            verdict = SourceVerdict[fr.get('verdict', 'SILENCE')]
            attr = attr_fn(verdict, bool(fr.get('is_speech', False)), float(fr.get('energy_db', -30.0)))
        frames.append(M2Frame(float(fr.get('start_s', 0.0)), attr))
    
    if frames:
        frames.append(M2Frame(frames[-1].time_s + 0.35, attr_fn(SourceVerdict.SILENCE, False, -70.0)))
    
    t0 = time.time()
    bridge = run_turn_taking_bridge(
        frames,
        scenario.agent_text,
        controller=controller,
        request_at_s=controller_request_time_s(
            scenario.expected_behavior,
            original_agent_start_s=original_agent_start_s,
        ),
    )
    ctrl_time = (time.time() - t0) * 1000
    
    print(f"  Controller time: {ctrl_time:.0f}ms")
    print(f"  Accepted initial: {bridge.accepted_initial_output}")
    print(f"  Resumed: {bridge.resumed_output}")
    
    # State transitions
    states = [d.state.name for d in bridge.decisions]
    transitions = []
    prev = None
    for s in states:
        if s != prev:
            transitions.append(s)
            prev = s
    print(f"  States: {' -> '.join(transitions[:15])}")
    
    metrics = compute_live_metrics(
        scenario=scenario,
        frame_records=frame_records,
        decisions=bridge.decisions,
        accepted_initial_output=bridge.accepted_initial_output,
        resumed_output=bridge.resumed_output,
        transitions=transitions,
        latency_ms=capture_time + attr_time + ctrl_time,
    )
    
    print(f"\n{'='*60}")
    print(f"Results for {scenario.name}:")
    summary = annotate_metric_mode(metrics.summary(), gated_playback=gated_playback)
    summary["gated_playback"] = gated_playback
    summary["tts_segments"] = tts_segments
    for k, v in summary.items():
        if k not in ("state_transitions", "notes"):
            print(f"  {k}: {v}")
    print(f"{'='*60}")
    
    # Save artifacts
    if args.save_dir:
        save_path = Path(args.save_dir) / scenario.name
        save_path.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path / "live_captured.npy", live_captured)
        np.save(save_path / "agent_pcm.npy", agent_pcm)
        
        serializable_records = []
        for fr in frame_records:
            sr = dict(fr)
            if hasattr(sr["verdict"], "name"):
                sr["verdict"] = sr["verdict"].name
            serializable_records.append(sr)
        
        with open(save_path / "frame_records.json", "w") as f:
            json.dump(serializable_records, f, indent=2, default=str)
        
        with open(save_path / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n  Saved artifacts to: {save_path}")
    
    return summary


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="M3 Live E2E Validation Harness")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a single scenario")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--input-device", type=int, default=3)
    parser.add_argument("--output-device", type=int, default=4)
    parser.add_argument("--listen-threshold-ms", type=float, default=100.0)
    parser.add_argument("--cooldown-ms", type=float, default=200.0)
    parser.add_argument("--save-dir", default=None, help="Directory to save artifacts")
    parser.add_argument("--calibrate", action="store_true",
                        help="Enable ambient noise calibration before each scenario")
    parser.add_argument("--calibrate-duration", type=float, default=2.0,
                        help="Duration of calibration listening phase (seconds)")
    parser.add_argument("--calibrate-margin-db", type=float, default=15.0,
                        help="Margin above noise floor for energy threshold (dB)")
    parser.add_argument("--human-output-device", type=int, default=None,
                        help="External speaker device for human speech (enables dual-speaker mode)")
    parser.add_argument("--both-external-min-frames", type=int, default=7,
                        help="Consecutive frames for overlap detection (default 7)")
    parser.add_argument("--both-external-energy-ratio", type=float, default=2.0,
                        help="Energy ratio threshold for overlap detection (default 2.0)")
    parser.add_argument("--room-filter", default=None,
                        help="Path to pre-fitted room filter .npy file for room-local scoring")
    parser.add_argument("--room-local-threshold", type=float, default=None,
                        help="Room-local score threshold for SELF override (default: auto from filter file)")
    parser.add_argument("--m4-enable", action="store_true",
                        help="Enable M4 speaker-embedding self-voice detector")
    parser.add_argument("--m4-min-buffer-ms", type=float, default=60.0,
                        help="M4 minimum buffer ms (default 60)")
    parser.add_argument("--m4-self-threshold", type=float, default=0.50,
                        help="M4 self-decision threshold (default 0.50)")
    parser.add_argument("--gated-playback", action="store_true",
                        help="Physically gate TTS playback around scheduled/controller-detected human speech")
    parser.add_argument("--gated-resume-delay-ms", type=float, default=200.0,
                        help="Delay before resuming gated TTS after human speech ends (default 200ms)")

    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable scenarios:")
        for name, sc in SCENARIOS.items():
            print(f"  {name:25s} — {sc.description}")
        return
    
    if not args.scenario and not args.all:
        parser.print_help()
        return
    
    scenarios_to_run = list(SCENARIOS.values()) if args.all else [SCENARIOS[args.scenario]]
    
    all_metrics = []
    for scenario in scenarios_to_run:
        try:
            if args.gated_playback:
                metrics = await run_gated_scenario(scenario, args)
            else:
                metrics = await run_scenario(scenario, args)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nERROR in scenario {scenario.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if len(all_metrics) > 1:
        print(f"\n\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        total_fi = sum(m.get("false_interrupts", 0) for m in all_metrics)
        total_fd = sum(m.get("false_defers", 0) for m in all_metrics)
        total_mi = sum(m.get("missed_interrupts", 0) for m in all_metrics)
        total_ti = sum(m.get("true_interrupts", 0) for m in all_metrics)
        total_sr = sum(m.get("successful_resumes", 0) for m in all_metrics)
        total_tr = sum(m.get("total_resumes_expected", 0) for m in all_metrics)
        
        print(f"  False interrupts:  {total_fi}")
        print(f"  False defers:      {total_fd}")
        print(f"  Missed interrupts: {total_mi}")
        print(f"  True interrupts:   {total_ti}")
        print(f"  Resume rate:       {total_sr}/{total_tr}")
        print(f"{'='*60}")
    
    if args.save_dir:
        overall_path = Path(args.save_dir) / "live_e2e_results.json"
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        with open(overall_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nSaved overall results to: {overall_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
