#!/usr/bin/env python3
"""M3 — Agentive Listening & Turn-Taking Demo.

Demonstrates social contingency:
1. Detects a world-agent talking via VAD + ThalamicAttributor.
2. Corollary wants to say something but REFRAINS while the world-agent is talking.
3. Corollary automatically speaks once the world-agent is silent.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is not installed.")
    sys.exit(1)

# Add corollary packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from cochlea.capture import SR
from cochlea.vad import SmoothedVAD, VAD_FRAME_SAMPLES
from thalamus.attributor import ThalamicAttributor, SourceVerdict
from pfc.agentive_controller import AgentiveController, AgentState
from broca.tts import synthesize

def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-10))

async def run_m3_demo(device: int):
    ctrl = AgentiveController(listen_threshold_ms=300, cooldown_ms=800)
    attributor = ThalamicAttributor(sr=SR)
    vad = SmoothedVAD()
    
    input_sr = int(sd.query_devices(device)["default_samplerate"] or SR)
    
    print(f"\n--- M3 Agentive Listening Demo ---")
    print(f"Device: {device}")
    print(f"Goal: Corollary will wait for YOU to finish talking before it responds.")
    print(f"Corollary wants to say: 'I hear you, please continue.'")
    print(f"\n[Corollary is IDLE and listening...]")

    # Prepare response
    response_pcm = await synthesize("I hear you, please continue.")

    # We'll use a 200ms chunk for processing
    CHUNK_MS = 200
    CHUNK_SAMPLES = int(input_sr * CHUNK_MS / 1000.0)
    
    stream = sd.InputStream(device=device, channels=1, samplerate=input_sr, dtype="float32", blocksize=1024)
    stream.start()

    has_requested_output = False
    demo_start_time = time.time()
    
    # Run for 15 seconds
    while time.time() - demo_start_time < 15.0:
        data, _ = stream.read(CHUNK_SAMPLES)
        audio_chunk = data[:, 0]
        
        # Resample for VAD if needed
        if input_sr != SR:
            import librosa
            chunk_16k = librosa.resample(audio_chunk, orig_sr=input_sr, target_sr=SR)
        else:
            chunk_16k = audio_chunk

        # Process frames in chunk
        num_vad_frames = len(chunk_16k) // VAD_FRAME_SAMPLES
        for i in range(num_vad_frames):
            frame = chunk_16k[i * VAD_FRAME_SAMPLES : (i+1) * VAD_FRAME_SAMPLES]
            is_speech = vad.process_frame(frame)
            rms_db = db(float(np.sqrt(np.mean(frame**2))))
            
            # Simple attribution for demo (not playing anything yet)
            res = attributor.attribute(
                match_score=0.0,
                observed_onset_samples=0,
                expected_onset_samples=None,
                rms_db=rms_db,
                is_playing=False,
                is_speech=is_speech
            )
            
            # Update PFC
            decision = ctrl.process_attribution(res)
            
        # UI logging
        if ctrl.state == AgentState.LISTEN:
            print("[LISTEN] Hearing agent...", end="\r")
        else:
            print("[IDLE] Listening...", end="\r")

        # Simulate PFC wanting to talk after 3 seconds
        if not has_requested_output and (time.time() - demo_start_time > 3.0):
            print("\n[PFC] Wants to respond!")
            accepted = ctrl.request_output("I hear you, please continue.")
            if accepted:
                print("[PFC] Accepted! Speaking now...")
                sd.play(response_pcm, samplerate=SR, blocking=True)
                has_requested_output = True
                print("[IDLE] Listening again...")
                ctrl.state = AgentState.IDLE # Reset state after speaking
            else:
                print("[PFC] Deferred! World-agent is still talking...")
                has_requested_output = True # We already requested, now waiting for trigger

        # If we have queued output, wait for IDLE state
        if has_requested_output and ctrl.has_deferred_output and ctrl.state == AgentState.IDLE:
            print("\n[PFC] World-agent finished! Executing deferred response...")
            sd.play(response_pcm, samplerate=SR, blocking=True)
            ctrl.pop_deferred_output()
            print("[IDLE] Listening again...")

    stream.stop()
    stream.close()
    print("\nDemo finished.")

if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run_m3_demo(args.device))
