#!/usr/bin/env python3
"""M1 Remote Capture — HTTP Server (Mac side).

Runs on the Mac. Provides two endpoints:
  POST /tts      — phone requests test utterance (server plays speakers + returns TTS audio)
  POST /upload   — phone uploads recorded audio for analysis

The phone acts as the mobile mic (record in any room/location),
the Mac handles TTS, ASR, and gating.

Usage:
    python scripts/m1_remote_capture_server.py

    # Find your Mac IP for the phone client:
    ipconfig getifaddr en0
    # → 192.168.1.50 → put in phone browser: http://192.168.1.50:8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

# Add packages to sys.path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from broca.tts import synthesize
from cochlea.capture import SR, pcm_to_mels, EfferenceBuffer
from stg.asr import transcribe, DEFAULT_MODEL_SIZE
from thalamus.gate import ThalamicGate


def play_mac_speakers(pcm: np.ndarray, sample_rate: int = SR, device: int | None = None):
    """Play audio through Mac speakers."""
    import sounddevice as sd
    sd.play(pcm, samplerate=sample_rate, device=device, blocking=False)
    sd.wait()


def compute_similarity(original: str, transcript: str) -> float:
    orig_words = set(original.lower().split())
    trans_words = set(transcript.lower().split())
    if not orig_words and not trans_words:
        return 1.0
    if not orig_words or not trans_words:
        return 0.0
    return len(orig_words & trans_words) / len(orig_words | trans_words)


def analyze_and_report(audio: np.ndarray, text: str, tts_pcm: np.ndarray) -> dict:
    """Run ASR + gating on captured audio."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_db = 20 * np.log10(rms) if rms > 1e-10 else -100

    if rms_db < -50:
        return {"error": f"Signal too quiet: {rms_db:.1f} dB"}

    # ASR
    asr_result = transcribe(audio, sample_rate=SR, model_size=DEFAULT_MODEL_SIZE)
    similarity = compute_similarity(text, asr_result.text)

    # Gating
    observed_mel = pcm_to_mels(audio)
    predicted_mel = pcm_to_mels(tts_pcm)

    if predicted_mel.shape[0] < observed_mel.shape[0]:
        pad = np.zeros(
            (observed_mel.shape[0] - predicted_mel.shape[0], predicted_mel.shape[1]),
            dtype=np.float32
        )
        full_efference_mel = np.concatenate([predicted_mel, pad], axis=0)
    else:
        full_efference_mel = predicted_mel[:observed_mel.shape[0]]

    gate = ThalamicGate()
    result = gate.gate(observed_mel, full_efference_mel)

    return {
        "rms_db": round(rms_db, 1),
        "original": text,
        "asr_output": asr_result.text,
        "similarity": round(similarity, 3),
        "svsr_db": round(result["svsr"], 1),
        "gate_verdict": "world" if result["is_world"] else "self_suppressed",
        "asr_text": asr_result.text,
    }


# Global: store the latest TTS PCM so we can gate against it
_latest_tts_pcm: np.ndarray | None = None
_latest_text: str = ""


class CorollaryHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  [http] {args[0]} {args[1]}")

    def do_GET(self):
        if self.path == "/":
            self._send_file("client.html", "text/html")
        elif self.path == "/client.html":
            self._send_file("client.html", "text/html")
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        global _latest_tts_pcm, _latest_text

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        if self.path == "/tts":
            # Phone requests the test utterance
            params = json.loads(body) if body else {}
            text = params.get("text", "Hello, this is Corollary testing with a remote microphone.")

            print(f"  🗣️  TTS request: \"{text}\"")
            tts_pcm = asyncio.run(synthesize(text))
            _latest_tts_pcm = tts_pcm
            _latest_text = text
            tts_dur = len(tts_pcm) / SR

            # Play on Mac speakers so phone can hear it
            print(f"  🔊 Playing on Mac speakers ({tts_dur:.2f}s)...")
            play_mac_speakers(tts_pcm)
            print(f"  ✅ Playback complete — record now on phone")

            # Return TTS audio as base64 (phone can play it locally too)
            pcm_b64 = base64.b64encode(tts_pcm.tobytes()).decode()
            self._json({
                "text": text,
                "tts_b64": pcm_b64,
                "sample_rate": SR,
                "duration": round(tts_dur, 2),
            })

        elif self.path == "/upload":
            # Phone uploads recorded audio for analysis
            if _latest_tts_pcm is None:
                self._json({"error": "No TTS generated yet. Call /tts first."}, 400)
                return

            params = json.loads(body)
            audio_b64 = params.get("audio_b64")
            if not audio_b64:
                self._json({"error": "Missing audio_b64"}, 400)
                return

            # Decode: int16 PCM → float32 normalized
            pcm_bytes = base64.b64decode(audio_b64)
            audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            print(f"  📊 Analyzing {len(audio)} samples ({len(audio)/SR:.1f}s)...")
            result = analyze_and_report(audio, _latest_text, _latest_tts_pcm)
            self._json(result)

        else:
            self._json({"error": "not found"}, 404)

    def _json(self, data: dict, status: int = 200):
        response = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def _send_file(self, filename: str, content_type: str):
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self._json({"error": f"{filename} not found"}, 404)


def get_local_ip() -> str:
    """Get the Mac's Wi-Fi IP address."""
    import subprocess
    try:
        result = subprocess.run(
            ["ipconfig", "getifaddr", "en0"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "localhost"


def main():
    parser = argparse.ArgumentParser(description="M1 Remote Capture Server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    args = parser.parse_args()

    ip = get_local_ip()
    url = f"http://{ip}:{args.port}"

    print("=" * 60)
    print("  COROLLARY — M1 Remote Capture Server")
    print("=" * 60)
    print(f"\n  📡 On your phone, open: {url}")
    print(f"  (must be on the same Wi-Fi network)")
    print(f"\n  Workflow:")
    print(f"  1. Open URL on phone browser")
    print(f"  2. Walk to wherever you want to test")
    print(f"  3. Tap 'Play TTS' — Mac plays, phone can listen")
    print(f"  4. Tap 'Record & Analyze' — phone records, Mac processes")
    print(f"\n  Waiting for connections...\n")

    server = HTTPServer(("0.0.0.0", args.port), CorollaryHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopping server.")
        server.server_close()


if __name__ == "__main__":
    main()
