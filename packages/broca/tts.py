"""Broca — TTS speech synthesis.

M0: Wraps edge-tts (free, no API key) to generate 16kHz PCM audio.
The efference copy of the output PCM is sent to the Cerebellum for
forward-model prediction.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import wave
import edge_tts
import numpy as np

# ── Global audio constants ────────────────────────────────────────────────
SR = 16000          # sample rate, Hz

# Default voice: en-US-AriaNeural (female, clear, widely available)
DEFAULT_VOICE = "en-US-AriaNeural"


async def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> np.ndarray:
    """Generate speech using edge-tts and return float32 PCM at SR.

    Parameters
    ----------
    text : str
        Text to synthesize.
    voice : str
        Edge-TTS voice identifier (e.g. "en-US-AriaNeural").
    rate : str
        Speed adjustment (e.g. "+20%", "-10%").
    pitch : str
        Pitch adjustment (e.g. "+10Hz", "-5Hz").

    Returns
    -------
    np.ndarray
        float32 PCM audio at 16kHz (or whatever SR is set to).
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    response: bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            response += chunk["data"]

    # Edge-tts returns MP3; decode via a simple WAV round-trip
    # Use sox/ffmpeg if available, otherwise simple resample hack
    return _decode_mp3_to_pcm(response)


def _decode_mp3_to_pcm(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes to float32 PCM.

    Tries ffmpeg/avconv first for accuracy. Falls back to
    a pure-Python approximation if neither is available.
    """
    import subprocess
    import os

    # Try ffmpeg (should be available on macOS)
    ffmpeg = None
    for cmd in ["ffmpeg", "avconv"]:
        try:
            subprocess.run(
                [cmd, "-version"],
                capture_output=True,
                check=True,
                timeout=5,
            )
            ffmpeg = cmd
            break
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if ffmpeg is None:
        # Fallback: try mutagen/pydub, last resort raise
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            segment = segment.set_frame_rate(SR).set_channels(1)
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            if segment.sample_width == 2:
                samples /= 32768.0
            elif segment.sample_width == 4:
                samples /= 2147483648.0
            return samples
        except ImportError:
            raise RuntimeError(
                "Cannot decode MP3: install ffmpeg or pydub. "
                "On macOS: brew install ffmpeg"
            )

    # ffmpeg path decode
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
        f_mp3.write(mp3_bytes)
        mp3_path = f_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        # Decode MP3 at native rate first, then resample to target SR
        # This avoids ffmpeg's low-quality resampling of low-bitrate MP3
        raw_wav_path = mp3_path.replace(".mp3", "_raw.wav")
        result = subprocess.run(
            [
                ffmpeg, "-y", "-i", mp3_path,
                "-ar", "24000", "-ac", "1",
                "-f", "s16le", "-c:a", "pcm_s16le",
                raw_wav_path,
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        with open(raw_wav_path, "rb") as f:
            raw = f.read()
        samples_24k = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        
        # High-quality resample from 24kHz to target SR
        from scipy.signal import resample as _resamp
        n_target = int(len(samples_24k) * SR / 24000)
        samples = _resamp(samples_24k, n_target).astype(np.float32)
        
        # Clean up temp files
        for p in [mp3_path, raw_wav_path]:
            if os.path.exists(p):
                os.unlink(p)
        
        # Trim leading silence
        if len(samples) > 0:
            peak = float(np.max(np.abs(samples)))
            if peak > 0.0:
                threshold = peak * 0.01
                onset = np.where(np.abs(samples) > threshold)[0]
                if len(onset):
                    samples = samples[int(onset[0]):]

        return samples
    finally:
        for p in [mp3_path]:
            if os.path.exists(p):
                os.unlink(p)


def list_voices() -> list[dict]:
    """List available edge-tts voices (synchronous, cached)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    voices = loop.run_until_complete(edge_tts.list_voices())
    return [
        {"name": v["ShortName"], "locale": v["Locale"], "gender": v.get("Gender", "")}
        for v in sorted(voices, key=lambda v: v.get("ShortName", ""))
    ]
