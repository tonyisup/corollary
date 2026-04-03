"""Cochlea — room acoustics simulation.

Generates synthetic room impulse responses and convolves audio
to simulate real-world listening conditions for the M0 simulated loop.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

# ── Default room parameters ────────────────────────────────────────────────
DEFAULT_RT60 = 0.3        # reverberation time (seconds) — small room
DEFAULT_FS = 16000        # sample rate
DEFAULT_NOISE_DB = -60    # noise floor


def generate_room_ir(
    rt60: float = DEFAULT_RT60,
    fs: int = DEFAULT_FS,
    length: float = 1.0,
    seed: int = None,
) -> np.ndarray:
    """Generate a synthetic room impulse response.

    Models a small room with exponential decay (RT60-based) plus
    random comb-filter reflections to simulate diffuse reverberation.

    Parameters
    ----------
    rt60 : float
        Time (seconds) for a 60 dB decay. Smaller = drier room.
    fs : int
        Sample rate.
    length : float
        Total IR length in seconds (covers direct sound + reverb tail).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Room impulse response, shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    n_samples = int(length * fs)
    ir = np.zeros(n_samples, dtype=np.float32)

    # Direct sound (first sample)
    ir[0] = 1.0

    # Early reflections: sparse impulses with exponential amplitude decay
    n_early = max(3, int(rt60 * fs / 100))
    early_delays = np.sort(rng.randint(10, int(0.05 * fs), size=n_early))
    early_amps = np.exp(-np.linspace(0, 2, n_early))
    for delay, amp in zip(early_delays, early_amps):
        if delay < n_samples:
            ir[delay] += amp * rng.uniform(0.5, 1.0)

    # Diffuse reverb: exponential decay with noise
    reverb_start = int(0.05 * fs)
    reverb = rng.randn(n_samples - reverb_start).astype(np.float32)
    decay = np.exp(-3.0 * np.arange(len(reverb)) / (rt60 * fs))
    reverb *= decay
    reverb *= 0.1  # weaker than direct sound

    ir[reverb_start:] += reverb

    # Normalize to unit energy
    energy = np.sum(ir ** 2)
    if energy > 0:
        ir /= np.sqrt(energy)

    return ir


def apply_room_ir(
    audio: np.ndarray,
    ir: np.ndarray = None,
    noise_db: float = DEFAULT_NOISE_DB,
) -> np.ndarray:
    """Convolve audio with a room impulse response and add noise floor.

    Parameters
    ----------
    audio : np.ndarray
        Input float32 PCM audio.
    ir : np.ndarray, optional
        Room impulse response. Generated via `generate_room_ir()` if None.
    noise_db : float
        Noise floor level in dB relative to peak audio level.

    Returns
    -------
    np.ndarray
        Room-simulated audio (length = len(audio) + len(ir) - 1).
    """
    if ir is None:
        ir = generate_room_ir()

    # Convolve
    output = fftconvolve(audio, ir, mode="full").astype(np.float32)

    # Normalize to prevent clipping / extreme silence
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    # Add noise floor
    peak = np.max(np.abs(audio))
    if peak > 0:
        noise_rms = peak * (10 ** (noise_db / 20.0))
        noise = np.random.randn(len(output)).astype(np.float32) * noise_rms
        output += noise

    return output


def estimate_rt60(ir: np.ndarray, fs: int = DEFAULT_FS) -> float:
    """Estimate RT60 from an impulse response using the energy decay curve.

    Parameters
    ----------
    ir : np.ndarray
        Impulse response.
    fs : int
        Sample rate.

    Returns
    -------
    float
        Estimated RT60 in seconds. Returns -1.0 if decay is less than 60 dB.
    """
    # Energy decay curve
    ir_sq = ir ** 2
    edc = np.cumsum(ir_sq[::-1])[::-1]

    if edc[0] == 0:
        return -1.0

    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-12)

    # Find where edc drops by 60 dB
    target = -60.0
    samples = np.where(edc_db <= target)[0]
    if len(samples) == 0:
        # Use extrapolation from -5 dB point (standard T20→T60 method)
        start_idx = np.where(edc_db <= -5)[0]
        if len(start_idx) == 0:
            return -1.0
        decay_rate = abs(edc_db[start_idx[0]] + 5) / (start_idx[0] / fs)
        if decay_rate > 0:
            return 60.0 / decay_rate
        return -1.0

    return float(samples[0]) / fs
