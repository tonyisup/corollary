"""
Dual-output scratch: white noise on one device and a short chirp on another.

PortAudio allows multiple OutputStreams to different devices at once. The
high-level sd.play() API cannot overlap (a second play() raises RuntimeError),
so this uses one OutputStream per device, each on its own thread.
"""

from __future__ import annotations

import argparse
import sys
import threading

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: pip install sounddevice numpy")
    sys.exit(1)


def list_devices() -> None:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 or d["max_output_channels"] > 0:
            print(
                f"[{i}] {d['name']} "
                f"(in:{d['max_input_channels']} out:{d['max_output_channels']})"
            )


def pick_samplerate(device_index: int) -> float:
    info = sd.query_devices(device_index)
    sr = info.get("default_samplerate")
    return float(sr) if sr else 48000.0


def play_array_blocking(data: np.ndarray, device: int, samplerate: float) -> None:
    """Play mono float32 in [-1, 1] on *device* (safe concurrent with other OutputStreams)."""
    if data.ndim == 1:
        data = data[:, np.newaxis]
    with sd.OutputStream(
        device=device,
        channels=data.shape[1],
        samplerate=samplerate,
        dtype="float32",
    ) as stream:
        stream.write(np.asarray(data, dtype=np.float32))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Play white noise and a chirp on two output devices at the same time."
    )
    p.add_argument("--list", action="store_true", help="Print devices and exit.")
    p.add_argument(
        "--noise-device",
        type=int,
        default=3,
        metavar="I",
        help="Output device index for white noise (default: 3).",
    )
    p.add_argument(
        "--chirp-device",
        type=int,
        default=4,
        metavar="I",
        help="Output device index for chirp (default: 4).",
    )
    p.add_argument("--noise-duration", type=float, default=2.0, help="Seconds of noise.")
    p.add_argument("--chirp-duration", type=float, default=0.35, help="Seconds of chirp.")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    for idx, label in ((args.noise_device, "noise"), (args.chirp_device, "chirp")):
        dev = sd.query_devices(idx)
        if int(dev["max_output_channels"]) < 1:
            print(f"ERROR: device {idx} ({label}) has no output channels.")
            sys.exit(1)

    sr_noise = pick_samplerate(args.noise_device)
    sr_chirp = pick_samplerate(args.chirp_device)

    rng = np.random.default_rng(0)
    n_noise = int(sr_noise * args.noise_duration)
    noise = (0.08 * rng.standard_normal(n_noise)).astype(np.float32)

    t = np.linspace(
        0.0,
        args.chirp_duration,
        int(sr_chirp * args.chirp_duration),
        endpoint=False,
    )
    f0, f1 = 600.0, 4000.0
    T = args.chirp_duration
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / T)
    chirp = (0.25 * np.sin(phase)).astype(np.float32)

    print(
        f"White noise ~{args.noise_duration}s on device {args.noise_device} @ {sr_noise:g} Hz"
    )
    print(
        f"Chirp ~{args.chirp_duration}s on device {args.chirp_device} @ {sr_chirp:g} Hz"
    )

    workers = [
        threading.Thread(
            target=play_array_blocking,
            args=(noise, args.noise_device, sr_noise),
            name="noise",
        ),
        threading.Thread(
            target=play_array_blocking,
            args=(chirp, args.chirp_device, sr_chirp),
            name="chirp",
        ),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    print("Done.")


if __name__ == "__main__":
    main()
