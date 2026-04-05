"""Cerebellum — Multi-Band NLMS Forward Model.

The biological cochlea doesn't process audio as a single broadband signal.
Stereocilia along the basilar membrane decompose sound into ~3500 parallel
frequency channels, each tuned to a narrow band. Hair cells at the base
respond to high frequencies; those at the apex respond to low frequencies.

This model mirrors that architecture: instead of learning one room IR for
the entire audio spectrum, it learns separate IRs for multiple frequency
bands. Each band's IR captures how the room responds to that frequency range.

This matters because:
- Low frequencies (vowels, ~300-800 Hz) travel differently through rooms
- High frequencies (consonants, ~3000-8000 Hz) are more directional
- Speaker/mic response varies dramatically across the spectrum
- A single broadband IR can't capture all of this

Usage:
    cerebellum = MultiBandBabblingCerebellum(
        band_edges=[(300, 800), (800, 1500), (1500, 3000), (3000, 8000)],
        ir_length_ms=300,
        nlms_mu=0.25,
    )
    error = cerebellum.update(known_pcm, captured_pcm)
    predicted_mel = cerebellum.predict_mel(efference_pcm)
    prediction_pcm = cerebellum.predict_pcm(efference_pcm)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from cochlea.capture import pcm_to_mels, SR
from cerebellum.trust_model import TrustTracker
from typing import List, Tuple, Optional


# Default frequency bands (approximate cochlear tonotopic mapping)
# Based on human hearing: 300-8000 Hz is the speech-relevant range
DEFAULT_BANDS = [
    (300, 800),      # Low: vowels, fundamental
    (800, 1500),     # Mid-low: first/second formants
    (1500, 3000),    # Mid: consonants, plosives
    (3000, 8000),    # High: fricatives, sibilants
]


class MultiBandBabblingCerebellum:
    """Forward model with separate NLMS IRs per frequency band.

    Each band learns its own room impulse response independently,
    then recombines predictions for the full-spectrum output.

    This mirrors how the cochlea's stereocilia process different
    frequency ranges along the basilar membrane.
    """

    def __init__(
        self,
        band_edges: List[Tuple[float, float]] = DEFAULT_BANDS,
        ir_length_ms: int = 300,
        sample_rate: int = SR,
        nlms_mu: float = 0.25,
        nlms_epsilon: float = 1e-6,
        filter_order: int = 4,
    ) -> None:
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2.0
        self.ir_length = int(sample_rate * ir_length_ms / 1000)
        self.nlms_mu = nlms_mu
        self.nlms_epsilon = nlms_epsilon
        self.band_edges = [(lo, hi) for lo, hi in band_edges if lo < self.nyquist and hi > 0]
        self.filter_order = filter_order

        # Build bandpass filters for each band
        self._sos_filters: List[np.ndarray] = []
        for lo, hi in self.band_edges:
            lo_norm = max(lo / self.nyquist, 0.01)
            hi_norm = min(hi / self.nyquist, 0.99)
            if lo_norm >= hi_norm:
                # Band is outside valid range — use allpass
                self._sos_filters.append(None)
            else:
                sos = butter(filter_order, [lo_norm, hi_norm], btype="band", output="sos")
                self._sos_filters.append(sos)

        # One IR per band, initialized as delta (dry signal)
        self.band_irs: List[np.ndarray] = []
        for _ in self.band_edges:
            ir = np.zeros(self.ir_length, dtype=np.float64)
            ir[0] = 1.0
            self.band_irs.append(ir)

        # Trust tracker
        self.trust = TrustTracker()
        self.total_updates: int = 0
        self.prediction_errors: List[float] = []

    def _filter_band(self, pcm: np.ndarray, band_idx: int) -> np.ndarray:
        """Filter PCM to a single frequency band."""
        sos = self._sos_filters[band_idx]
        if sos is None:
            return pcm.copy().astype(np.float64)
        return sosfilt(sos, pcm.astype(np.float64))

    def _filter_all_bands(self, pcm: np.ndarray) -> List[np.ndarray]:
        """Split PCM into all frequency bands."""
        return [self._filter_band(pcm, i) for i in range(len(self.band_edges))]

    def _nlms_update_band(
        self,
        known_band: np.ndarray,
        captured_band: np.ndarray,
        ir: np.ndarray,
    ) -> np.ndarray:
        """NLMS update for a single frequency band.

        Normalizes both signals to unit energy so bands with different
        power levels (bass vs treble) converge at similar rates.
        """
        N = len(ir)
        mu = self.nlms_mu
        eps = self.nlms_epsilon

        max_len = min(len(known_band), len(captured_band))
        known = known_band[:max_len]
        captured = captured_band[:max_len]

        # Per-band normalization
        k_rms = float(np.sqrt(np.mean(known ** 2)) + 1e-10)
        c_rms = float(np.sqrt(np.mean(captured ** 2)) + 1e-10)
        known_norm = known / k_rms
        captured_norm = captured / c_rms

        # Pad for look-back
        pad = np.zeros(N - 1, dtype=np.float64)
        ext = np.concatenate([pad, known_norm, pad])

        # Normalize the existing IR too
        ir_rms = float(np.sqrt(np.mean(ir ** 2)) + 1e-10)
        ir_norm = ir / ir_rms * (c_rms / k_rms)

        # Prediction
        pred = fftconvolve(known_norm, ir_norm, mode="full")[:max_len]
        error = captured_norm - pred

        # NLMS sample-by-sample (normalized)
        for t in range(max_len):
            e = error[t]
            if abs(e) < 1e-15:
                continue
            x = ext[t:t + N][::-1]
            x_power = float(np.dot(x, x))
            if x_power > eps:
                ir_norm += mu * e * x / (x_power + eps)

        # Convert back to unnormalized IR
        ir[:] = ir_norm * c_rms

        return error * c_rms

    def update(self, known_pcm: np.ndarray, captured_pcm: np.ndarray) -> float:
        """Update all band IRs from one babbling pair.

        Parameters
        ----------
        known_pcm : np.ndarray
            The known input signal (TTS output), float32.
        captured_pcm : np.ndarray
            What the microphone captured, float32.

        Returns
        -------
        float
            RMS prediction error across all bands.
        """
        known = known_pcm.astype(np.float64)
        captured = captured_pcm.astype(np.float64)

        # Filter into bands
        known_bands = self._filter_all_bands(known)
        captured_bands = self._filter_all_bands(captured)

        band_errors = []
        for i in range(len(self.band_edges)):
            # Normalize each band to unit energy before NLMS
            # This prevents low-freq bands (high energy) from dominating
            k_rms = float(np.sqrt(np.mean(known_bands[i] ** 2)))
            c_rms = float(np.sqrt(np.mean(captured_bands[i] ** 2)))
            amplitude_scale = c_rms / max(k_rms, 1e-10)

            if k_rms > 1e-10:
                known_norm = known_bands[i] / k_rms
            else:
                known_norm = known_bands[i]
            if c_rms > 1e-10:
                captured_norm = captured_bands[i] / c_rms
            else:
                captured_norm = captured_bands[i]

            # Scale the IR by the band's amplitude ratio
            # This way the IR captures both the room shape AND the band's gain
            self.band_irs[i] *= amplitude_scale

            err = self._nlms_update_band(known_norm, captured_norm, self.band_irs[i])
            band_errors.append(float(np.sqrt(np.mean(err ** 2))))

        # Overall error (RMS across bands)
        overall_error = float(np.sqrt(np.mean(np.array(band_errors) ** 2)))
        self.total_updates += 1
        self.prediction_errors.append(overall_error)

        # Update trust
        error_db = 20 * np.log10(max(overall_error, 1e-10))
        if error_db < -30:
            self.trust.confirm(error_db)
        elif error_db > -10:
            self.trust.fail()
        else:
            self.trust.confirm(error_db)

        return overall_error

    def predict_pcm(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the raw PCM the cochlea will receive, recombining all bands."""
        bands = self._filter_all_bands(efference_pcm.astype(np.float64))

        band_outputs = []
        total_energy = 0.0
        for i in range(len(self.band_edges)):
            band_pred = fftconvolve(bands[i], self.band_irs[i], mode="full")
            energy = np.sum(band_pred ** 2)
            total_energy += energy
            band_outputs.append(band_pred)

        # Combine all bands
        max_len = max(len(b) for b in band_outputs)
        output = np.zeros(max_len, dtype=np.float64)
        for b in band_outputs:
            min_len = min(len(b), max_len)
            output[:min_len] += b[:min_len]

        # Scale to match the energy of the input (prevents blowup)
        input_energy = np.sum(efference_pcm.astype(np.float64) ** 2)
        output_energy = np.sum(output ** 2)
        if total_energy > 1e-10 and output_energy > 1e-10:
            scale = np.sqrt(input_energy / output_energy)
            output *= min(scale, 2.0)  # Cap scaling to prevent instability

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.9

        return output.astype(np.float32)

    def predict_mel(self, efference_pcm: np.ndarray) -> np.ndarray:
        """Predict the mel spectrogram the cochlea should produce."""
        pred_pcm = self.predict_pcm(efference_pcm)
        return pcm_to_mels(pred_pcm)

    def get_band_irs(self) -> List[np.ndarray]:
        """Return copies of all band IRs."""
        return [ir.copy() for ir in self.band_irs]

    def get_composite_ir(self) -> np.ndarray:
        """Return the composite broadband IR (sum of all band IRs)."""
        composite = np.zeros(self.ir_length, dtype=np.float64)
        for ir in self.band_irs:
            min_len = min(len(composite), len(ir))
            composite[:min_len] += ir[:min_len]
        return composite

    def trust_status(self) -> dict:
        return self.trust.get_status()

    def get_convergence(self) -> dict:
        if not self.prediction_errors:
            return {"updates": 0, "current_error_db": None, "trend": None}
        errors = self.prediction_errors
        current = errors[-1]
        trend = "improving" if len(errors) >= 2 and errors[-1] < errors[-2] else "stable"
        return {
            "updates": self.total_updates,
            "current_error": current,
            "current_error_db": 20 * np.log10(max(current, 1e-10)),
            "mean_error": float(np.mean(errors)),
            "min_error": float(np.min(errors)),
            "trend": trend,
        }
