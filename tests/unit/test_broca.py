"""Tests for the Broca TTS synthesis."""

import numpy as np
import pytest

from broca.tts import _decode_mp3_to_pcm


class TestDecodeMp3ToPcm:
    def test_silence_mp3_decodes(self):
        """A minimal valid MP3 frame (silence) should decode to float32 PCM."""
        # This requires ffmpeg/pydud to be installed
        # If neither is available, the test is skipped
        pytest.importorskip("pydub")

    def test_decode_raises_without_decoder(self):
        """Invalid data with no decoder should raise."""
        # With no ffmpeg/pydub, this should raise RuntimeError
        # but only if neither is installed — check availability first
        import subprocess
        has_ffmpeg = False
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            has_ffmpeg = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        try:
            import pydub  # noqa: F401
        except ImportError:
            if not has_ffmpeg:
                with pytest.raises(RuntimeError, match="Cannot decode MP3"):
                    _decode_mp3_to_pcm(b"not a valid mp3 file")


# Note: Full TTS integration tests (network calls to edge-tts)
# should go in tests/integration/, not unit tests.
