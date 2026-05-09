"""Tests for STG speaker identification model loading."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from stg.speaker import SpeakerIdentifier


def test_real_speaker_identifier_expands_speechbrain_cache_path(monkeypatch, tmp_path):
    """SpeechBrain savedir should use an expanded absolute cache path, not repo-local '~/'."""
    captured = {}

    class FakeSpeakerRecognition:
        @classmethod
        def from_hparams(cls, *, source, savedir):
            captured["source"] = source
            captured["savedir"] = savedir
            return "fake-model"

    speechbrain_mod = types.ModuleType("speechbrain")
    inference_mod = types.ModuleType("speechbrain.inference")
    speaker_mod = types.ModuleType("speechbrain.inference.speaker")
    speaker_mod.SpeakerRecognition = FakeSpeakerRecognition

    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_mod)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_mod)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", speaker_mod)
    monkeypatch.setenv("HOME", str(tmp_path))

    identifier = SpeakerIdentifier(use_real=True)
    identifier._load_model()

    expected = tmp_path / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
    savedir = captured["savedir"]
    assert savedir == str(expected)
    assert Path(savedir).is_absolute()
    assert not savedir.startswith("~")
