import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from thalamus.attributor import SourceVerdict
from thalamus.dual_input_fusion import ChannelEvidence, fuse_dual_input_evidence


def test_dual_input_self_only():
    res = fuse_dual_input_evidence([
        ChannelEvidence(self_score=0.40, world_score=0.01, self_evidence_db=28.0, world_evidence_db=-60.0),
        ChannelEvidence(self_score=0.02, world_score=0.01, self_evidence_db=1.0, world_evidence_db=-70.0),
    ])
    assert res.verdict == SourceVerdict.SELF
    assert res.self_present is True
    assert res.world_present is False


def test_dual_input_world_only_from_world_favored_mic():
    res = fuse_dual_input_evidence([
        ChannelEvidence(self_score=0.01, world_score=0.30, self_evidence_db=-60.0, world_evidence_db=1.0),
        ChannelEvidence(self_score=0.01, world_score=0.78, self_evidence_db=-70.0, world_evidence_db=4.0),
    ])
    assert res.verdict == SourceVerdict.WORLD
    assert res.world_present is True


def test_dual_input_world_detected_by_template_match_even_with_noisy_baseline():
    res = fuse_dual_input_evidence([
        ChannelEvidence(self_score=0.45, world_score=0.05, self_evidence_db=25.0, world_evidence_db=0.1),
        ChannelEvidence(self_score=0.02, world_score=0.80, self_evidence_db=6.0, world_evidence_db=0.0),
    ])
    assert res.verdict == SourceVerdict.BOTH
    assert res.world_present is True
    assert res.world_favored_score == 0.80


def test_dual_input_replay_control_forces_world_not_self():
    res = fuse_dual_input_evidence([
        ChannelEvidence(self_score=0.35, world_score=0.35, self_evidence_db=20.0, world_evidence_db=2.0),
        ChannelEvidence(self_score=0.60, world_score=0.60, self_evidence_db=5.0, world_evidence_db=3.0),
    ], mode="replay_control")
    assert res.verdict == SourceVerdict.WORLD
    assert res.self_present is False


def test_dual_input_silence_when_no_evidence():
    res = fuse_dual_input_evidence([
        ChannelEvidence(self_score=0.01, world_score=0.01, self_evidence_db=0.0, world_evidence_db=0.0),
        ChannelEvidence(self_score=0.01, world_score=0.01, self_evidence_db=0.0, world_evidence_db=0.0),
    ])
    assert res.verdict == SourceVerdict.SILENCE
