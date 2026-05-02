import numpy as np
import sys
sys.path.insert(0, 'packages')
from thalamus.attributor import ThalamicAttributor, SourceVerdict

class AlwaysSelf:
    def is_self(self, pcm):
        return True, 0.95

attributor = ThalamicAttributor(
    m4_detector=AlwaysSelf(),
    m4_self_threshold=0.1,
    sr=16000,
    m4_min_buffer_ms=1,   # tiny buffer so it triggers immediately
)

frame = np.ones(16000 // 30, dtype=np.float32)  # 30 ms frame

# Simulate a speech+playback window
res = attributor.attribute(
    match_score=0.8,
    observed_onset_samples=800,
    expected_onset_samples=800,
    rms_db=-30.0,
    is_playing=True,
    is_speech=True,
    frame_audio=frame,
)

print('verdict:', res.verdict.name)
print('metadata:', res.metadata)
print('confidence:', res.confidence)

# If we get here and verdict is SELF with m4 metadata, plumbing works.
assert res.verdict == SourceVerdict.SELF, 'M4 override did not promote to SELF'
assert res.metadata is not None
assert res.metadata.get('m4_attempted') is True
assert res.metadata.get('m4_confidence') == 0.95
print('Plumbing test PASSED')