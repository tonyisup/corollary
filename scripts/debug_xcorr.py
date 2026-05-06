#!/usr/bin/env python3
"""Quick debug: check xcorr scores for recorded scenarios."""
import numpy as np, json, sys
from pathlib import Path

sys.path.insert(0, 'packages')
from cochlea.capture import SR
from cochlea.vad import VAD_FRAME_SAMPLES

# Import xcorr functions directly from the harness module file
import importlib.util
spec = importlib.util.spec_from_file_location("m2_harness", "scripts/m2_speech_preservation_harness.py")
mod = importlib.util.module_from_spec(spec)

# We need to mock sounddevice to avoid the import error
import types
mock_sd = types.ModuleType('sounddevice')
mock_sd.query_devices = lambda: []
mock_sd.play = lambda *a, **k: None
mock_sd.wait = lambda: None
mock_sd.InputStream = lambda *a, **k: None
sys.modules['sounddevice'] = mock_sd

spec.loader.exec_module(mod)

normalized_xcorr = mod.normalized_xcorr
local_xcorr = mod.local_xcorr

v2 = Path('data/m3_interrupt_v2')

for name in ['human_early', 'human_before', 'self_only', 'world_only']:
    sd = v2 / name
    meta = json.load(open(sd / 'metadata.json'))
    captured = np.load(str(sd / 'captured.npy'))
    agent_pcm_path = sd / 'agent_pcm.npy'
    agent_pcm = np.load(str(agent_pcm_path)) if agent_pcm_path.exists() else None
    input_sr = meta['input_sr']

    if input_sr != SR:
        import librosa
        observed = librosa.resample(captured, orig_sr=input_sr, target_sr=SR).astype(np.float32)
    else:
        observed = captured

    if agent_pcm is not None and len(agent_pcm) > 0:
        dry_global_score, xcorr_shift = normalized_xcorr(observed, agent_pcm)
        self_expected_shift = int((meta['agent_start_s'] + 0.3) * SR)
        dry_local_score, dry_local_shift = local_xcorr(observed, agent_pcm, self_expected_shift, search_ms=256.4)
        print(f'{name}: global={dry_global_score:.4f} local={dry_local_score:.4f} shift={dry_local_shift} ({dry_local_shift/SR*1000:.1f}ms)')
    else:
        print(f'{name}: no agent PCM')
