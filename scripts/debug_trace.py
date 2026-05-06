#!/usr/bin/env python3
"""Quick debug: trace human_early scenario frame by frame."""
import numpy as np, json, sys
from pathlib import Path

sys.path.insert(0, 'packages')
sys.path.insert(0, 'scripts')

from cochlea.capture import SR
from cochlea.vad import VAD_FRAME_SAMPLES, SmoothedVAD
from thalamus.attributor import ThalamicAttributor, SourceVerdict, AttributionResult
from pfc.agentive_controller import AgentiveController

# Import xcorr functions
import importlib.util
spec = importlib.util.spec_from_file_location("m2_harness", "scripts/m2_speech_preservation_harness.py")
mod = importlib.util.module_from_spec(spec)
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

sd = Path('data/m3_interrupt_v2/human_early')
meta = json.load(open(sd / 'metadata.json'))
captured = np.load(str(sd / 'captured.npy'))
agent_pcm = np.load(str(sd / 'agent_pcm.npy'))
input_sr = meta['input_sr']

if input_sr != SR:
    import librosa
    observed = librosa.resample(captured, orig_sr=input_sr, target_sr=SR).astype(np.float32)
else:
    observed = captured

dry_global_score, xcorr_shift = normalized_xcorr(observed, agent_pcm)
self_expected_shift = int((meta['agent_start_s'] + 0.3) * SR)
self_score, self_shift = local_xcorr(observed, agent_pcm, self_expected_shift, search_ms=256.4)

gt_human = np.load(str(sd / 'gt_human_speaking.npy'))
gt_agent = np.load(str(sd / 'gt_agent_playing.npy'))
if input_sr != SR:
    gt_human = np.interp(np.linspace(0, len(gt_human), len(observed)), np.arange(len(gt_human)), gt_human.astype(float)) > 0.5
    gt_agent = np.interp(np.linspace(0, len(gt_agent), len(observed)), np.arange(len(gt_agent)), gt_agent.astype(float)) > 0.5

attributor = ThalamicAttributor(sr=SR, similarity_threshold=0.05, causal_tolerance_ms=175.0,
                                 m4_detector=None, m4_self_threshold=0.50, m4_min_buffer_ms=60.0)
controller = AgentiveController(listen_threshold_ms=100.0, cooldown_ms=200.0, motor_holdoff_ms=200.0)
vad = SmoothedVAD()

controller.request_output(meta['agent_text'], current_time=0.5)

print(f'{"frame":>5} {"t":>6} {"vad":>5} {"verdict":>8} {"state":>8} {"cs":>4} {"mi":>4} {"motor":>6}')
for i in range(13, 25):
    t_s = i * 0.03
    idx = i * VAD_FRAME_SAMPLES
    frame = observed[idx:idx+VAD_FRAME_SAMPLES]
    rms_db = 20 * np.log10(max(np.sqrt(np.mean(frame**2)), 1e-10))
    vad_speech = vad.process_frame(frame)
    is_human = bool(gt_human[idx]) if idx < len(gt_human) else False
    is_agent = bool(gt_agent[idx]) if idx < len(gt_agent) else False
    
    if is_agent and is_human:
        verdict = SourceVerdict.BOTH
        match_score = self_score
    elif is_agent:
        verdict = SourceVerdict.SELF
        match_score = self_score
    elif is_human:
        verdict = SourceVerdict.WORLD
        match_score = 0.0
    else:
        verdict = SourceVerdict.SILENCE
        match_score = 0.0
    
    attr = AttributionResult(
        verdict=verdict, confidence=0.8, self_match_score=match_score,
        is_in_motor_window=is_agent, causal_score=0.5 if verdict == SourceVerdict.SELF else 0.0,
        energy_db=rms_db, is_speech=vad_speech,
    )
    
    result = attributor.attribute(match_score=match_score, observed_onset_samples=self_shift if is_agent else idx,
                                   expected_onset_samples=self_shift if is_agent else None,
                                   rms_db=rms_db, is_playing=is_agent, is_speech=vad_speech, frame_audio=frame)
    decision = controller.process_attribution(result, current_time=t_s)
    
    marker = " <-- FALSE_DEFER" if (not decision.can_speak and is_agent and not is_human) else ""
    print(f'{i:5d} {t_s:6.3f} {str(vad_speech):>5} {result.verdict.name:>8} {decision.state.name:>8} '
          f'{str(decision.can_speak):>4} {str(decision.should_interrupt):>4} {str(result.is_in_motor_window):>6}{marker}')
