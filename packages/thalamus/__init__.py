"""Thalamus — Source Attribution and Gating.

The Thalamus handles source attribution (self vs. world) and 
sensory gating (attenuation of self-caused signals).
"""

from .gate import ThalamicGate
from .attributor import ThalamicAttributor, SourceVerdict, AttributionResult
from .dual_input_fusion import ChannelEvidence, DualInputFusionConfig, DualInputFusionResult, fuse_dual_input_evidence
