import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from broca.tts import synthesize
import sounddevice as sd

import sounddevice as sd
for i, d in enumerate(sd.query_devices()):
    if d['max_input_channels'] > 0 or d['max_output_channels'] > 0:
        print(f'[{i}] {d["name"]} (in:{d["max_input_channels"]} out:{d["max_output_channels"]})')