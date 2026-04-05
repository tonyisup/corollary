import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
from broca.tts import synthesize
import sounddevice as sd

async def test():
    pcm = await synthesize('This is coming from the test speakers.')
    print('Playing through monitor speakers...')
    sd.play(pcm, samplerate=16000, device=3, blocking=True)
    print('Done!')

asyncio.run(test())