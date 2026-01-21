#!/usr/bin/env python3
"""
STEP 1: Record TTS only (stay SILENT)

This will:
1. Generate TTS audio
2. Play it through speakers
3. Record what the microphone picks up
4. Save to test_recordings/mic_tts_only.wav

⚠️  STAY SILENT during this recording!
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import edge_tts
import asyncio
import os
import time
import threading

async def main():
    os.makedirs("test_recordings", exist_ok=True)

    # Generate TTS (10 second text)
    print("Generating TTS...")
    text = """Hello, this is a test of the acoustic echo cancellation system.
    I am speaking for approximately ten seconds to provide enough audio data for analysis.
    This recording will help us understand the amplitude ranges, timing synchronization,
    and how well the echo cancellation works in practice."""

    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save("test_recordings/tts_reference.mp3")
    print("✅ TTS generated")

    # Load TTS
    tts_data, tts_sr = sf.read("test_recordings/tts_reference.mp3")
    tts_duration = len(tts_data) / tts_sr

    print(f"\nTTS duration: {tts_duration:.1f} seconds")
    print("\n" + "="*60)
    print("RECORDING MIC (TTS ONLY)")
    print("="*60)
    print("⚠️  STAY SILENT!")
    print("Let the mic pick up ONLY the speakers")
    print("\nStarting in 3 seconds...")

    time.sleep(3)

    # Play and record simultaneously
    def play_tts():
        time.sleep(0.3)  # Small delay
        sd.play(tts_data, tts_sr)

    play_thread = threading.Thread(target=play_tts)
    play_thread.start()

    # Record
    record_duration = max(10.0, tts_duration + 0.5)
    print(f"🔴 Recording for {record_duration:.1f} seconds...")

    recording = sd.rec(
        int(record_duration * 16000),
        samplerate=16000,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    play_thread.join()

    # Save
    sf.write("test_recordings/mic_tts_only.wav", recording, 16000)

    print("\n✅ Recording saved to: test_recordings/mic_tts_only.wav")
    print("\nNext step:")
    print("  python3 record_step2_tts_human.py")

if __name__ == "__main__":
    asyncio.run(main())
