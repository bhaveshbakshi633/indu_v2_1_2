#!/usr/bin/env python3
"""
Automatic recording script (no user prompts).
Run this twice manually:
  1st run: Stay silent (TTS only)
  2nd run: Speak during playback (TTS + human)
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import edge_tts
import asyncio
import os
import sys
from datetime import datetime
import time
import threading

SAMPLE_RATE = 16000
OUTPUT_DIR = "test_recordings"

async def generate_tts(text: str, output_file: str):
    """Generate TTS audio file"""
    print(f"Generating TTS...")
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)
    print(f"✅ TTS saved to: {output_file}")

def record_audio(duration_seconds: float, filename: str):
    """Record audio from microphone"""
    print(f"\n🎤 Recording for {duration_seconds:.1f} seconds...")
    print("Starting in: 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("🔴 RECORDING NOW!\n")

    # Record
    recording = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()

    print("\n✅ Recording complete!")

    # Save as WAV
    sf.write(filename, recording, SAMPLE_RATE)
    print(f"Saved to: {filename}")

    return recording

def play_audio(filename: str):
    """Play audio file"""
    data, sr = sf.read(filename)
    print(f"🔊 Playing TTS...")
    sd.play(data, sr)
    sd.wait()

async def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 record_test_samples_auto.py tts_only")
        print("  python3 record_test_samples_auto.py tts_human")
        sys.exit(1)

    test_type = sys.argv[1]

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Test text (approximately 10 seconds)
    test_text = """Hello, this is a test of the acoustic echo cancellation system.
    I am speaking for approximately ten seconds to provide enough audio data for analysis.
    This recording will help us understand the amplitude ranges, timing synchronization,
    and how well the echo cancellation works in practice."""

    # Generate TTS
    tts_file = f"{OUTPUT_DIR}/tts_reference.mp3"

    # Only generate if doesn't exist
    if not os.path.exists(tts_file):
        await generate_tts(test_text, tts_file)
    else:
        print(f"Using existing TTS: {tts_file}")

    # Load TTS to check duration
    tts_data, tts_sr = sf.read(tts_file)
    tts_duration = len(tts_data) / tts_sr
    print(f"TTS duration: {tts_duration:.2f} seconds")

    # Record for 10 seconds (or TTS + 1s)
    record_duration = max(10.0, tts_duration + 1.0)

    if test_type == "tts_only":
        print("\n" + "="*60)
        print("TEST 1: TTS ONLY")
        print("="*60)
        print("⚠️  STAY SILENT during this recording")
        print("Let the microphone pick up ONLY the TTS from speakers")
        print()

        def play_tts_delayed():
            time.sleep(0.5)
            play_audio(tts_file)

        play_thread = threading.Thread(target=play_tts_delayed)
        play_thread.start()

        recording = record_audio(record_duration, f"{OUTPUT_DIR}/mic_tts_only.wav")
        play_thread.join()

    elif test_type == "tts_human":
        print("\n" + "="*60)
        print("TEST 2: TTS + HUMAN")
        print("="*60)
        print("⚠️  SPEAK during this recording while TTS is playing")
        print("Say things like: 'Wait', 'Stop', 'Hello', etc.")
        print()

        def play_tts_delayed():
            time.sleep(0.5)
            play_audio(tts_file)

        play_thread = threading.Thread(target=play_tts_delayed)
        play_thread.start()

        recording = record_audio(record_duration, f"{OUTPUT_DIR}/mic_tts_human.wav")
        play_thread.join()

    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(1)

    print(f"\n✅ Recording saved!")

if __name__ == "__main__":
    asyncio.run(main())
