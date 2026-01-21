#!/usr/bin/env python3
"""
Simplest possible microphone test
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os

print("Available audio devices:")
print(sd.query_devices())
print()

duration = 5  # Record 5 seconds
sample_rate = 16000

print(f"Recording {duration} seconds from default microphone...")
print("SPEAK NOW!")
print()

# Record
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='int16',
    blocking=True  # Wait until done
)

print("Done recording!")

# Stats
audio_float = audio.flatten().astype(np.float32)
rms = np.sqrt(np.mean(audio_float ** 2))
max_val = np.abs(audio_float).max()

print(f"\nRecording stats:")
print(f"  Shape: {audio.shape}")
print(f"  Min/Max: [{audio.min()}, {audio.max()}]")
print(f"  RMS: {rms:.1f}")
print(f"  Max absolute: {max_val:.1f}")

if max_val < 10:
    print("\n❌ PROBLEM: Recording is silent or nearly silent!")
    print("\nTroubleshooting:")
    print("1. Check if microphone is muted")
    print("2. Check system audio settings")
    print("3. Try: pacmd list-sources  (to see available inputs)")
else:
    print("\n✅ Microphone is working!")

# Save
os.makedirs("test_recordings", exist_ok=True)
sf.write("test_recordings/simple_test.wav", audio, sample_rate)
print(f"\nSaved to: test_recordings/simple_test.wav")
print("Play it: aplay test_recordings/simple_test.wav")
