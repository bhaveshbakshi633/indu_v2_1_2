#!/usr/bin/env python3
"""
Test microphone setup with keyboard control
Press Enter to start/stop recording
"""

import sounddevice as sd
import numpy as np
import soundfile as sf
import sys
import threading

print("="*60)
print("AUDIO DEVICES")
print("="*60)
print(sd.query_devices())

print("\n" + "="*60)
print("DEFAULT DEVICES")
print("="*60)
print(f"Default input: {sd.default.device[0]}")
print(f"Default output: {sd.default.device[1]}")

# Get default input device info
try:
    default_input = sd.query_devices(sd.default.device[0], 'input')
    print(f"\nDefault input device details:")
    print(f"  Name: {default_input['name']}")
    print(f"  Max input channels: {default_input['max_input_channels']}")
    print(f"  Default sample rate: {default_input['default_samplerate']}")
except Exception as e:
    print(f"Error querying default input: {e}")

print("\n" + "="*60)
print("KEYBOARD-CONTROLLED RECORDING")
print("="*60)

sample_rate = 16000
recording_chunks = []
is_recording = False
stream = None

def audio_callback(indata, frames, time, status):
    """Called for each audio block while recording"""
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        recording_chunks.append(indata.copy())

print("\n🔴 RECORDING NOW... (Press Enter to STOP)")
is_recording = True

# Start input stream
stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    dtype='int16',
    callback=audio_callback
)

stream.start()

# Wait for Enter to stop
input()

is_recording = False
stream.stop()
stream.close()

print("✅ Recording stopped!")

# Combine all chunks
if recording_chunks:
    recording = np.concatenate(recording_chunks, axis=0)

    # Check if we got any audio
    recording_float = recording.astype(np.float32)
    rms = np.sqrt(np.mean(recording_float ** 2))
    max_val = np.abs(recording_float).max()
    duration = len(recording) / sample_rate

    print(f"\nRecording stats:")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(recording)}")
    print(f"  Data type: {recording.dtype}")
    print(f"  Range: [{recording.min()}, {recording.max()}]")
    print(f"  RMS Energy: {rms:.1f}")
    print(f"  Max absolute value: {max_val:.1f}")

    if rms < 10.0:
        print("\n⚠️  WARNING: Recording is very quiet or silent!")
        print("Possible issues:")
        print("  1. Wrong microphone selected")
        print("  2. Microphone muted or volume too low")
        print("  3. Permission issue")
        print("  4. Need to specify device explicitly")
    else:
        print("\n✅ Microphone is working!")

    # Save test recording
    import os
    os.makedirs("test_recordings", exist_ok=True)
    sf.write("test_recordings/mic_test.wav", recording, sample_rate)
    print(f"\nSaved test to: test_recordings/mic_test.wav")
    print("Play it back with: aplay test_recordings/mic_test.wav")
else:
    print("\n❌ No audio was recorded!")
