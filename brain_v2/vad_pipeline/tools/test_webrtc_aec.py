#!/usr/bin/env python3
"""
Test WebRTC AEC implementation
"""

import numpy as np
from webrtc_audio_processing import AudioProcessingModule as AP

print("="*60)
print("WebRTC AEC Test")
print("="*60)

# Initialize WebRTC
print("\n1. Initializing WebRTC AudioProcessingModule...")
aec = AP(aec_type=1, enable_ns=True, agc_type=0, enable_vad=False)
aec.set_stream_format(16000, 1)  # 16kHz, mono
aec.set_aec_level(2)  # High suppression
aec.set_ns_level(2)   # Noise suppression
print("✅ Initialized (AEC level: High, NS level: 2)")

# Simulate TTS audio (440 Hz sine wave)
print("\n2. Generating test signals...")
sample_rate = 16000
duration = 0.5  # 0.5 seconds
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

# TTS signal (reference) - loud sine wave
tts_signal = np.sin(2 * np.pi * 440 * t) * 10000
tts_signal = tts_signal.astype(np.int16)

# Mic signal (with echo + some noise)
mic_signal = (tts_signal * 0.7 + np.random.normal(0, 100, len(tts_signal))).astype(np.int16)

print(f"  TTS signal: {len(tts_signal)} samples")
print(f"  Mic signal: {len(mic_signal)} samples")

# Process with WebRTC AEC
print("\n3. Processing with WebRTC AEC (10ms frames)...")
frame_size = 160  # 10ms at 16kHz
cleaned_frames = []

for i in range(0, len(mic_signal), frame_size):
    # Get frames
    ref_frame = tts_signal[i:i+frame_size]
    mic_frame = mic_signal[i:i+frame_size]

    if len(ref_frame) < frame_size:
        break  # Skip incomplete frames

    # Feed reference (what's playing) - convert to bytes
    aec.process_reverse_stream(ref_frame.tobytes())

    # Process mic (returns cleaned) - convert to bytes
    cleaned_bytes = aec.process_stream(mic_frame.tobytes())

    # Convert back to numpy array
    cleaned_frame = np.frombuffer(cleaned_bytes, dtype=np.int16)
    cleaned_frames.append(cleaned_frame)

cleaned_signal = np.concatenate(cleaned_frames)

print(f"  Processed {len(cleaned_frames)} frames")

# Calculate results
print("\n4. Results:")
tts_rms = np.sqrt(np.mean(tts_signal.astype(np.float32) ** 2))
mic_rms = np.sqrt(np.mean(mic_signal.astype(np.float32) ** 2))
cleaned_rms = np.sqrt(np.mean(cleaned_signal.astype(np.float32) ** 2))

print(f"  Original TTS RMS: {tts_rms:.1f}")
print(f"  Mic (with echo) RMS: {mic_rms:.1f}")
print(f"  Cleaned (after AEC) RMS: {cleaned_rms:.1f}")

echo_reduction = mic_rms - cleaned_rms
reduction_percent = (echo_reduction / mic_rms) * 100 if mic_rms > 0 else 0

print(f"\n  Echo reduction: {echo_reduction:.1f} ({reduction_percent:.1f}%)")

if cleaned_rms < mic_rms * 0.5:
    print("\n✅ WebRTC AEC is working! Echo significantly reduced.")
else:
    print("\n⚠️  WebRTC AEC may need time to adapt (normal for first few frames)")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
