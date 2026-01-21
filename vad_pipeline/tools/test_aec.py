#!/usr/bin/env python3
"""
Quick AEC diagnostic tool to check if echo cancellation is working
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def test_aec_subtraction():
    """Test if basic echo subtraction works"""

    # Simulate TTS audio (sine wave at 440 Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    tts_signal = np.sin(2 * np.pi * 440 * t) * 10000  # Amplitude 10000

    # Simulate mic picking up TTS (with some attenuation and noise)
    echo_attenuation = 0.6  # Echo is 60% of original
    mic_signal = tts_signal * echo_attenuation + np.random.normal(0, 100, len(tts_signal))

    # Apply AEC with different suppression factors
    suppression_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("=" * 60)
    print("AEC Subtraction Test")
    print("=" * 60)
    print(f"TTS RMS energy: {np.sqrt(np.mean(tts_signal ** 2)):.1f}")
    print(f"Mic (with echo) RMS energy: {np.sqrt(np.mean(mic_signal ** 2)):.1f}")
    print()

    results = []
    for factor in suppression_factors:
        # Subtract echo
        cleaned_signal = mic_signal - (tts_signal * factor)
        cleaned_energy = np.sqrt(np.mean(cleaned_signal ** 2))

        results.append(cleaned_energy)
        print(f"Suppression {factor:.1f}: Cleaned RMS = {cleaned_energy:.1f}")

    # Find optimal suppression factor
    optimal_idx = np.argmin(results)
    optimal_factor = suppression_factors[optimal_idx]

    print()
    print(f"Optimal suppression factor: {optimal_factor:.1f}")
    print(f"(Should be close to echo attenuation: {echo_attenuation:.1f})")
    print()

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    time_range = slice(0, 1000)  # Show first 1000 samples

    axes[0].plot(t[time_range], tts_signal[time_range])
    axes[0].set_title('TTS Signal (what\'s playing)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(t[time_range], mic_signal[time_range])
    axes[1].set_title('Mic Signal (with echo)')
    axes[1].set_ylabel('Amplitude')

    cleaned = mic_signal - (tts_signal * optimal_factor)
    axes[2].plot(t[time_range], cleaned[time_range])
    axes[2].set_title(f'Cleaned Signal (after AEC, factor={optimal_factor:.1f})')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig('aec_test_result.png', dpi=150)
    print(f"Saved visualization to: aec_test_result.png")
    print()

def test_timing_sync():
    """Test timing synchronization"""

    sample_rate = 16000
    websocket_chunk_size = 1365  # WebSocket sends this many samples

    # Simulate 3 seconds of TTS audio
    tts_duration = 3.0
    tts_samples = int(sample_rate * tts_duration)
    tts_audio = np.random.normal(0, 1000, tts_samples)

    print("=" * 60)
    print("Timing Synchronization Test")
    print("=" * 60)
    print(f"TTS duration: {tts_duration}s ({tts_samples} samples)")
    print(f"WebSocket chunk size: {websocket_chunk_size} samples")
    print(f"Expected chunks: {tts_samples / websocket_chunk_size:.1f}")
    print()

    # Simulate receiving WebSocket chunks
    position = 0
    chunk_count = 0

    while position < len(tts_audio):
        # Get mic chunk (simulated)
        mic_chunk_size = min(websocket_chunk_size, len(tts_audio) - position)

        # Get corresponding playback chunk
        playback_chunk = tts_audio[position:position + mic_chunk_size]

        # Check if we're staying in sync
        if chunk_count % 10 == 0:
            print(f"Chunk {chunk_count}: Position {position}/{len(tts_audio)} "
                  f"({position / sample_rate:.2f}s / {tts_duration:.2f}s)")

        position += mic_chunk_size
        chunk_count += 1

    print()
    print(f"Total chunks processed: {chunk_count}")
    print(f"Final position: {position} (should be {len(tts_audio)})")
    print(f"Timing drift: {abs(position - len(tts_audio))} samples")
    print()

def test_amplitude_matching():
    """Test if amplitude ranges match between TTS and mic"""

    print("=" * 60)
    print("Amplitude Matching Test")
    print("=" * 60)

    # Typical Edge TTS output range (after loading from MP3)
    tts_float32_range = (-1.0, 1.0)

    # Typical mic input (int16 from WebSocket, converted to float32)
    mic_int16_max = 32768
    mic_float32_range = (-mic_int16_max, mic_int16_max)

    print(f"TTS audio range (float32 from file): {tts_float32_range}")
    print(f"Mic audio range (int16→float32): {mic_float32_range}")
    print()

    # Simulate TTS at file range
    tts_signal = np.random.uniform(-1.0, 1.0, 1000)

    # Simulate mic at int16 range
    mic_signal = np.random.uniform(-mic_int16_max, mic_int16_max, 1000)

    print(f"TTS RMS: {np.sqrt(np.mean(tts_signal ** 2)):.3f}")
    print(f"Mic RMS: {np.sqrt(np.mean(mic_signal ** 2)):.1f}")
    print()

    # Need to scale TTS to match mic range
    tts_scaled = tts_signal * mic_int16_max

    print(f"TTS scaled RMS: {np.sqrt(np.mean(tts_scaled ** 2)):.1f}")
    print()
    print("⚠️  IMPORTANT: TTS audio must be scaled to match mic amplitude!")
    print(f"   Multiply TTS by {mic_int16_max} before subtracting")
    print()

if __name__ == "__main__":
    print("\n")
    test_aec_subtraction()
    test_timing_sync()
    test_amplitude_matching()
    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)
