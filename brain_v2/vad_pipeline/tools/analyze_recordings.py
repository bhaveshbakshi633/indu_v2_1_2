#!/usr/bin/env python3
"""
Analyze test recordings to diagnose AEC issues.

Compares TTS reference with microphone recordings to determine:
- Amplitude scaling needed
- Optimal suppression factor
- Echo cancellation effectiveness
"""

import numpy as np
import soundfile as sf
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

def analyze_recordings(tts_file, mic_tts_only_file, mic_tts_human_file=None):
    """Analyze recordings and determine AEC parameters"""

    print("="*60)
    print("LOADING AUDIO FILES")
    print("="*60)

    # Load TTS reference
    print(f"\nLoading TTS reference: {tts_file}")
    tts_ref, tts_sr = sf.read(tts_file)

    # Resample to 16kHz if needed
    if tts_sr != 16000:
        print(f"  Resampling from {tts_sr}Hz to 16000Hz...")
        tts_ref = scipy.signal.resample(tts_ref, int(len(tts_ref) * 16000 / tts_sr))

    # Convert stereo to mono if needed
    if len(tts_ref.shape) > 1:
        print(f"  Converting stereo to mono...")
        tts_ref = np.mean(tts_ref, axis=1)

    tts_ref = tts_ref.astype(np.float32)

    # Load mic recording (TTS only)
    print(f"\nLoading mic recording (TTS only): {mic_tts_only_file}")
    mic_tts_only, mic_sr = sf.read(mic_tts_only_file)
    if len(mic_tts_only.shape) > 1:
        mic_tts_only = mic_tts_only[:, 0]
    mic_tts_only = mic_tts_only.astype(np.float32)

    # Load mic recording (TTS + human) if provided
    if mic_tts_human_file and os.path.exists(mic_tts_human_file):
        print(f"\nLoading mic recording (TTS + human): {mic_tts_human_file}")
        mic_tts_human, _ = sf.read(mic_tts_human_file)
        if len(mic_tts_human.shape) > 1:
            mic_tts_human = mic_tts_human[:, 0]
        mic_tts_human = mic_tts_human.astype(np.float32)
    else:
        mic_tts_human = None

    # ============================================
    # ANALYSIS
    # ============================================
    print("\n" + "="*60)
    print("AMPLITUDE ANALYSIS")
    print("="*60)

    print(f"\nTTS Reference:")
    print(f"  Samples: {len(tts_ref)}")
    print(f"  Duration: {len(tts_ref)/16000:.2f}s")
    print(f"  Data type: {tts_ref.dtype}")
    print(f"  Range: [{tts_ref.min():.3f}, {tts_ref.max():.3f}]")
    print(f"  RMS Energy: {np.sqrt(np.mean(tts_ref ** 2)):.3f}")

    print(f"\nMic Recording (TTS only):")
    print(f"  Samples: {len(mic_tts_only)}")
    print(f"  Duration: {len(mic_tts_only)/16000:.2f}s")
    print(f"  Data type: {mic_tts_only.dtype}")
    print(f"  Range: [{mic_tts_only.min():.1f}, {mic_tts_only.max():.1f}]")
    print(f"  RMS Energy: {np.sqrt(np.mean(mic_tts_only ** 2)):.1f}")

    if mic_tts_human is not None:
        print(f"\nMic Recording (TTS + Human):")
        print(f"  Samples: {len(mic_tts_human)}")
        print(f"  Duration: {len(mic_tts_human)/16000:.2f}s")
        print(f"  Data type: {mic_tts_human.dtype}")
        print(f"  Range: [{mic_tts_human.min():.1f}, {mic_tts_human.max():.1f}]")
        print(f"  RMS Energy: {np.sqrt(np.mean(mic_tts_human ** 2)):.1f}")

    # Calculate scaling factor
    tts_rms = np.sqrt(np.mean(tts_ref ** 2))
    mic_tts_rms = np.sqrt(np.mean(mic_tts_only ** 2))

    if tts_rms > 0:
        scaling_factor = mic_tts_rms / tts_rms
        print(f"\n" + "="*60)
        print("AMPLITUDE SCALING FACTOR")
        print("="*60)
        print(f"TTS RMS: {tts_rms:.3f}")
        print(f"Mic RMS (echo): {mic_tts_rms:.1f}")
        print(f"\n⚠️  TTS needs to be scaled by: {scaling_factor:.1f}x")
        print(f"   (Multiply TTS by {scaling_factor:.1f} before subtraction)")

    # ============================================
    # TEST ECHO SUBTRACTION
    # ============================================
    print("\n" + "="*60)
    print("TESTING ECHO SUBTRACTION")
    print("="*60)

    # Align lengths (use shorter)
    test_len = min(len(tts_ref), len(mic_tts_only))
    tts_test = tts_ref[:test_len]
    mic_test = mic_tts_only[:test_len]

    print(f"\nOriginal mic RMS (with echo): {np.sqrt(np.mean(mic_test ** 2)):.1f}")

    suppression_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print(f"\nTesting suppression factors (with {scaling_factor:.1f}x scaling):")
    for factor in suppression_factors:
        # Scale TTS to match mic amplitude
        tts_scaled = tts_test * scaling_factor

        # Subtract with suppression factor
        cleaned = mic_test - (tts_scaled * factor)
        cleaned_rms = np.sqrt(np.mean(cleaned ** 2))

        results.append(cleaned_rms)
        print(f"  Factor {factor:.1f}: Cleaned RMS = {cleaned_rms:.1f}")

    # Find optimal
    optimal_idx = np.argmin(results)
    optimal_factor = suppression_factors[optimal_idx]

    print(f"\n⭐ Optimal suppression factor: {optimal_factor:.1f}")
    print(f"   (Gives lowest residual energy: {results[optimal_idx]:.1f})")

    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Show first 3 seconds
    display_samples = min(16000 * 3, test_len)
    time_axis = np.arange(display_samples) / 16000.0

    # Plot 1: TTS reference
    axes[0].plot(time_axis, tts_test[:display_samples], color='blue', alpha=0.7)
    axes[0].set_title(f'TTS Reference (RMS: {tts_rms:.3f})')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mic recording (with echo)
    axes[1].plot(time_axis, mic_test[:display_samples], color='red', alpha=0.7)
    axes[1].set_title(f'Microphone Recording - TTS Only (RMS: {mic_tts_rms:.1f})')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: TTS scaled to mic amplitude
    tts_scaled = tts_test * scaling_factor
    axes[2].plot(time_axis, tts_scaled[:display_samples], color='green', alpha=0.7)
    axes[2].set_title(f'TTS Scaled {scaling_factor:.1f}x to Match Mic (RMS: {np.sqrt(np.mean(tts_scaled**2)):.1f})')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Cleaned signal (after AEC)
    cleaned_optimal = mic_test - (tts_scaled * optimal_factor)
    axes[3].plot(time_axis, cleaned_optimal[:display_samples], color='purple', alpha=0.7)
    axes[3].set_title(f'After AEC (factor={optimal_factor:.1f}, RMS: {results[optimal_idx]:.1f})')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    output_plot = 'test_recordings/aec_analysis.png'
    plt.savefig(output_plot, dpi=150)
    print(f"✅ Saved plot: {output_plot}")

    # ============================================
    # RECOMMENDATIONS
    # ============================================
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR server.py")
    print("="*60)

    print(f"\n1. In server.py line ~268, change:")
    print(f"   FROM:")
    print(f"     audio_np = audio_np - (playback_chunk * self.echo_suppression_factor)")
    print(f"\n   TO:")
    print(f"     playback_scaled = playback_chunk * {scaling_factor:.1f}")
    print(f"     audio_np = audio_np - (playback_scaled * {optimal_factor:.1f})")

    print(f"\n2. Or use a variable for the scaling factor:")
    print(f"   self.playback_scaling_factor = {scaling_factor:.1f}")
    print(f"   self.echo_suppression_factor = {optimal_factor:.1f}")

    return {
        'scaling_factor': scaling_factor,
        'optimal_suppression': optimal_factor,
        'tts_rms': tts_rms,
        'mic_rms': mic_tts_rms,
        'cleaned_rms': results[optimal_idx]
    }

if __name__ == "__main__":
    import glob

    # Auto-detect files in test_recordings
    tts_files = glob.glob("test_recordings/tts*.mp3") + glob.glob("test_recordings/tts*.wav")
    mic_tts_only_files = glob.glob("test_recordings/*tts_only*.wav")
    mic_tts_human_files = glob.glob("test_recordings/*tts_human*.wav")

    if not tts_files:
        print("Error: No TTS reference file found in test_recordings/")
        print("Expected: test_recordings/tts_*.mp3 or tts_*.wav")
        sys.exit(1)

    if not mic_tts_only_files:
        print("Error: No mic recording (TTS only) found in test_recordings/")
        print("Expected: test_recordings/*tts_only*.wav")
        sys.exit(1)

    tts_file = tts_files[0]
    mic_tts_only = mic_tts_only_files[0]
    mic_tts_human = mic_tts_human_files[0] if mic_tts_human_files else None

    print(f"Found files:")
    print(f"  TTS reference: {tts_file}")
    print(f"  Mic (TTS only): {mic_tts_only}")
    if mic_tts_human:
        print(f"  Mic (TTS+human): {mic_tts_human}")

    analyze_recordings(tts_file, mic_tts_only, mic_tts_human)

    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
