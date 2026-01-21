#!/usr/bin/env python3
"""
STEP 2: Record TTS + You speaking (improved)

This will record 10 seconds total:
- First 5 seconds: TTS playing (you can stay silent or speak)
- Next 5 seconds: NO TTS, just you speaking alone

This gives us both echo + clean speech in one recording.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import time
import threading

def main():
    os.makedirs("test_recordings", exist_ok=True)

    # Load TTS (should exist from step 1)
    if not os.path.exists("test_recordings/tts_reference.mp3"):
        print("Error: Run record_step1_tts_only.py first!")
        return

    tts_data, tts_sr = sf.read("test_recordings/tts_reference.mp3")

    # Trim TTS to 5 seconds
    tts_5sec_samples = int(5.0 * tts_sr)
    tts_5sec = tts_data[:tts_5sec_samples]

    print("\n" + "="*60)
    print("RECORDING MIC (TWO PARTS)")
    print("="*60)
    print("This will record for 10 seconds total:")
    print()
    print("  📢 Seconds 0-5: TTS will play (you can speak over it)")
    print("  🎤 Seconds 5-10: NO TTS, just YOU speaking")
    print()
    print("This gives us both echo and clean speech!")
    print("\nStarting in 3 seconds...")

    time.sleep(3)

    # Start recording first (10 seconds total)
    record_duration = 10.0
    sample_rate = 16000

    print(f"🔴 Recording for {record_duration:.0f} seconds...")
    print()

    # Start playback in background (only for first 5 seconds)
    def play_tts_delayed():
        time.sleep(0.3)  # Small delay
        print("  [0-5s] 📢 TTS PLAYING - you can speak over it or stay silent")
        sd.play(tts_5sec, tts_sr)
        # TTS will stop after ~5 seconds
        time.sleep(5.5)
        print("  [5-10s] 🎤 TTS STOPPED - now YOU speak alone!")

    play_thread = threading.Thread(target=play_tts_delayed)
    play_thread.start()

    # Record full 10 seconds
    recording = sd.rec(
        int(record_duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    play_thread.join()

    # Save
    sf.write("test_recordings/mic_tts_human.wav", recording, sample_rate)

    print("\n✅ Recording saved to: test_recordings/mic_tts_human.wav")
    print("\nRecording breakdown:")
    print("  First 5s: TTS echo + (optional) your voice")
    print("  Last 5s: Just your voice (no echo)")
    print("\nNext step:")
    print("  python3 analyze_recordings.py")

if __name__ == "__main__":
    main()
