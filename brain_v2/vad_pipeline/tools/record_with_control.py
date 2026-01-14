#!/usr/bin/env python3
"""
Recording with TTS control

- Recording starts immediately
- Type 't' and Enter to START TTS playback
- Type 's' and Enter to STOP TTS playback
- Type 'q' and Enter to QUIT and save recording
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import sys
import os

# Configuration
SAMPLE_RATE = 16000
TTS_FILE = "test_recordings/tts_reference.mp3"

# Recording state
recording_chunks = []
is_recording = True
tts_stream = None
tts_thread = None

def audio_callback(indata, frames, time, status):
    """Called for each audio block while recording"""
    if status:
        print(status, file=sys.stderr)
    if is_recording:
        recording_chunks.append(indata.copy())

def play_tts_loop():
    """Play TTS in a loop"""
    global tts_stream
    try:
        # Load TTS
        tts_data, tts_sr = sf.read(TTS_FILE)

        # Trim to first 5 seconds
        duration_samples = int(5.0 * tts_sr)
        tts_5sec = tts_data[:duration_samples]

        print("    [TTS] Playing...")

        # Play on loop
        while tts_stream is not None:
            sd.play(tts_5sec, tts_sr)
            sd.wait()

    except Exception as e:
        print(f"    [TTS] Error: {e}")

def start_tts():
    """Start TTS playback"""
    global tts_stream, tts_thread

    if tts_stream is not None:
        print("  [TTS already playing]")
        return

    tts_stream = True  # Just a flag
    tts_thread = threading.Thread(target=play_tts_loop, daemon=True)
    tts_thread.start()
    print("  ✅ TTS started")

def stop_tts():
    """Stop TTS playback"""
    global tts_stream

    if tts_stream is None:
        print("  [TTS not playing]")
        return

    tts_stream = None
    sd.stop()
    print("  ✅ TTS stopped")

def main():
    global is_recording

    # Check if TTS file exists
    if not os.path.exists(TTS_FILE):
        print(f"Error: TTS file not found: {TTS_FILE}")
        print("Run: python3 record_step1_tts_only.py first")
        return

    print("="*60)
    print("RECORDING WITH TTS CONTROL")
    print("="*60)
    print("\nCommands:")
    print("  t + Enter = Start TTS playback")
    print("  s + Enter = Stop TTS playback")
    print("  q + Enter = Quit and save recording")
    print("\n🔴 RECORDING NOW...")
    print()

    # Start recording stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        callback=audio_callback
    )
    stream.start()

    # Command loop
    while True:
        try:
            cmd = input("> ").strip().lower()

            if cmd == 't':
                start_tts()
            elif cmd == 's':
                stop_tts()
            elif cmd == 'q':
                print("\n✅ Stopping recording...")
                break
            else:
                print(f"  Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\n✅ Stopping recording...")
            break

    # Stop everything
    stop_tts()
    is_recording = False
    stream.stop()
    stream.close()

    # Save recording
    if recording_chunks:
        recording = np.concatenate(recording_chunks, axis=0)
        duration = len(recording) / SAMPLE_RATE
        rms = np.sqrt(np.mean(recording.astype(np.float32) ** 2))

        print(f"\nRecording stats:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  RMS Energy: {rms:.1f}")

        os.makedirs("test_recordings", exist_ok=True)
        output_file = "test_recordings/mic_controlled.wav"
        sf.write(output_file, recording, SAMPLE_RATE)
        print(f"\n✅ Saved to: {output_file}")
    else:
        print("\n❌ No audio was recorded!")

if __name__ == "__main__":
    main()
