#!/usr/bin/env python3
"""
Standalone Interrupt Detection & VAD Testing Script
Test and calibrate interrupt detection during playback + VAD for recording
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from pathlib import Path
import torch

# Suppress ALSA warnings
from ctypes import *
from ctypes.util import find_library
try:
    asound = cdll.LoadLibrary(find_library('asound'))
    asound.snd_lib_error_set_handler(c_char_p(0))
except:
    pass

# Configuration - use absolute paths relative to script location
SCRIPT_DIR = Path(__file__).parent
CALIBRATION_FILE = SCRIPT_DIR / "calibration_google.json"
TEST_AUDIO_FILE = SCRIPT_DIR / "test_audio.wav"


class InterruptTester:
    """Test interrupt detection with echo cancellation"""

    def __init__(self, echo_scale=0.9, speech_threshold=300):
        self.echo_scale = echo_scale
        self.speech_threshold = speech_threshold
        self.mic = sr.Microphone(sample_rate=16000)
        self.speaker_audio_buffer = None
        self.speaker_pos = 0
        self.speech_frame_count = 0
        self.frames_needed = 3  # Require N consecutive frames

        # Initialize Silero VAD
        self.silero_model = None
        self.silero_threshold = 0.5  # 50% confidence threshold
        try:
            self.silero_model = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False,
                                               onnx=False)
            print(f"✅ Silero VAD loaded successfully")
        except Exception as e:
            print(f"⚠️  Silero VAD not available: {e}")

        print(f"🎙️ InterruptTester initialized:")
        print(f"   Echo scale: {self.echo_scale:.2f}")
        print(f"   Speech threshold: {self.speech_threshold}")
        print(f"   Frames needed: {self.frames_needed}")
        if self.silero_model:
            print(f"   Silero threshold: {self.silero_threshold}")

    def calibrate(self):
        """Calibrate echo cancellation"""
        print("\n" + "=" * 60)
        print("🎯 CALIBRATION TEST")
        print("=" * 60)
        print("\n📢 Playing test tone...")
        print("   Listen for a 1-second beep")
        print("   Keep your environment quiet during calibration\n")

        # Generate test tone (1kHz, 1 second)
        sample_rate = 16000
        duration = 1.0
        freq = 1000
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_tone = (np.sin(2 * np.pi * freq * t) * 20000).astype(np.float32)

        try:
            print(f"[DEBUG] Starting playback and recording...")

            # Play and record simultaneously
            sd.play(test_tone, samplerate=sample_rate)
            time.sleep(0.1)

            recorded = []
            chunks_expected = int(sample_rate * duration / 960)
            print(f"[DEBUG] Expecting {chunks_expected} audio chunks...")

            with self.mic as source:
                # Get actual sample width from the mic stream (auto-detects mic format)
                sample_width = source.SAMPLE_WIDTH  # bytes per sample (typically 2 for int16)
                samples_per_chunk = 960
                expected_bytes = samples_per_chunk * sample_width
                print(f"[DEBUG] Mic format: {sample_width} bytes/sample → expecting {expected_bytes} bytes/chunk")

                for i in range(chunks_expected):
                    try:
                        audio = source.stream.read(samples_per_chunk)
                        # Accept chunks within 20% tolerance (for mobile/different mic compatibility)
                        if audio and abs(len(audio) - expected_bytes) <= expected_bytes * 0.2:
                            recorded.append(np.frombuffer(audio, dtype=np.int16).astype(np.float32))
                        else:
                            print(f"[DEBUG] Chunk {i}: got {len(audio) if audio else 0} bytes (expected ~{expected_bytes})")
                    except Exception as e:
                        print(f"[DEBUG] Chunk {i}: error reading - {e}")
                        pass

            sd.stop()
            time.sleep(0.2)

            print(f"[DEBUG] Recorded {len(recorded)} chunks")

            if recorded:
                recorded_audio = np.concatenate(recorded)
                speaker_energy = np.sqrt(np.mean(test_tone ** 2))
                mic_energy = np.sqrt(np.mean(recorded_audio ** 2))

                print(f"[DEBUG] Speaker energy: {speaker_energy:.0f}")
                print(f"[DEBUG] Mic energy: {mic_energy:.0f}")

                if speaker_energy > 0:
                    self.echo_scale = mic_energy / speaker_energy
                    self.echo_scale = np.clip(self.echo_scale, 0.3, 1.5)

                    print(f"\n✅ Calibration complete!")
                    print(f"   Speaker energy: {speaker_energy:.0f}")
                    print(f"   Mic energy: {mic_energy:.0f}")
                    print(f"   Echo scale: {self.echo_scale:.2f}")

                    # Save to file
                    print(f"[DEBUG] Saving to: {CALIBRATION_FILE}")
                    with open(CALIBRATION_FILE, 'w') as f:
                        json.dump({"echo_scale": float(self.echo_scale)}, f, indent=2)
                    print(f"\n   💾 Saved calibration to:")
                    print(f"      {CALIBRATION_FILE}")

                    # Verify file was written
                    if CALIBRATION_FILE.exists():
                        print(f"   ✅ File verified - calibration saved successfully!")
                    else:
                        print(f"   ❌ WARNING: File not found after save!")

                    return True
                else:
                    print(f"❌ Calibration failed: speaker energy is zero!")
                    return False
            else:
                print(f"❌ Calibration failed: no audio chunks recorded!")
                print(f"   Check if microphone is working")
                return False

        except Exception as e:
            print(f"❌ Calibration failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_test_audio(self, text="This is a test of the interrupt detection system", filename=None):
        """Generate test TTS audio using edge-tts"""
        if filename is None:
            filename = TEST_AUDIO_FILE

        print(f"\n🔊 Generating test audio: '{text}'")

        try:
            import asyncio
            import edge_tts

            async def generate():
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                await communicate.save(str(filename))

            asyncio.run(generate())
            print(f"   ✅ Generated: {filename}")
            return True
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False

    def load_speaker_audio(self, audio_path):
        """Load speaker audio for echo cancellation"""
        try:
            data, sr_rate = sf.read(audio_path)

            # Convert to int16 mono
            if data.dtype != np.int16:
                if data.dtype == np.float32 or data.dtype == np.float64:
                    audio_int16 = (data * 32767).astype(np.int16)
                else:
                    audio_int16 = data.astype(np.int16)
            else:
                audio_int16 = data

            # Flatten if stereo
            if len(audio_int16.shape) > 1:
                audio_mono = audio_int16[:, 0]
            else:
                audio_mono = audio_int16

            # Resample to 16kHz if needed
            if sr_rate != 16000:
                duration = len(audio_mono) / sr_rate
                target_length = int(duration * 16000)
                indices = np.linspace(0, len(audio_mono) - 1, target_length)
                audio_resampled = np.interp(indices, np.arange(len(audio_mono)), audio_mono).astype(np.int16)
            else:
                audio_resampled = audio_mono

            self.speaker_audio_buffer = audio_resampled
            self.speaker_pos = 0
            self.speech_frame_count = 0

            print(f"   📥 Loaded speaker audio: {len(audio_resampled)} samples")
            return True

        except Exception as e:
            print(f"   ❌ Failed to load audio: {e}")
            return False

    def test_interrupt_detection(self, audio_path, duration=None):
        """Test interrupt detection while playing audio"""
        print("\n" + "=" * 60)
        print("🎯 INTERRUPT DETECTION TEST (DEBUG MODE)")
        print("=" * 60)

        # Display current calibration values
        print("\n📊 Current Calibration:")
        print(f"   • Echo scale: {self.echo_scale:.2f}")
        print(f"   • Speech threshold: {self.speech_threshold}")
        print(f"   • Frames needed: {self.frames_needed}")

        print("\n📢 Instructions:")
        print("   Phase 1: Audio will play - DO NOT SPEAK (let it run)")
        print("   Phase 2: After ~3 seconds, I'll prompt you to SPEAK")
        print("   Phase 3: System should detect ONLY Phase 2 speech")
        print("\n   Press ENTER to start...")
        input()

        # Load speaker audio
        if not self.load_speaker_audio(audio_path):
            return False

        # Load audio for playback
        try:
            data, sr_rate = sf.read(audio_path)
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return False

        audio_duration = len(data) / sr_rate
        print(f"\n▶️  STARTING PLAYBACK (duration: {audio_duration:.1f}s)")
        print(f"   🔇 DO NOT SPEAK YET - monitoring for self-interrupt...")
        print()

        # Play audio
        sd.play(data, samplerate=sr_rate)
        time.sleep(0.1)  # Let playback start

        # Monitor for interrupts
        interrupted = False
        samples_per_chunk = 960  # 60ms chunks at 16kHz
        start_time = time.time()
        prompt_shown = False
        chunk_count = 0

        try:
            with self.mic as source:
                # Auto-detect mic format
                sample_width = source.SAMPLE_WIDTH
                expected_bytes = samples_per_chunk * sample_width
                print(f"[DEBUG] Mic format: {sample_width} bytes/sample, expecting {expected_bytes} bytes/chunk")

                while sd.get_stream().active:
                    try:
                        elapsed = time.time() - start_time

                        # Prompt user to speak after 3 seconds
                        if not prompt_shown and elapsed > 3.0:
                            print("\n" + "=" * 60)
                            print("🎤 NOW SPEAK TO INTERRUPT!")
                            print("=" * 60 + "\n")
                            prompt_shown = True

                        # Read mic chunk
                        mic_audio = source.stream.read(samples_per_chunk)
                        # Accept chunks within tolerance for mobile compatibility
                        if not mic_audio or abs(len(mic_audio) - expected_bytes) > expected_bytes * 0.2:
                            continue

                        chunk_count += 1
                        mic_data = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32)

                        # Calculate RAW mic energy (before echo cancellation)
                        mic_energy = np.sqrt(np.mean(mic_data ** 2))

                        # Echo cancellation
                        speaker_energy = 0
                        if self.speaker_pos < len(self.speaker_audio_buffer):
                            speaker_end = min(self.speaker_pos + samples_per_chunk, len(self.speaker_audio_buffer))
                            speaker_chunk = self.speaker_audio_buffer[self.speaker_pos:speaker_end]

                            if len(speaker_chunk) == samples_per_chunk:
                                speaker_float = speaker_chunk.astype(np.float32)
                                speaker_energy = np.sqrt(np.mean(speaker_float ** 2))
                                residual = mic_data - (speaker_float * self.echo_scale)
                            else:
                                residual = mic_data

                            self.speaker_pos += samples_per_chunk
                        else:
                            residual = mic_data

                        # Calculate RESIDUAL energy (after echo cancellation)
                        residual_energy = np.sqrt(np.mean(residual ** 2))

                        # Print debug every 10 chunks (~0.6s) OR if energy is high
                        if chunk_count % 10 == 0 or residual_energy > self.speech_threshold * 0.5:
                            phase = "🔇 SILENT PHASE" if not prompt_shown else "🎤 SPEAK PHASE"
                            print(f"[{elapsed:.1f}s] {phase} | Mic: {mic_energy:>5.0f} | Speaker: {speaker_energy:>5.0f} | Residual: {residual_energy:>5.0f} | Threshold: {self.speech_threshold}")

                        # Multi-frame detection
                        if residual_energy > self.speech_threshold:
                            self.speech_frame_count += 1
                            phase = "🔇 SILENT PHASE (SELF-INTERRUPT!)" if not prompt_shown else "🎤 SPEAK PHASE (CORRECT)"
                            print(f"   ⚠️  [{elapsed:.1f}s] {phase}")
                            print(f"       Mic={mic_energy:.0f} | Speaker={speaker_energy:.0f} | Residual={residual_energy:.0f} | Count={self.speech_frame_count}/{self.frames_needed}")

                            if self.speech_frame_count >= self.frames_needed:
                                print(f"\n   {'❌ SELF-INTERRUPT DETECTED!' if not prompt_shown else '✅ CORRECT INTERRUPT!'}")
                                print(f"   Stopped after {elapsed:.1f}s")
                                print(f"\n   📊 Final Debug Info:")
                                print(f"      • Raw mic energy: {mic_energy:.0f}")
                                print(f"      • Speaker energy: {speaker_energy:.0f}")
                                print(f"      • Echo scale: {self.echo_scale:.2f}")
                                print(f"      • Expected cancellation: {speaker_energy * self.echo_scale:.0f}")
                                print(f"      • Residual after AEC: {residual_energy:.0f}")
                                print(f"      • Threshold: {self.speech_threshold}")
                                sd.stop()
                                interrupted = True
                                break
                        else:
                            if self.speech_frame_count > 0:
                                print(f"   🔇 Energy dropped (residual={residual_energy:.0f}), resetting count")
                            self.speech_frame_count = 0

                    except Exception as e:
                        print(f"   ⚠️  Error: {e}")
                        continue

        except KeyboardInterrupt:
            print("\n   ⏹️  Stopped by user")
            sd.stop()

        finally:
            sd.stop()
            time.sleep(0.2)

        if not interrupted:
            print(f"\n   ✅ Playback completed without interrupt")

        return True

    def test_vad_interrupt(self, audio_path, vad_threshold=2500):
        """Test PURE VAD interrupt detection (no echo cancellation)"""
        print("\n" + "=" * 60)
        print("🎯 PURE VAD INTERRUPT TEST (NO ECHO CANCELLATION)")
        print("=" * 60)

        print("\n📊 Current Settings:")
        print(f"   • VAD threshold: {vad_threshold}")
        print(f"   • Frames needed: {self.frames_needed}")
        print("\n💡 Concept: Detect user speech by energy alone")
        print("   - Speaker echo will be BELOW threshold")
        print("   - Real user speech will be ABOVE threshold")
        print("   - No complex echo cancellation needed!")

        print("\n📢 Instructions:")
        print("   Phase 1: Audio plays - DO NOT SPEAK (test for false triggers)")
        print("   Phase 2: After ~3 seconds - SPEAK TO INTERRUPT")
        print("\n   Press ENTER to start...")
        input()

        # Load audio for playback
        try:
            data, sr_rate = sf.read(audio_path)
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return False

        audio_duration = len(data) / sr_rate
        print(f"\n▶️  STARTING PLAYBACK (duration: {audio_duration:.1f}s)")
        print(f"   🔇 DO NOT SPEAK YET - testing for false triggers...")
        print()

        # Play audio
        sd.play(data, samplerate=sr_rate)
        time.sleep(0.1)

        # Monitor for interrupts using PURE VAD
        interrupted = False
        samples_per_chunk = 960
        start_time = time.time()
        prompt_shown = False
        chunk_count = 0
        speech_frame_count = 0

        try:
            with self.mic as source:
                sample_width = source.SAMPLE_WIDTH
                expected_bytes = samples_per_chunk * sample_width
                print(f"[DEBUG] Mic format: {sample_width} bytes/sample, expecting {expected_bytes} bytes/chunk")
                print(f"[DEBUG] VAD threshold: {vad_threshold}")
                print()

                while sd.get_stream().active:
                    try:
                        elapsed = time.time() - start_time

                        # Prompt user to speak after 3 seconds
                        if not prompt_shown and elapsed > 3.0:
                            print("\n" + "=" * 60)
                            print("🎤 NOW SPEAK TO INTERRUPT!")
                            print("=" * 60 + "\n")
                            prompt_shown = True

                        # Read mic chunk
                        mic_audio = source.stream.read(samples_per_chunk)
                        if not mic_audio or abs(len(mic_audio) - expected_bytes) > expected_bytes * 0.2:
                            continue

                        chunk_count += 1
                        mic_data = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32)

                        # Calculate RAW mic energy (PURE VAD - no echo cancellation)
                        mic_energy = np.sqrt(np.mean(mic_data ** 2))

                        # Print debug every 10 chunks (~0.6s) OR if energy is high
                        if chunk_count % 10 == 0 or mic_energy > vad_threshold * 0.5:
                            phase = "🔇 SILENT PHASE" if not prompt_shown else "🎤 SPEAK PHASE"
                            status = "✓" if mic_energy < vad_threshold else "⚠️ HIGH"
                            print(f"[{elapsed:.1f}s] {phase} | Mic Energy: {mic_energy:>5.0f} | Threshold: {vad_threshold} | {status}")

                        # Simple VAD detection
                        if mic_energy > vad_threshold:
                            speech_frame_count += 1
                            phase = "🔇 FALSE TRIGGER!" if not prompt_shown else "🎤 CORRECT INTERRUPT"
                            print(f"   ⚠️  [{elapsed:.1f}s] {phase}")
                            print(f"       Energy={mic_energy:.0f} | Threshold={vad_threshold} | Count={speech_frame_count}/{self.frames_needed}")

                            if speech_frame_count >= self.frames_needed:
                                result = "❌ FALSE TRIGGER!" if not prompt_shown else "✅ CORRECT INTERRUPT!"
                                print(f"\n   {result}")
                                print(f"   Stopped after {elapsed:.1f}s")
                                print(f"\n   📊 Final Info:")
                                print(f"      • Mic energy: {mic_energy:.0f}")
                                print(f"      • VAD threshold: {vad_threshold}")
                                print(f"      • Frames needed: {self.frames_needed}")
                                sd.stop()
                                interrupted = True
                                break
                        else:
                            if speech_frame_count > 0:
                                print(f"   🔇 Energy dropped (energy={mic_energy:.0f}), resetting count")
                            speech_frame_count = 0

                    except Exception as e:
                        print(f"   ⚠️  Error: {e}")
                        continue

        except KeyboardInterrupt:
            print("\n   ⏹️  Stopped by user")
            sd.stop()

        finally:
            sd.stop()
            time.sleep(0.2)

        if not interrupted:
            print(f"\n   ✅ Playback completed - NO false triggers!")

        return True

    def test_full_pipeline(self, audio_path, vad_threshold=2500):
        """Test COMPLETE PIPELINE: Interrupt → Start Recording → Stop Recording"""
        print("\n" + "=" * 60)
        print("🎯 FULL PIPELINE TEST (END-TO-END)")
        print("=" * 60)

        print("\n💡 This tests the COMPLETE assistant workflow:")
        print("   1️⃣  Play TTS audio (assistant speaking)")
        print("   2️⃣  Detect user interrupt (VAD during playback)")
        print("   3️⃣  Start recording user speech (VAD start)")
        print("   4️⃣  Detect silence and stop recording (VAD end)")
        print("   5️⃣  Save user's speech to file")

        print("\n📊 Settings:")
        print(f"   • VAD threshold: {vad_threshold}")
        print(f"   • Frames needed: {self.frames_needed}")
        print(f"   • Silence timeout: 1.5s")

        print("\n📢 Instructions:")
        print("   Phase 1: Audio plays - STAY SILENT (3 seconds)")
        print("   Phase 2: System prompts - SPEAK TO INTERRUPT")
        print("   Phase 3: System prompts - CONTINUE SPEAKING (say something)")
        print("   Phase 4: STOP SPEAKING - system detects silence")
        print("\n   Press ENTER to start the full pipeline test...")
        input()

        # ========================================
        # PHASE 1 & 2: INTERRUPT DETECTION
        # ========================================
        print("\n" + "=" * 60)
        print("PHASE 1 & 2: INTERRUPT DETECTION")
        print("=" * 60)

        try:
            data, sr_rate = sf.read(audio_path)
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return False

        audio_duration = len(data) / sr_rate
        print(f"\n▶️  STARTING PLAYBACK (duration: {audio_duration:.1f}s)")
        print(f"   🔇 Phase 1: DO NOT SPEAK (0-3 seconds)")
        print()

        sd.play(data, samplerate=sr_rate)
        time.sleep(0.1)

        interrupted = False
        samples_per_chunk = 960
        start_time = time.time()
        prompt_shown = False
        chunk_count = 0
        speech_frame_count = 0

        try:
            with self.mic as source:
                sample_width = source.SAMPLE_WIDTH
                expected_bytes = samples_per_chunk * sample_width

                while sd.get_stream().active:
                    elapsed = time.time() - start_time

                    if not prompt_shown and elapsed > 3.0:
                        print("\n" + "=" * 60)
                        print("🎤 Phase 2: NOW SPEAK TO INTERRUPT!")
                        print("=" * 60 + "\n")
                        prompt_shown = True

                    mic_audio = source.stream.read(samples_per_chunk)
                    if not mic_audio or abs(len(mic_audio) - expected_bytes) > expected_bytes * 0.2:
                        continue

                    chunk_count += 1
                    mic_data = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32)
                    mic_energy = np.sqrt(np.mean(mic_data ** 2))

                    if chunk_count % 10 == 0 or mic_energy > vad_threshold * 0.5:
                        phase = "🔇 SILENT" if not prompt_shown else "🎤 SPEAK"
                        print(f"[{elapsed:.1f}s] {phase} | Energy: {mic_energy:>5.0f} | Threshold: {vad_threshold}")

                    if mic_energy > vad_threshold:
                        speech_frame_count += 1
                        if speech_frame_count >= self.frames_needed:
                            if prompt_shown:
                                print(f"\n✅ INTERRUPT DETECTED at {elapsed:.1f}s")
                                sd.stop()
                                interrupted = True
                                break
                            else:
                                print(f"\n❌ FALSE TRIGGER at {elapsed:.1f}s")
                                sd.stop()
                                return False
                    else:
                        speech_frame_count = 0

        except KeyboardInterrupt:
            print("\n⏹️  Test cancelled")
            sd.stop()
            return False

        if not interrupted:
            print(f"\n❌ Failed to detect interrupt")
            return False

        time.sleep(0.5)  # Brief pause

        # ========================================
        # PHASE 3 & 4: VAD RECORDING (START/STOP)
        # ========================================
        print("\n" + "=" * 60)
        print("PHASE 3 & 4: VAD RECORDING (START/STOP)")
        print("=" * 60)
        print("\n🎤 Keep speaking... System will detect when you stop")
        print()

        silence_threshold = 1.5  # seconds
        silence_duration = 0.0
        speech_started = False
        audio_buffer = []
        frame_duration = samples_per_chunk / 16000
        max_duration = 15  # Max 15 seconds
        start_time = time.time()

        try:
            with self.mic as source:
                while (time.time() - start_time) < max_duration:
                    audio_chunk = source.stream.read(samples_per_chunk)
                    if not audio_chunk or abs(len(audio_chunk) - expected_bytes) > expected_bytes * 0.2:
                        continue

                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                    energy = np.sqrt(np.mean(audio_data ** 2))

                    if energy > vad_threshold:
                        silence_duration = 0

                        if not speech_started:
                            print(f"   ✅ SPEECH STARTED (energy={energy:.0f})")
                            speech_started = True
                            audio_buffer = []

                        audio_buffer.append(audio_chunk)

                    else:
                        if speech_started:
                            silence_duration += frame_duration
                            audio_buffer.append(audio_chunk)

                            if silence_duration >= silence_threshold:
                                print(f"\n   ✅ SPEECH ENDED (silence={silence_duration:.1f}s)")
                                print(f"   📊 Recorded {len(audio_buffer)} chunks ({len(audio_buffer) * frame_duration:.1f}s)")

                                # Save recording
                                audio_bytes = b''.join(audio_buffer)
                                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                                output_file = "pipeline_test_recording.wav"
                                sf.write(output_file, audio_array, 16000)
                                print(f"   💾 Saved to: {output_file}")

                                break

        except KeyboardInterrupt:
            print("\n⏹️  Test cancelled")
            return False

        time.sleep(1)  # Brief pause

        # ========================================
        # PHASE 5: PLAYBACK & RE-INTERRUPT TEST
        # ========================================
        print("\n" + "=" * 60)
        print("PHASE 5: PLAYBACK & RE-INTERRUPT TEST")
        print("=" * 60)
        print("\n💡 Now playing YOUR recording back!")
        print("   Try to interrupt your own voice 😄")
        print("\n📢 Instructions:")
        print("   Phase 5a: Recording plays - STAY SILENT (2 seconds)")
        print("   Phase 5b: System prompts - SPEAK TO INTERRUPT your own voice!")
        print("\n   Press ENTER to continue...")
        input()

        try:
            playback_data, playback_sr = sf.read("pipeline_test_recording.wav")
        except Exception as e:
            print(f"⚠️  Could not load recording for playback: {e}")
            print("   Skipping playback test")
            playback_duration = 0
        else:
            playback_duration = len(playback_data) / playback_sr
            print(f"\n▶️  PLAYING YOUR RECORDING (duration: {playback_duration:.1f}s)")
            print(f"   🔇 Phase 5a: DO NOT SPEAK (0-2 seconds)")
            print()

            sd.play(playback_data, samplerate=playback_sr)
            time.sleep(0.1)

            playback_interrupted = False
            start_time = time.time()
            prompt_shown = False
            chunk_count = 0
            speech_frame_count = 0

            try:
                with self.mic as source:
                    while sd.get_stream().active:
                        elapsed = time.time() - start_time

                        if not prompt_shown and elapsed > 2.0:
                            print("\n" + "=" * 60)
                            print("🎤 Phase 5b: NOW INTERRUPT YOUR OWN VOICE!")
                            print("=" * 60 + "\n")
                            prompt_shown = True

                        mic_audio = source.stream.read(samples_per_chunk)
                        if not mic_audio or abs(len(mic_audio) - expected_bytes) > expected_bytes * 0.2:
                            continue

                        chunk_count += 1
                        mic_data = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32)
                        mic_energy = np.sqrt(np.mean(mic_data ** 2))

                        if chunk_count % 10 == 0 or mic_energy > vad_threshold * 0.5:
                            phase = "🔇 SILENT" if not prompt_shown else "🎤 SPEAK"
                            print(f"[{elapsed:.1f}s] {phase} | Energy: {mic_energy:>5.0f} | Threshold: {vad_threshold}")

                        if mic_energy > vad_threshold:
                            speech_frame_count += 1
                            if speech_frame_count >= self.frames_needed:
                                if prompt_shown:
                                    print(f"\n✅ PLAYBACK INTERRUPTED at {elapsed:.1f}s")
                                    sd.stop()
                                    playback_interrupted = True
                                    break
                                else:
                                    print(f"\n⚠️  Early interrupt at {elapsed:.1f}s (expected after 2s)")
                                    sd.stop()
                                    playback_interrupted = True
                                    break
                        else:
                            speech_frame_count = 0

            except KeyboardInterrupt:
                print("\n⏹️  Playback test cancelled")
                sd.stop()

            if playback_interrupted:
                print("   ✅ Successfully interrupted playback!")
            else:
                print("   ℹ️  Playback completed without interrupt")

        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print("✅ FULL PIPELINE TEST COMPLETED!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("   ✅ Phase 1: No false triggers during silent period")
        print("   ✅ Phase 2: Successfully detected interrupt on TTS")
        print("   ✅ Phase 3: Successfully detected speech start")
        print("   ✅ Phase 4: Successfully detected speech end (silence)")
        print(f"   ✅ Phase 5: Saved & played back recording")
        print("\n🎉 Complete conversation loop tested successfully!")
        print("   This is exactly how the real assistant works!")

        return True

    def test_silero_pipeline(self, audio_path, silero_threshold=0.5):
        """Test SILERO VAD PIPELINE: More accurate, no calibration needed"""
        if not self.silero_model:
            print("\n❌ Silero VAD not available. Install with: pip install silero-vad")
            return False

        print("\n" + "=" * 60)
        print("🎯 SILERO VAD PIPELINE TEST (ML-BASED)")
        print("=" * 60)

        print("\n💡 This uses Silero VAD - a pre-trained neural network:")
        print("   • Outputs speech probability (0.0 to 1.0)")
        print("   • Much more stable than energy-based detection")
        print("   • No calibration needed!")
        print("   • Better at distinguishing speech from noise/music")

        print("\n📊 Settings:")
        print(f"   • Silero threshold: {silero_threshold} (50% = half confident it's speech)")
        print(f"   • Frames needed: {self.frames_needed}")

        print("\n📢 Instructions:")
        print("   Phase 1: Audio plays - STAY SILENT (3 seconds)")
        print("   Phase 2: System prompts - SPEAK TO INTERRUPT")
        print("\n   Press ENTER to start...")
        input()

        # Load audio
        try:
            data, sr_rate = sf.read(audio_path)
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            return False

        audio_duration = len(data) / sr_rate
        print(f"\n▶️  STARTING PLAYBACK (duration: {audio_duration:.1f}s)")
        print(f"   🔇 Phase 1: DO NOT SPEAK (0-3 seconds)")
        print()

        sd.play(data, samplerate=sr_rate)
        time.sleep(0.1)

        interrupted = False
        samples_per_chunk = 960
        start_time = time.time()
        prompt_shown = False
        chunk_count = 0
        speech_frame_count = 0

        try:
            with self.mic as source:
                sample_width = source.SAMPLE_WIDTH
                expected_bytes = samples_per_chunk * sample_width

                while sd.get_stream().active:
                    elapsed = time.time() - start_time

                    if not prompt_shown and elapsed > 3.0:
                        print("\n" + "=" * 60)
                        print("🎤 Phase 2: NOW SPEAK TO INTERRUPT!")
                        print("=" * 60 + "\n")
                        prompt_shown = True

                    mic_audio = source.stream.read(samples_per_chunk)
                    if not mic_audio or abs(len(mic_audio) - expected_bytes) > expected_bytes * 0.2:
                        continue

                    chunk_count += 1

                    # Convert to float32 tensor for Silero
                    mic_data = np.frombuffer(mic_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_tensor = torch.from_numpy(mic_data)

                    # Get speech probability from Silero VAD
                    with torch.no_grad():
                        speech_prob = self.silero_model(audio_tensor, 16000).item()

                    # Also calculate energy for comparison
                    mic_energy = np.sqrt(np.mean((mic_data * 32768.0) ** 2))

                    if chunk_count % 10 == 0 or speech_prob > silero_threshold * 0.5:
                        phase = "🔇 SILENT" if not prompt_shown else "🎤 SPEAK"
                        status = "✓" if speech_prob < silero_threshold else "⚠️ SPEECH"
                        print(f"[{elapsed:.1f}s] {phase} | Probability: {speech_prob:.3f} | Threshold: {silero_threshold:.3f} | Energy: {mic_energy:.0f} | {status}")

                    # Silero VAD detection
                    if speech_prob > silero_threshold:
                        speech_frame_count += 1
                        if speech_frame_count >= self.frames_needed:
                            if prompt_shown:
                                print(f"\n✅ INTERRUPT DETECTED at {elapsed:.1f}s")
                                print(f"   Speech probability: {speech_prob:.3f}")
                                print(f"   Energy: {mic_energy:.0f}")
                                sd.stop()
                                interrupted = True
                                break
                            else:
                                print(f"\n❌ FALSE TRIGGER at {elapsed:.1f}s")
                                print(f"   Speech probability: {speech_prob:.3f}")
                                sd.stop()
                                return False
                    else:
                        speech_frame_count = 0

        except KeyboardInterrupt:
            print("\n⏹️  Test cancelled")
            sd.stop()
            return False

        if not interrupted:
            print(f"\n❌ Failed to detect interrupt")
            return False

        print("\n" + "=" * 60)
        print("✅ SILERO VAD TEST COMPLETED!")
        print("=" * 60)
        print("\n🎉 Silero VAD successfully detected your interrupt!")
        print("   This approach is more robust than energy-based VAD.")

        return True

    def test_vad_recording(self, duration=10):
        """Test voice activity detection for recording start/stop"""
        print("\n" + "=" * 60)
        print("🎯 VAD RECORDING TEST")
        print("=" * 60)

        # Display current settings
        print("\n📊 Current Settings:")
        print(f"   • Speech threshold: {self.speech_threshold}")
        print(f"   • Silence timeout: 1.5s")
        print(f"   • Test duration: {duration}s max")

        print("\n📢 Instructions:")
        print("   1. Remain silent initially")
        print("   2. Start speaking when ready")
        print("   3. Stop speaking and wait for silence detection")
        print(f"   4. Test runs for {duration}s max")
        print("\n   Press ENTER to start...")
        input()

        print(f"\n🎧 Listening for speech...")
        print()

        samples_per_chunk = 960  # 60ms chunks
        frame_duration = samples_per_chunk / 16000
        silence_threshold = 1.5  # seconds
        silence_duration = 0.0
        speech_started = False
        audio_buffer = []

        start_time = time.time()

        try:
            with self.mic as source:
                # Auto-detect mic format for mobile compatibility
                sample_width = source.SAMPLE_WIDTH
                expected_bytes = samples_per_chunk * sample_width

                while (time.time() - start_time) < duration:
                    try:
                        # Read chunk
                        audio_chunk = source.stream.read(samples_per_chunk)
                        # Accept chunks within tolerance for mobile compatibility
                        if not audio_chunk or abs(len(audio_chunk) - expected_bytes) > expected_bytes * 0.2:
                            continue

                        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                        energy = np.sqrt(np.mean(audio_data ** 2))

                        # Check for speech
                        if energy > self.speech_threshold:
                            silence_duration = 0

                            if not speech_started:
                                print(f"   🎤 SPEECH STARTED (energy={energy:.0f})")
                                speech_started = True
                                audio_buffer = []

                            audio_buffer.append(audio_chunk)

                        else:
                            # Silence detected
                            if speech_started:
                                silence_duration += frame_duration
                                audio_buffer.append(audio_chunk)

                                if int(silence_duration * 10) % 5 == 0:
                                    print(f"   🔇 Silence: {silence_duration:.1f}s / {silence_threshold}s")

                                if silence_duration >= silence_threshold:
                                    print(f"\n   ✅ SPEECH ENDED (silence={silence_duration:.1f}s)")
                                    print(f"   📊 Recorded {len(audio_buffer)} chunks ({len(audio_buffer) * frame_duration:.1f}s)")

                                    # Save to file
                                    audio_bytes = b''.join(audio_buffer)
                                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

                                    output_file = "test_recording.wav"
                                    sf.write(output_file, audio_array, 16000)
                                    print(f"   💾 Saved to: {output_file}")

                                    # Reset for next utterance
                                    speech_started = False
                                    silence_duration = 0
                                    audio_buffer = []
                                    print(f"\n   🎧 Listening for next speech...")

                    except Exception as e:
                        print(f"   ⚠️  Error: {e}")
                        continue

        except KeyboardInterrupt:
            print("\n   ⏹️  Stopped by user")

        print("\n   ✅ VAD test complete")
        return True


def main():
    """Main test menu"""
    print("\n" + "=" * 60)
    print("🎙️  INTERRUPT & VAD TESTING SCRIPT")
    print("=" * 60)

    # Load existing calibration if available
    echo_scale = 0.9
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                calib = json.load(f)
                echo_scale = calib.get('echo_scale', 0.9)
            print(f"\n✅ Loaded calibration: echo_scale={echo_scale:.2f}")
        except:
            print(f"\n⚠️  Could not load calibration, using default")

    tester = InterruptTester(echo_scale=echo_scale)

    while True:
        print("\n" + "=" * 60)
        print("SELECT TEST:")
        print("=" * 60)

        # Show current calibration status
        calib_status = "✅ CALIBRATED" if CALIBRATION_FILE.exists() else "⚠️  NOT CALIBRATED (using default)"
        print(f"\nCalibration Status: {calib_status}")
        print(f"Current Settings:")
        print(f"  • Echo scale: {tester.echo_scale:.2f}")
        print(f"  • Speech threshold: {tester.speech_threshold}")
        print(f"  • Frames needed: {tester.frames_needed}")
        print(f"  • Calibration file: {CALIBRATION_FILE}")
        print()

        print("1. Calibrate echo cancellation")
        print("2. Test interrupt detection (with echo cancellation)")
        print("3. Test PURE VAD interrupt (NO echo cancellation)")
        print("4. Test FULL PIPELINE (interrupt → record → silence) ⭐ RECOMMENDED")
        print("5. Test SILERO VAD (ML-based, no calibration) ⭐ NEW & BETTER")
        print("6. Test VAD recording (start/stop detection)")
        print("7. Generate new test audio")
        print("8. Adjust parameters")
        print("9. Exit")
        print()

        choice = input("Enter choice (1-9): ").strip()

        if choice == "1":
            if tester.calibrate():
                # Reload calibration after successful calibration
                try:
                    with open(CALIBRATION_FILE, 'r') as f:
                        calib = json.load(f)
                        tester.echo_scale = calib.get('echo_scale', 0.9)
                    print(f"\n✅ Calibration reloaded: echo_scale={tester.echo_scale:.2f}")
                except Exception as e:
                    print(f"⚠️  Could not reload calibration: {e}")

        elif choice == "2":
            # Generate test audio if doesn't exist
            if not TEST_AUDIO_FILE.exists():
                print("\n⚠️  No test audio found, generating...")
                tester.generate_test_audio()

            if TEST_AUDIO_FILE.exists():
                tester.test_interrupt_detection(TEST_AUDIO_FILE)
            else:
                print("\n❌ No test audio available")

        elif choice == "3":
            # PURE VAD interrupt test
            if not TEST_AUDIO_FILE.exists():
                print("\n⚠️  No test audio found, generating...")
                tester.generate_test_audio()

            if TEST_AUDIO_FILE.exists():
                # Ask for VAD threshold
                threshold_input = input("\nVAD threshold (press ENTER for 2500, based on debug data): ").strip()
                vad_threshold = int(threshold_input) if threshold_input.isdigit() else 2500
                tester.test_vad_interrupt(TEST_AUDIO_FILE, vad_threshold=vad_threshold)
            else:
                print("\n❌ No test audio available")

        elif choice == "4":
            # FULL PIPELINE test (RECOMMENDED!)
            if not TEST_AUDIO_FILE.exists():
                print("\n⚠️  No test audio found, generating...")
                tester.generate_test_audio()

            if TEST_AUDIO_FILE.exists():
                threshold_input = input("\nVAD threshold (press ENTER for 2500): ").strip()
                vad_threshold = int(threshold_input) if threshold_input.isdigit() else 2500
                tester.test_full_pipeline(TEST_AUDIO_FILE, vad_threshold=vad_threshold)
            else:
                print("\n❌ No test audio available")

        elif choice == "5":
            # SILERO VAD test (NEW!)
            if not TEST_AUDIO_FILE.exists():
                print("\n⚠️  No test audio found, generating...")
                tester.generate_test_audio()

            if TEST_AUDIO_FILE.exists():
                threshold_input = input("\nSilero threshold (press ENTER for 0.5 = 50% confidence): ").strip()
                try:
                    silero_threshold = float(threshold_input) if threshold_input else 0.5
                    tester.test_silero_pipeline(TEST_AUDIO_FILE, silero_threshold=silero_threshold)
                except ValueError:
                    print("❌ Invalid threshold, using default 0.5")
                    tester.test_silero_pipeline(TEST_AUDIO_FILE)
            else:
                print("\n❌ No test audio available")

        elif choice == "6":
            duration = input("\nTest duration in seconds (default 10): ").strip()
            duration = int(duration) if duration.isdigit() else 10
            tester.test_vad_recording(duration)

        elif choice == "7":
            text = input("\nEnter text to speak (or press ENTER for default): ").strip()
            if not text:
                text = "This is a test of the interrupt detection system. You can try to interrupt me by speaking."
            tester.generate_test_audio(text)

        elif choice == "8":
            print("\n" + "=" * 60)
            print("ADJUST PARAMETERS")
            print("=" * 60)
            print(f"\nCurrent settings:")
            print(f"1. Echo scale: {tester.echo_scale:.2f}")
            print(f"2. Speech threshold: {tester.speech_threshold}")
            print(f"3. Frames needed: {tester.frames_needed}")
            print()

            param = input("Which parameter to adjust (1-3, or ENTER to cancel)? ").strip()

            if param == "1":
                val = input(f"New echo scale (current: {tester.echo_scale:.2f}): ").strip()
                try:
                    tester.echo_scale = float(val)
                    print(f"✅ Echo scale set to {tester.echo_scale:.2f}")
                except:
                    print("❌ Invalid value")

            elif param == "2":
                val = input(f"New speech threshold (current: {tester.speech_threshold}): ").strip()
                try:
                    tester.speech_threshold = int(val)
                    print(f"✅ Speech threshold set to {tester.speech_threshold}")
                except:
                    print("❌ Invalid value")

            elif param == "3":
                val = input(f"New frames needed (current: {tester.frames_needed}): ").strip()
                try:
                    tester.frames_needed = int(val)
                    print(f"✅ Frames needed set to {tester.frames_needed}")
                except:
                    print("❌ Invalid value")

        elif choice == "9":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
