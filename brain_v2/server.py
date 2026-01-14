#!/usr/bin/env python3
"""
BR_AI_N Voice Assistant Server
==============================

VOICE INTERACTION FLOW:
=======================

1. USER SPEAKS → VAD DETECTS VOICE
   - Browser captures microphone audio (16kHz, 16-bit PCM)
   - Audio chunks sent to server via WebSocket (/stream)
   - VAD (Voice Activity Detection) monitors incoming audio
   - When speech detected → START recording audio for STT

2. USER STOPS SPEAKING → VAD DETECTS SILENCE
   - After 0.5s of silence → STOP recording
   - Collected audio sent to STT backend for transcription

3. STT CONVERTS AUDIO → TEXT
   - Backend options: google, whisper_server, whisper (local)
   - Transcribed text returned to main pipeline

4. LLM GENERATES RESPONSE (with optional RAG)
   - If RAG enabled: Query InduAgent (FAISS vector search + Ollama)
   - If RAG disabled: Stream directly from Ollama API
   - Response text chunked for TTS

5. TTS GENERATES AUDIO → PLAYBACK (PARALLEL PIPELINE)
   - Response text cleaned (remove markdown, special chars)
   - Split into chunks (50 chars first, 150 chars rest)
   - PARALLEL PROCESSING:
     ┌─────────────────────────┐     ┌─────────────────────────┐
     │   TTSGenerator Thread   │     │   AudioPlayer Thread    │
     │   (Background)          │     │   (Background)          │
     │                         │     │                         │
     │  text → TTS → .mp3 ────────→  queue.get() → play()     │
     │                         │     │                         │
     │  Generates chunk N+1    │     │  Plays chunk N          │
     │  WHILE chunk N plays    │     │  SIMULTANEOUSLY         │
     └─────────────────────────┘     └─────────────────────────┘
   - TTSGenerator: Converts text chunks to audio files in background
   - AudioPlayer: Plays audio files from queue as they become ready
   - Overlap: Generation of next chunk happens during playback of current

6. INTERRUPT DETECTION (RUNS DURING PLAYBACK)
   - VADInterruptMonitor runs continuously during audio playback
   - If user speaks during playback:
     → STOP audio playback immediately
     → DISCARD remaining audio chunks
     → RESET to step 1 (ready for new input)

CONFIG (config.json):
=====================
- tts.backend: "edge" | "pyttsx3"
- stt.backend: "google" | "whisper_server" | "whisper"
- llm.model: Ollama model name (e.g., "llama3.1:8b")
- llm.use_rag: true | false (enable RAG retrieval)
- llm.options.temperature: 0.0-1.0 (default 0.2)
- llm.local.host/port: Ollama server address

KEY COMPONENTS:
===============
- WebStreamingAssistant: Main orchestrator class
- InduAgent: RAG agent (LangChain + FAISS + Ollama)
- TTSGenerator: Background TTS generation thread
- AudioPlayer: Audio playback queue with interrupt support
- VADInterruptMonitor: Monitors for user speech during playback
"""

import json
import threading
import requests
import time
import re
import os
import tempfile
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import asyncio
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_sock import Sock
import edge_tts

# TTS imports
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Chatterbox TTS import
try:
    from chatterbox_tts_client import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    print("⚠️ ChatterboxTTS not available")
    CHATTERBOX_AVAILABLE = False

# Import VAD Pipeline components
from vad_pipeline import VADProcessor, VADConfig

# Import RAG Agent
try:
    from indu_rag import InduAgent
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG not available: {e}")
    RAG_AVAILABLE = False

# ROS2 imports for playback status publishing
try:
    import rclpy
    from std_msgs.msg import Bool
    ROS2_AVAILABLE = True
except ImportError:
    print("⚠️ ROS2 not available - playback status will not be published to ROS2 topic")
    ROS2_AVAILABLE = False


# ============================================================
# ROS2 PLAYBACK STATUS PUBLISHER
# ============================================================

class PlaybackStatusPublisher:
    """Publishes playback status continuously to ROS2 topic /brain/playback_status at 10Hz"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern to ensure only one ROS2 node exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.node = None
        self.publisher = None
        self.publish_thread = None
        self._running = False
        self._is_playing = False  # Current playback state
        self._state_lock = threading.Lock()

        if not ROS2_AVAILABLE:
            print("⚠️ ROS2 not available - PlaybackStatusPublisher disabled")
            return

        try:
            # Initialize ROS2 if not already initialized
            # Use signal_handler_options=None to not override Python's signal handlers
            if not rclpy.ok():
                rclpy.init(signal_handler_options=None)

            # Create node
            self.node = rclpy.create_node('brain_playback_status')

            # Create publisher for Bool messages
            self.publisher = self.node.create_publisher(Bool, '/brain/playback_status', 10)

            # Start continuous publish thread (publishes at 10Hz)
            self._running = True
            self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
            self.publish_thread.start()

            print("✅ ROS2 PlaybackStatusPublisher initialized - publishing to /brain/playback_status @ 10Hz")

        except Exception as e:
            print(f"❌ Failed to initialize ROS2 publisher: {e}")
            self.node = None
            self.publisher = None

    def _publish_loop(self):
        """Background thread that continuously publishes status at 10Hz"""
        while self._running and self.node is not None:
            try:
                # Publish current state
                with self._state_lock:
                    current_state = self._is_playing

                msg = Bool()
                msg.data = current_state
                self.publisher.publish(msg)

                # Sleep to maintain ~10Hz publish rate
                time.sleep(0.1)

            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                if self._running:  # Only log if not shutting down
                    print(f"⚠️ ROS2 publish error: {e}")
                break

    def set_playing(self, is_playing: bool):
        """Update playback status (will be published continuously)"""
        with self._state_lock:
            if self._is_playing != is_playing:
                self._is_playing = is_playing
                print(f"📡 ROS2: /brain/playback_status → {is_playing}")

    def shutdown(self):
        """Cleanup ROS2 resources"""
        self._running = False
        if self.publish_thread:
            self.publish_thread.join(timeout=1)
        if self.node:
            self.node.destroy_node()
        # Don't call rclpy.shutdown() as other nodes may be using it


# Global ROS2 publisher instance
_ros2_publisher = None

def init_ros2_publisher():
    """Initialize the ROS2 publisher at startup"""
    global _ros2_publisher
    if _ros2_publisher is None and ROS2_AVAILABLE:
        _ros2_publisher = PlaybackStatusPublisher()
    return _ros2_publisher

def get_ros2_publisher():
    """Get the ROS2 publisher singleton"""
    return _ros2_publisher

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
sock = Sock(app)
CONFIG_FILE = Path(__file__).parent / "config.json"
CALIBRATION_FILE = Path(__file__).parent / "calibration_google.json"
CONVERSATIONS_FILE = Path(__file__).parent / "conversations.json"


# ============================================================
# CONFIGURATION MANAGEMENT
# ============================================================

def load_config():
    """Load configuration from config.json"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def save_config(config_data):
    """Save configuration to config.json"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


# ============================================================
# WHISPER SERVER API
# ============================================================

def transcribe_with_whisper_server(audio_data, config):
    """Transcribe audio using external Whisper ASR server"""
    try:
        stt_config = config.get('stt', {}).get('whisper_server', {})
        host = stt_config.get('host', '127.0.0.1')
        port = stt_config.get('port', 8001)
        model = stt_config.get('model', 'medium')
        language = stt_config.get('language', 'en')

        # Initial prompt helps Whisper recognize domain-specific terms from the knowledge base
        # This significantly improves accuracy for proper nouns and technical terms
        initial_prompt = stt_config.get('initial_prompt',
            # Names and People
            "Indu, INDU, Dr. Sudhir Srivastava, Dr. Sudhir Prem Srivastava, Doctor Srivastava, "
            "Vishwa, Dr. Vishwa Srivastava, Milan Rao, Dr. Mohit Bhandari, "
            # Company and Organizations
            "SS Innovations, SSi, SSi International, SSII, Nasdaq, CDSCO, FDA, IRCAD, SAIMS, "
            # Products
            "SSi Mantra, Mantra, SSi Mantra 3, MantrAsana, SSII MantrAsana, SSi Mudra, SSi Maya, SSi Yantra, SSi Sutra, "
            # Medical Terms
            "telesurgery, tele-surgery, tele-proctoring, tele-mentoring, TECAB, "
            "robotic cardiac surgery, robotic surgery, cardiothoracic surgery, thoracic surgery, "
            "minimally invasive surgery, coronary artery bypass, atrial septal defect, "
            "gastric bypass, pyeloplasty, MIDCAB, laparoscopic, endoscopic, endosurgical, "
            "urology, gynecology, oncology, pediatric, humanoid robot, "
            # Technical Terms
            "3D 4K, 3D HD, modular arms, surgeon command center, dual console, "
            # Locations
            "Gurugram, New Delhi, Jaipur, Indore, Strasbourg"
        )

        url = f"http://{host}:{port}/transcribe"

        # Convert AudioData to WAV bytes
        wav_data = audio_data.get_wav_data(convert_rate=16000)

        # Send to whisper server with initial_prompt for better accuracy on domain-specific terms
        files = {'file': ('audio.wav', wav_data, 'audio/wav')}
        params = {'model': model, 'language': language, 'initial_prompt': initial_prompt}

        response = requests.post(url, files=files, params=params, timeout=30)
        response.raise_for_status()

        result = response.json()
        text = result.get('text', '').strip()

        return text if text else None

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Whisper server at {url}")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ Whisper server timeout")
        return None
    except Exception as e:
        print(f"❌ Whisper server error: {e}")
        return None


# ============================================================
# CALIBRATION
# ============================================================

def run_calibration():
    """Run audio calibration for interrupt detection"""
    try:
        print("\n🎙️ Starting calibration from web UI...")
        print("   Playing test tone (listen for beep)...")

        # Generate a test tone (1kHz sine wave, 1 second)
        sample_rate = 16000
        duration = 1.0
        freq = 1000
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_tone = (np.sin(2 * np.pi * freq * t) * 20000).astype(np.float32)

        # Create microphone
        mic = sr.Microphone(sample_rate=16000)

        # Play test tone
        sd.play(test_tone, samplerate=sample_rate)
        time.sleep(0.1)  # Let playback start

        # Record what the mic hears
        recorded = []
        with mic as source:
            for _ in range(int(sample_rate * duration / 960)):
                try:
                    audio = source.stream.read(960)
                    if audio and len(audio) == 960:
                        recorded.append(np.frombuffer(audio, dtype=np.int16).astype(np.float32))
                except:
                    pass

        sd.stop()
        time.sleep(0.2)

        if recorded:
            recorded_audio = np.concatenate(recorded)
            # Calculate actual echo scale (how much of speaker appears in mic)
            speaker_energy = np.sqrt(np.mean(test_tone ** 2))
            mic_energy = np.sqrt(np.mean(recorded_audio ** 2))

            if speaker_energy > 0:
                echo_scale = mic_energy / speaker_energy
                echo_scale = np.clip(echo_scale, 0.5, 1.5)  # Reasonable range
                print(f"   ✅ Calibration complete (echo scale: {echo_scale:.2f})")

                # Save to file
                try:
                    with open(CALIBRATION_FILE, 'w') as f:
                        json.dump({"echo_scale": float(echo_scale)}, f)
                    print(f"   💾 Saved to {CALIBRATION_FILE}")
                    return {
                        'success': True,
                        'echo_scale': float(echo_scale),
                        'message': f'Calibration successful! Echo scale: {echo_scale:.2f}. Restart the assistant to apply.'
                    }
                except Exception as e:
                    print(f"   ⚠️ Could not save calibration: {e}")
                    return {
                        'success': False,
                        'error': f'Could not save calibration: {e}'
                    }

        return {
            'success': False,
            'error': 'No audio recorded during calibration'
        }

    except Exception as e:
        print(f"   ❌ Calibration failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================
# VAD-BASED INTERRUPT MONITOR FOR WEB VERSION
# ============================================================

class VADInterruptMonitor:
    """Monitors incoming audio chunks for interrupts using Silero VAD with AEC"""

    def __init__(self, vad_config: VADConfig = None):
        self.config = vad_config or VADConfig()

        # Initialize VAD processor with interrupt detection enabled
        self.vad = VADProcessor(
            threshold=self.config.vad_threshold,
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
            interrupt_threshold=self.config.interrupt_threshold,
            interrupt_frame_duration=self.config.interrupt_frame_duration,
            interrupt_frames_needed=self.config.interrupt_frames_needed,
            verbose=self.config.verbose  # Pass verbose flag for detailed logging
        )

        self.interrupted = False
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=100)

        # WebRTC Acoustic Echo Cancellation (AEC)
        try:
            from webrtc_audio_processing import AudioProcessingModule as AP
            # aec_type: 0=disabled, 1=AEC, 2=AECM (mobile)
            # agc_type: 0=disabled, 1=adaptive analog, 2=adaptive digital, 3=fixed
            self.aec = AP(aec_type=1, enable_ns=True, agc_type=0, enable_vad=False)
            self.aec.set_stream_format(self.config.sample_rate, 1)  # 16kHz, mono
            self.aec.set_aec_level(1)  # Moderate suppression (0=low, 1=moderate, 2=high) - lowered for better interrupt detection
            self.aec.set_ns_level(1)   # Noise suppression level (0-3) - lowered to preserve user voice
            self.use_webrtc_aec = True
            if self.config.verbose:
                print("[AEC] Using WebRTC Audio Processing (AEC + Noise Suppression)")
        except ImportError:
            self.use_webrtc_aec = False
            if self.config.verbose:
                print("[AEC] WebRTC not available, AEC disabled")

        # For reference audio (TTS playback) - needed for WebRTC AEC
        self.playback_buffer = np.array([], dtype=np.float32)
        self.playback_position = 0

    def start(self):
        """Start monitoring for interrupts"""
        if self.monitoring:
            return

        self.interrupted = False
        self.monitoring = True
        print(f"[VAD] Interrupt monitor starting (threshold: {self.config.interrupt_threshold}, 67% rule, {self.config.interrupt_frame_duration}s frames, need {self.config.interrupt_frames_needed} frames)")
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        if self.monitoring:
            print("[VAD] Interrupt monitor stopping")
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def is_interrupted(self):
        """Check if interrupt detected"""
        with self.lock:
            return self.interrupted

    def reset(self):
        """Reset interrupt flag and VAD state"""
        with self.lock:
            self.interrupted = False

        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Reset VAD interrupt detection state
        self.vad.reset_interrupt_detection()

        # Reset AEC state
        self.playback_buffer = np.array([], dtype=np.float32)
        self.playback_position = 0

    def set_playback_audio(self, audio_data: np.ndarray):
        """Set the audio that's currently being played (for echo cancellation)

        Args:
            audio_data: Float32 numpy array of TTS audio being played
        """
        with self.lock:
            self.playback_buffer = audio_data.astype(np.float32)
            self.playback_position = 0
            if self.config.verbose:
                print(f"[AEC] Playback buffer set: {len(audio_data)} samples ({len(audio_data)/self.config.sample_rate:.2f}s)")

    def add_audio_chunk(self, audio_data):
        """Add audio chunk to queue for monitoring (non-blocking)"""
        try:
            # Non-blocking put - if queue is full, remove oldest and add new
            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass

    def _monitor_loop(self):
        """Background loop that checks audio chunks using VAD"""
        try:
            poll_count = 0
            chunk_count = 0
            while self.monitoring:
                try:
                    # Get audio chunk from queue (with timeout)
                    try:
                        audio_data = self.audio_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    # Convert int16 audio to float32 numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

                    # Apply Acoustic Echo Cancellation (AEC)
                    with self.lock:
                        if self.use_webrtc_aec and len(self.playback_buffer) > 0:
                            # WebRTC AEC: Process in 10ms frames (160 samples at 16kHz)
                            frame_size = 160
                            chunk_len = len(audio_np)
                            playback_end = min(self.playback_position + chunk_len, len(self.playback_buffer))
                            playback_chunk = self.playback_buffer[self.playback_position:playback_end]

                            # Pad if needed
                            if len(playback_chunk) < chunk_len:
                                playback_chunk = np.pad(playback_chunk, (0, chunk_len - len(playback_chunk)), mode='constant')

                            # Process frame by frame
                            cleaned_frames = []
                            for i in range(0, chunk_len, frame_size):
                                # Get 10ms frames
                                mic_frame = audio_np[i:i+frame_size]
                                ref_frame = playback_chunk[i:i+frame_size]

                                # Pad frames if needed
                                if len(mic_frame) < frame_size:
                                    mic_frame = np.pad(mic_frame, (0, frame_size - len(mic_frame)), mode='constant')
                                if len(ref_frame) < frame_size:
                                    ref_frame = np.pad(ref_frame, (0, frame_size - len(ref_frame)), mode='constant')

                                # Convert to int16 for WebRTC
                                mic_frame_int16 = mic_frame.astype(np.int16)
                                ref_frame_int16 = ref_frame.astype(np.int16)

                                # Feed reference signal (what's playing) - as bytes
                                self.aec.process_reverse_stream(ref_frame_int16.tobytes())

                                # Process mic signal (returns cleaned audio) - as bytes
                                cleaned_bytes = self.aec.process_stream(mic_frame_int16.tobytes())

                                # Convert back from bytes to numpy array
                                cleaned_frame = np.frombuffer(cleaned_bytes, dtype=np.int16).astype(np.float32)

                                cleaned_frames.append(cleaned_frame)

                            # Reassemble cleaned audio
                            audio_np = np.concatenate(cleaned_frames)[:chunk_len]

                            # Update position
                            self.playback_position += chunk_len

                            if self.config.verbose and chunk_count % 20 == 0:
                                echo_energy = np.sqrt(np.mean(playback_chunk ** 2))
                                mic_energy = np.sqrt(np.mean(audio_np ** 2))
                                print(f"  [WebRTC AEC] Pos: {self.playback_position}/{len(self.playback_buffer)} | "
                                      f"Echo: {echo_energy:.0f} | Cleaned: {mic_energy:.0f}")

                    # Silero VAD expects 512 samples at 16kHz
                    # WebSocket sends 1365 samples per chunk, so we need to split it
                    chunk_size = self.config.chunk_size  # 512 samples

                    for i in range(0, len(audio_np), chunk_size):
                        chunk = audio_np[i:i + chunk_size]

                        # Skip if chunk is too small (need exactly 512 samples)
                        if len(chunk) < chunk_size:
                            # Pad with zeros if needed
                            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                        # Use VAD interrupt detection on CLEANED audio (strict: threshold=0.95, 67% rule)
                        interrupt_confirmed, prob = self.vad.detect_interrupt(chunk)

                        chunk_count += 1

                        if interrupt_confirmed:
                            print(f"\n🚨 [VAD INTERRUPT] CONFIRMED! (VAD: {prob:.3f}, total chunks processed: {chunk_count})")
                            with self.lock:
                                self.interrupted = True
                            return  # Exit monitoring once interrupted

                        # Verbose logging - shows every chunk during interrupt monitoring
                        if self.config.verbose and chunk_count % 10 == 0:
                            stats = self.vad.get_stats()
                            print(f"  [INTERRUPT MONITOR] Chunk #{chunk_count} | VAD prob: {prob:.3f} | "
                                  f"Frames confirmed: {stats['interrupt_frames_confirmed']}/{self.config.interrupt_frames_needed} | "
                                  f"Buffer: {stats['interrupt_buffer_size']}/{int(self.config.sample_rate * self.config.interrupt_frame_duration / self.config.chunk_size)}")

                        poll_count += 1

                except Exception as e:
                    print(f"[VAD] Error during monitoring: {e}")
                    import traceback
                    traceback.print_exc()
                    pass

        except Exception as e:
            print(f"[VAD] Monitor thread error: {e}")
            import traceback
            traceback.print_exc()
            pass
        finally:
            self.monitoring = False
            print(f"[VAD] Monitor thread exiting (processed {chunk_count} chunks total)")


# ============================================================
# REAL-TIME TTS PIPELINE (Parallel Audio Generation & Playback)
# ============================================================

class AudioStreamManager:
    """Manages audio playback - sends to G1 robot speaker via HTTP."""

    def __init__(self):
        self.stream_lock = threading.Lock()
        self.is_playing = False
        self.playback_end_time = 0

        # G1 audio config load karo
        self.g1_enabled = False
        self.g1_url = None
        self.g1_gain = 1.0  # Audio amplification factor
        try:
            config_path = Path(__file__).parent / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            g1_config = config.get("g1_audio", {})
            if g1_config.get("enabled", False):
                host = g1_config.get("host", "192.168.123.164")
                port = g1_config.get("port", 5050)
                self.g1_url = f"http://{host}:{port}/play_audio"
                self.g1_gain = g1_config.get("gain", 3.0)  # Default 3x amplification
                self.g1_enabled = True
                print(f"  G1 Audio enabled: {self.g1_url} (gain: {self.g1_gain}x)")
        except Exception as e:
            print(f"  G1 Audio config error: {e}")

    def _stop_unlocked(self):
        """Stop stream (internal - call only with lock held)."""
        try:
            if self.is_playing:
                # G1 pe stop command bhejo (optional)
                if self.g1_enabled:
                    try:
                        requests.post(self.g1_url.replace("/play_audio", "/stop"), timeout=1)
                    except:
                        pass
                self.is_playing = False
        except Exception:
            self.is_playing = False

    def safe_stop(self):
        """Safely stop any active stream (thread-safe)."""
        with self.stream_lock:
            self._stop_unlocked()

    def safe_play(self, data, samplerate):
        """Send audio to G1 robot speaker via HTTP POST."""
        with self.stream_lock:
            try:
                # Ensure previous stream is stopped
                self._stop_unlocked()

                # Convert to 16kHz mono 16-bit PCM for G1
                import scipy.signal

                # Mono me convert karo agar stereo hai
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # 16kHz me resample karo
                if samplerate != 16000:
                    num_samples = int(len(data) * 16000 / samplerate)
                    data = scipy.signal.resample(data, num_samples)

                # Amplify audio (G1 speaker volume boost)
                data = data * self.g1_gain

                # 16-bit PCM me convert karo (clip after amplification)
                data = np.clip(data, -1.0, 1.0)
                pcm_data = (data * 32767).astype(np.int16)
                pcm_bytes = pcm_data.tobytes()

                # Calculate playback duration
                duration_sec = len(pcm_data) / 16000.0

                if self.g1_enabled:
                    # G1 ko HTTP POST se bhejo
                    try:
                        response = requests.post(
                            self.g1_url,
                            data=pcm_bytes,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=5
                        )
                        if response.status_code == 200:
                            self.is_playing = True
                            self.playback_end_time = time.time() + duration_sec
                            print(f"  Sent to G1: {len(pcm_bytes)} bytes, {duration_sec:.2f}s")
                            return True
                        else:
                            print(f"  G1 error: {response.status_code}")
                            return False
                    except Exception as e:
                        print(f"  G1 send error: {e}")
                        return False
                else:
                    # Fallback: local playback (disabled when G1 enabled)
                    sd.play(data, samplerate=16000)
                    self.is_playing = True
                    self.playback_end_time = time.time() + duration_sec
                    return True

            except Exception as e:
                print(f"  Stream play error: {e}")
                self.is_playing = False
                return False

    def is_active(self):
        """Check if audio is still playing (based on estimated duration)."""
        with self.stream_lock:
            if not self.is_playing:
                return False
            # Check if playback time has elapsed
            if time.time() >= self.playback_end_time:
                self.is_playing = False
                return False
            return True

    def wait_for_completion(self):
        """Wait for current playback to complete."""
        try:
            while self.is_active():
                time.sleep(0.05)
        except Exception:
            pass


class AudioPlayer:
    """Manages audio playback queue - plays chunks while TTS generates next ones"""

    def __init__(self, interrupt_monitor=None, completion_threshold=0.5):
        self.queue = queue.Queue()
        self.stream_manager = AudioStreamManager()
        self.state_lock = threading.Lock()  # Thread safety for state flags
        self.playing = False
        self._stop_flag = False
        self._interrupted = False
        self.player_thread = None
        self.interrupt_monitor = interrupt_monitor
        self.chunks_played = 0
        self.tts_generator = None  # Will be set after TTSGenerator is created
        self._playback_started_published = False  # Track if we published True
        self._playback_ending_published = False  # Track if we published False
        self.completion_threshold = completion_threshold  # Threshold (0-1) for publishing False on last chunk

    @property
    def stop_flag(self):
        with self.state_lock:
            return self._stop_flag

    @stop_flag.setter
    def stop_flag(self, value):
        with self.state_lock:
            self._stop_flag = value

    @property
    def interrupted(self):
        with self.state_lock:
            return self._interrupted

    @interrupted.setter
    def interrupted(self, value):
        with self.state_lock:
            self._interrupted = value

    def start(self):
        """Start the audio player thread."""
        self.stop_flag = False
        self.interrupted = False
        self.chunks_played = 0
        self._playback_started_published = False
        self._playback_ending_published = False
        self.player_thread = threading.Thread(target=self._player_loop, daemon=True)
        self.player_thread.start()

    def _publish_playback_status(self, is_playing: bool):
        """Update playback status (published continuously to ROS2 topic)"""
        ros2_pub = get_ros2_publisher()
        if ros2_pub:
            ros2_pub.set_playing(is_playing)
        else:
            print(f"📢 Playback status: {'STARTED' if is_playing else 'ENDING'} (ROS2 not available)")

    def _is_last_chunk(self):
        """Check if current chunk is the last one (queue empty and TTS done)"""
        if self.tts_generator is None:
            return self.queue.empty()
        return self.queue.empty() and self.tts_generator.is_done()

    def stop(self):
        """Stop the audio player."""
        self.stop_flag = True
        self.stream_manager.safe_stop()  # Also stop any playing audio
        self.queue.put(None)  # Signal to stop
        if self.player_thread:
            self.player_thread.join(timeout=2)

    def add(self, audio_path, chunk_num):
        """Add audio file to playback queue."""
        self.queue.put((audio_path, chunk_num))

    def is_done(self):
        """Check if playback is complete."""
        return self.queue.empty() and not self.playing

    def _player_loop(self):
        """Main playback loop - runs in separate thread."""
        while not self.stop_flag and not self.interrupted:
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:
                    break

                audio_path, chunk_num = item

                self.playing = True
                interrupted = self._play_audio(audio_path, chunk_num)
                self.playing = False

                if interrupted:
                    self.interrupted = True
                    # Clear remaining queue
                    while not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except queue.Empty:
                            break
                    break

            except queue.Empty:
                continue

    def _play_audio(self, audio_path, chunk_num):
        """Play a single audio file with interrupt detection."""
        if not os.path.exists(audio_path):
            return False

        try:
            # Load audio
            data, sr_rate = sf.read(audio_path)

            # Calculate audio duration for progress tracking
            audio_duration = len(data) / sr_rate

            # Set playback audio for echo cancellation
            if self.interrupt_monitor is not None:
                if sr_rate != 16000:
                    import scipy.signal
                    data_16k = scipy.signal.resample(data, int(len(data) * 16000 / sr_rate))
                else:
                    data_16k = data

                if len(data_16k.shape) > 1:
                    data_16k = np.mean(data_16k, axis=1)

                self.interrupt_monitor.set_playback_audio(data_16k.astype(np.float32))

            # Use stream manager for safe playback
            print(f"  🔊 Playing chunk {chunk_num}...")
            if not self.stream_manager.safe_play(data, sr_rate):
                return False

            # Publish playback STARTED on first chunk
            if not self._playback_started_published:
                self._playback_started_published = True
                self._publish_playback_status(True)

            # Check if this is the last chunk (for publishing ENDING status)
            is_last = self._is_last_chunk()

            # Wait for playback with interrupt checking
            interrupted = False
            playback_start_time = time.time()

            try:
                while self.stream_manager.is_active():
                    if self.stop_flag:
                        self.stream_manager.safe_stop()
                        break

                    # Check for interrupt
                    if self.interrupt_monitor and self.interrupt_monitor.is_interrupted():
                        print(f"  🛑 Interrupt detected during chunk {chunk_num}")
                        self.stream_manager.safe_stop()
                        interrupted = True
                        # Immediately publish False on interrupt
                        if not self._playback_ending_published:
                            self._playback_ending_published = True
                            self._publish_playback_status(False)
                        break

                    # If last chunk, publish ENDING at threshold progress
                    if is_last and not self._playback_ending_published:
                        elapsed = time.time() - playback_start_time
                        progress = elapsed / audio_duration if audio_duration > 0 else 1.0
                        if progress >= self.completion_threshold:
                            self._playback_ending_published = True
                            self._publish_playback_status(False)

                    time.sleep(0.05)

                # Final stop if still playing
                if self.stream_manager.is_active():
                    self.stream_manager.safe_stop()

                # Fallback: if last chunk finished naturally and False wasn't published yet, publish now
                if is_last and not self._playback_ending_published:
                    self._playback_ending_published = True
                    self._publish_playback_status(False)

            except Exception as e:
                print(f"  ⚠️ Error during playback: {e}")
                self.stream_manager.safe_stop()

            self.chunks_played += 1

            if not interrupted:
                print(f"  ✓ Chunk {chunk_num} done")

            return interrupted

        except Exception as e:
            print(f"  ✗ Playback error: {e}")
            return False
        finally:
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass


class TTSGenerator:
    """Manages TTS generation in background thread."""

    def __init__(self, audio_player, backend="edge", voice="en-US-AriaNeural", rate=150, volume=1.0, host=None, port=None, seed=42, cfg_scale=1.3):
        self.audio_player = audio_player
        self.backend = backend
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.host = host  # For VibeVoice
        self.port = port  # For VibeVoice
        self.seed = seed  # For VibeVoice
        self.cfg_scale = cfg_scale  # For VibeVoice
        self.task_queue = queue.Queue()
        self.state_lock = threading.Lock()  # Thread safety
        self._stop_flag = False
        self.thread = None
        self._chunk_count = 0
        self.generation_done = False

    @property
    def stop_flag(self):
        with self.state_lock:
            return self._stop_flag

    @stop_flag.setter
    def stop_flag(self, value):
        with self.state_lock:
            self._stop_flag = value

    @property
    def chunk_count(self):
        with self.state_lock:
            return self._chunk_count

    @chunk_count.setter
    def chunk_count(self, value):
        with self.state_lock:
            self._chunk_count = value

    def start(self):
        """Start the TTS generator thread."""
        self.stop_flag = False
        self.chunk_count = 0
        self.generation_done = False
        self.thread = threading.Thread(target=self._generator_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the TTS generator."""
        self.stop_flag = True
        self.task_queue.put(None)
        if self.thread:
            self.thread.join(timeout=5)

    def add_text(self, text):
        """Add text to be converted to speech (thread-safe)."""
        with self.state_lock:
            self._chunk_count += 1
            chunk_num = self._chunk_count
        self.task_queue.put((text, chunk_num))

    def mark_done(self):
        """Signal that no more text will be added."""
        self.task_queue.put(None)

    def is_done(self):
        """Check if all TTS generation is complete."""
        return self.generation_done and self.task_queue.empty()

    def _generator_loop(self):
        """Main TTS generation loop."""
        while not self.stop_flag:
            try:
                item = self.task_queue.get(timeout=0.5)
                if item is None:
                    self.generation_done = True
                    break

                text, chunk_num = item

                # Skip if audio player was interrupted
                if self.audio_player.interrupted:
                    print(f"  ⏭ Skipping TTS chunk {chunk_num} (interrupted)")
                    continue

                # Generate TTS
                # Use .wav for VibeVoice and Chatterbox, .mp3 for others
                suffix = '.wav' if self.backend in ('vibevoice', 'chatterbox') else '.mp3'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_path = temp_file.name
                temp_file.close()

                print(f"  🎤 TTS chunk {chunk_num} ({len(text)} chars)...")
                start = time.time()

                try:
                    asyncio.run(self._generate_tts(text, temp_path))
                    gen_time = time.time() - start
                    print(f"  ✓ TTS chunk {chunk_num} ready ({gen_time:.2f}s)")

                    # Queue for playback
                    self.audio_player.add(temp_path, chunk_num)
                except Exception as e:
                    print(f"  ✗ TTS error chunk {chunk_num}: {e}")
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

            except queue.Empty:
                continue

        self.generation_done = True

    async def _generate_tts(self, text, output_path):
        """Generate TTS audio file using configured backend with fallback."""
        if self.backend == 'pyttsx3':
            # pyttsx3 generates its own file, so we generate and move it
            temp_path = await generate_tts_chunk_pyttsx3(text, self.rate, self.volume)
            if temp_path and temp_path != output_path:
                os.rename(temp_path, output_path)
        elif self.backend == 'edge':
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
        elif self.backend == 'vibevoice':
            # Use VibeVoice local server
            temp_path = await generate_tts_chunk_vibevoice(
                text, self.host, self.port, self.voice, self.seed, self.cfg_scale
            )
            if temp_path and temp_path != output_path:
                os.rename(temp_path, output_path)
        elif self.backend == 'chatterbox':
            # Use Chatterbox TTS server with Edge TTS fallback
            try:
                temp_path = await generate_tts_chunk_chatterbox(
                    text, self.host, self.port
                )
                if temp_path and temp_path != output_path:
                    os.rename(temp_path, output_path)
            except Exception as e:
                print(f"  ⚠️ Chatterbox failed ({e}), falling back to Edge TTS...")
                # Fallback to Edge TTS
                # Output path might be .wav, need .mp3 for edge
                edge_output = output_path.replace('.wav', '.mp3')
                fallback_voice = self.voice or "en-IN-NeerjaExpressiveNeural"
                communicate = edge_tts.Communicate(text, fallback_voice)
                await communicate.save(edge_output)
                # Rename back if needed
                if edge_output != output_path:
                    os.rename(edge_output, output_path)
        else:
            raise ValueError(f"Unknown TTS backend: {self.backend}")


# ============================================================
# TEXT CLEANING AND TTS HELPERS
# ============================================================

def clean_text_for_tts(text):
    """Remove formatting symbols and special characters that don't make sense when spoken"""
    # Remove text in parentheses
    text = re.sub(r'\([^)]*\)', '', text)

    # Replace "Dr" with "Doctor"
    text = re.sub(r'\bDr\b', 'Doctor', text)

    # Remove markdown bold/italic/strikethrough
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__ -> bold
    text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_ -> italic
    text = re.sub(r'~~(.+?)~~', r'\1', text)      # ~~strike~~ -> strike

    # Remove markdown links but keep text
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # [text](url) -> text

    # Remove bullet points and list markers
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove multiple asterisks, underscores, dashes at line start
    text = re.sub(r'^\s*[\*\-_]{2,}\s*$', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove lines that are only symbols
    lines = [line.strip() for line in text.split('\n') if line.strip() and not re.match(r'^[\*\-_=]+$', line.strip())]
    text = '\n'.join(lines)

    return text.strip()


def split_text_into_chunks(text, chunk_size_first=50, chunk_size_rest=150):
    """Split text into chunks: 50 chars for first chunk, 150 chars for rest"""
    sentences = re.split(r'([।.!?]+)', text)

    chunks = []
    current_chunk = ""
    is_first_chunk = True
    max_chars = chunk_size_first  # Start with smaller size for first chunk

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        ending = sentences[i+1] if i+1 < len(sentences) else ""
        full_sentence = sentence + ending

        if len(current_chunk) + len(full_sentence) <= max_chars:
            current_chunk += full_sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Switch to larger chunk size after first chunk
                if is_first_chunk:
                    is_first_chunk = False
                    max_chars = chunk_size_rest

            if len(full_sentence) > max_chars:
                # Split by words if sentence is too long
                words = full_sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chars:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                            # Switch to larger chunk size after first chunk
                            if is_first_chunk:
                                is_first_chunk = False
                                max_chars = chunk_size_rest
                        temp_chunk = word + " "
                current_chunk = temp_chunk
            else:
                current_chunk = full_sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def _generate_pyttsx3_sync(text, rate, volume):
    """Synchronous helper for pyttsx3 (must run in separate thread)"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()

        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        return temp_path
    except Exception as e:
        print(f"❌ pyttsx3 sync error: {e}")
        return None


async def generate_tts_chunk_pyttsx3(text, rate=150, volume=1.0):
    """Generate TTS audio using pyttsx3 (offline)"""
    if not PYTTSX3_AVAILABLE:
        print("❌ pyttsx3 not installed. Install with: pip install pyttsx3")
        return None

    try:
        loop = asyncio.get_event_loop()
        temp_path = await loop.run_in_executor(None, _generate_pyttsx3_sync, text, rate, volume)
        return temp_path
    except Exception as e:
        print(f"❌ pyttsx3 error: {e}")
        return None


async def generate_tts_chunk_edge(text, voice):
    """Generate TTS audio using Edge TTS (online)"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_path)

        return temp_path
    except Exception as e:
        print(f"❌ Edge TTS error: {e}")
        return None


async def generate_tts_chunk_vibevoice(text, host, port, voice, seed=42, cfg_scale=1.3):
    """Generate TTS audio using VibeVoice local server"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()

        # Make POST request to VibeVoice server
        url = f"http://{host}:{port}/tts"
        payload = {
            "text": text,
            "voice": voice,
            "seed": seed,
            "cfg_scale": cfg_scale
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=payload, timeout=30)
        )

        if response.status_code == 200:
            # Save the WAV audio
            with open(temp_path, 'wb') as f:
                f.write(response.content)

            # Log metadata from headers
            gen_time = response.headers.get('X-Generation-Time', 'N/A')
            audio_dur = response.headers.get('X-Audio-Duration', 'N/A')
            rtf = response.headers.get('X-RTF', 'N/A')
            print(f"  ℹ️  VibeVoice: gen={gen_time}s, dur={audio_dur}s, RTF={rtf}")

            return temp_path
        else:
            print(f"❌ VibeVoice server error: {response.status_code} - {response.text}")
            try:
                os.unlink(temp_path)
            except:
                pass
            return None

    except requests.exceptions.ConnectionError:
        print(f"❌ VibeVoice connection error: Cannot connect to {host}:{port}")
        print(f"   Make sure VibeVoice server is running: docker-compose up -d")
        try:
            os.unlink(temp_path)
        except:
            pass
        return None
    except Exception as e:
        print(f"❌ VibeVoice error: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return None


async def generate_tts_chunk_chatterbox(text, host, port):
    """Generate TTS audio using Chatterbox TTS server"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()

        # Make GET request to Chatterbox server
        url = f"http://{host}:{port}/tts"
        params = {"text": text}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, params=params, timeout=120)
        )

        if response.status_code == 200:
            # Save the WAV audio
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            return temp_path
        else:
            print(f"❌ Chatterbox server error: {response.status_code} - {response.text}")
            try:
                os.unlink(temp_path)
            except:
                pass
            return None

    except requests.exceptions.ConnectionError:
        print(f"❌ Chatterbox connection error: Cannot connect to {host}:{port}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return None
    except Exception as e:
        print(f"❌ Chatterbox error: {e}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return None


async def generate_tts_chunk(text, backend, voice=None, rate=150, volume=1.0, host=None, port=None, seed=42, cfg_scale=1.3):
    """Generate TTS audio for a single chunk using configured backend"""
    if backend == 'pyttsx3':
        return await generate_tts_chunk_pyttsx3(text, rate, volume)
    elif backend == 'edge':
        return await generate_tts_chunk_edge(text, voice)
    elif backend == 'vibevoice':
        if host is None or port is None:
            print(f"❌ VibeVoice backend requires host and port")
            return None
        return await generate_tts_chunk_vibevoice(text, host, port, voice, seed, cfg_scale)
    elif backend == 'chatterbox':
        if host is None or port is None:
            print(f"❌ Chatterbox backend requires host and port")
            return None
        return await generate_tts_chunk_chatterbox(text, host, port)
    else:
        print(f"❌ Unknown TTS backend: {backend}")
        return None


def play_audio(audio_path, check_interrupt_callback=None, interrupt_monitor=None):
    """Play audio on SERVER speakers and check for interrupts"""
    if not os.path.exists(audio_path):
        return False

    try:
        # Load and play audio
        data, sr_rate = sf.read(audio_path)

        # Set playback audio in interrupt monitor for echo cancellation
        if interrupt_monitor is not None:
            # Resample to 16kHz if needed (for AEC)
            if sr_rate != 16000:
                import scipy.signal
                data_16k = scipy.signal.resample(data, int(len(data) * 16000 / sr_rate))
            else:
                data_16k = data

            # Convert to mono if stereo
            if len(data_16k.shape) > 1:
                data_16k = np.mean(data_16k, axis=1)

            interrupt_monitor.set_playback_audio(data_16k.astype(np.float32))

        sd.play(data, samplerate=sr_rate)

        # Wait for playback to complete while checking interrupt flag
        interrupted = False
        poll_count = 0
        try:
            while sd.get_stream().active:
                # Check if user started speaking (interrupt)
                if check_interrupt_callback and check_interrupt_callback():
                    print(f"    🛑 Interrupt callback returned True - STOPPING playback")
                    sd.stop()
                    interrupted = True
                    break
                poll_count += 1
                # Log every 10 polls (every 0.5 seconds)
                if poll_count % 10 == 0:
                    print(f"    ⏸  Playback active (checking interrupts every 50ms, poll #{poll_count})")
                time.sleep(0.05)  # Poll every 50ms
        except Exception as e:
            print(f"    ⚠️  Exception during playback: {e}")
            pass

        # Stop audio playback
        sd.stop()
        time.sleep(0.15)  # Let audio resources settle

        return interrupted
    except Exception as e:
        print(f"❌ Playback error: {e}")
        return False
    finally:
        # Cleanup - ensure audio is fully stopped
        try:
            sd.stop()
            time.sleep(0.1)
        except:
            pass

        # Delete temp file
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass


# ============================================================
# WEB STREAMING ASSISTANT
# ============================================================

class WebStreamingAssistant:
    """Handles web-based audio streaming and processing"""

    def __init__(self):
        self.audio_buffer = []
        self.is_processing = False
        self.recognizer = sr.Recognizer()
        self.state = "idle"  # idle, listening, transcribing, speaking

        # VAD configuration and processor
        self.vad_config = VADConfig(
            sample_rate=16000,
            chunk_size=512,
            vad_threshold=0.5,  # Regular speech detection
            silence_timeout=1.5,
            interrupt_threshold=0.7,  # Interrupt detection (lowered from 0.98 for better sensitivity)
            interrupt_frame_duration=0.3,  # Shorter frame duration (lowered from 0.5s)
            interrupt_frames_needed=2,  # Need 2 consecutive frames (increased for stability)
            verbose=True  # Enable verbose logging for diagnosis
        )

        # Initialize VAD processor for speech detection
        self.vad = VADProcessor(
            threshold=self.vad_config.vad_threshold,
            sample_rate=self.vad_config.sample_rate,
            chunk_size=self.vad_config.chunk_size,
            verbose=self.vad_config.verbose  # Pass verbose flag
        )

        # VAD state tracking
        self.speech_started = False
        self.silence_duration = 0.0
        self.silence_threshold = self.vad_config.silence_timeout
        self.frame_duration = self.vad_config.chunk_size / self.vad_config.sample_rate  # 32ms
        self.frame_count = 0

        # Load configuration
        self.config = load_config()

        # LLM settings with deployment mode support
        self.llm_model = self.config['llm']['model']
        self.llm_temperature = self.config['llm'].get('options', {}).get('temperature', 0.2)

        # Determine deployment mode (local or remote)
        deployment_mode = self.config['llm'].get('deployment_mode', 'local')

        if deployment_mode == 'local':
            self.llm_host = self.config['llm']['local']['host']
            self.llm_port = self.config['llm']['local']['port']
            self.deployment_mode = '🏠 LOCAL'
        elif deployment_mode == 'remote':
            self.llm_host = self.config['llm']['remote']['host']
            self.llm_port = self.config['llm']['remote']['port']
            self.deployment_mode = '🌐 REMOTE'
        else:
            # Fallback to old format (host/port at root level)
            self.llm_host = self.config['llm'].get('host', '127.0.0.1')
            self.llm_port = self.config['llm'].get('port', 11434)
            self.deployment_mode = '⚠️  LEGACY'

        self.ollama_url = f"http://{self.llm_host}:{self.llm_port}/api/chat"

        # TTS settings
        self.tts_backend = self.config['tts']['backend']

        # Load backend-specific settings
        if self.tts_backend == 'edge':
            self.tts_voice = self.config['tts']['edge']['voice']
            self.tts_rate = 150  # Not used for edge
            self.tts_volume = 1.0  # Not used for edge
            self.tts_host = None
            self.tts_port = None
            self.tts_seed = 42
            self.tts_cfg_scale = 1.3
        elif self.tts_backend == 'pyttsx3':
            self.tts_voice = None  # Not used for pyttsx3
            self.tts_rate = self.config['tts']['pyttsx3'].get('rate', 150)
            self.tts_volume = self.config['tts']['pyttsx3'].get('volume', 1.0)
            self.tts_host = None
            self.tts_port = None
            self.tts_seed = 42
            self.tts_cfg_scale = 1.3
        elif self.tts_backend == 'vibevoice':
            self.tts_voice = self.config['tts']['vibevoice'].get('voice', 'hi-Aish_woman')
            self.tts_host = self.config['tts']['vibevoice'].get('host', 'localhost')
            self.tts_port = self.config['tts']['vibevoice'].get('port', 8787)
            self.tts_seed = self.config['tts']['vibevoice'].get('seed', 42)
            self.tts_cfg_scale = self.config['tts']['vibevoice'].get('cfg_scale', 1.3)
            self.tts_rate = 150  # Not used for vibevoice
            self.tts_volume = 1.0  # Not used for vibevoice
        elif self.tts_backend == 'chatterbox':
            self.tts_voice = self.config['tts']['chatterbox'].get('fallback_voice', 'en-IN-NeerjaExpressiveNeural')  # Edge TTS fallback voice
            self.tts_host = self.config['tts']['chatterbox'].get('host', '172.16.6.19')
            self.tts_port = self.config['tts']['chatterbox'].get('port', 8000)
            self.tts_rate = 150  # Not used for chatterbox
            self.tts_volume = 1.0  # Not used for chatterbox
            self.tts_seed = 42  # Not used for chatterbox
            self.tts_cfg_scale = 1.3  # Not used for chatterbox
        else:
            self.tts_voice = None
            self.tts_rate = 150
            self.tts_volume = 1.0
            self.tts_host = None
            self.tts_port = None
            self.tts_seed = 42
            self.tts_cfg_scale = 1.3

        # Conversation history
        self.messages = []
        self.max_messages = self.config['assistant']['max_messages']

        # VAD-based interrupt detection
        self.interrupt_monitor = VADInterruptMonitor(vad_config=self.vad_config)

        # Active playback components (for cleanup on disconnect)
        self.active_audio_player = None
        self.active_tts_generator = None

        # Chunk sizes for TTS
        self.chunk_size_first = self.config['assistant']['chunk_size_first']
        self.chunk_size_rest = self.config['assistant']['chunk_size_rest']

        # Initialize RAG Agent (if available)
        self.use_rag = self.config.get('llm', {}).get('use_rag', True)
        self.rag_agent = None
        if self.use_rag and RAG_AVAILABLE:
            try:
                print("🧠 Initializing RAG Agent...")
                self.rag_agent = InduAgent(
                    knowledge_base_path=str(Path(__file__).parent / "indu_knowledge_base.txt"),
                    system_prompt_path=str(Path(__file__).parent / "indu_system_prompt.txt"),
                    model_name=self.llm_model,
                    vector_store_path=str(Path(__file__).parent / "indu_vectorstore"),
                    ollama_host=self.llm_host,
                    ollama_port=self.llm_port,
                    temperature=self.llm_temperature,
                    debug_mode=False
                )
                print("✅ RAG Agent initialized")
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
                self.rag_agent = None

        print(f"🤖 WebStreamingAssistant initialized:")
        print(f"   LLM: {self.llm_model} @ {self.llm_host}:{self.llm_port} ({self.deployment_mode})")
        print(f"   RAG: {'Enabled' if self.rag_agent else 'Disabled'}")

        # Print TTS info based on backend
        if self.tts_backend == 'edge':
            print(f"   TTS: {self.tts_backend} (voice: {self.tts_voice})")
        elif self.tts_backend == 'pyttsx3':
            print(f"   TTS: {self.tts_backend} (rate: {self.tts_rate}, volume: {self.tts_volume})")
        elif self.tts_backend == 'vibevoice':
            print(f"   TTS: {self.tts_backend} (voice: {self.tts_voice}, server: {self.tts_host}:{self.tts_port})")
        elif self.tts_backend == 'chatterbox':
            print(f"   TTS: {self.tts_backend} (server: {self.tts_host}:{self.tts_port})")
        else:
            print(f"   TTS: {self.tts_backend}")

        print(f"   VAD threshold: {self.vad_config.vad_threshold} (regular speech)")
        print(f"   Interrupt threshold: {self.vad_config.interrupt_threshold} (67% rule, 0.5s frames)")
        print(f"   Silence timeout: {self.silence_threshold}s")
        print(f"   TTS chunks: {self.chunk_size_first} (first), {self.chunk_size_rest} (rest)")
        print(f"   Interrupt monitor: VADInterruptMonitor initialized")

    def cleanup(self):
        """Clean up on client disconnect - stop all active playback and reset state"""
        print("🧹 Cleaning up on disconnect...")

        # Stop active audio player
        if self.active_audio_player:
            try:
                self.active_audio_player.stop()
                print("   ✓ Audio player stopped")
            except Exception as e:
                print(f"   ⚠️ Error stopping audio player: {e}")
            self.active_audio_player = None

        # Stop active TTS generator
        if self.active_tts_generator:
            try:
                self.active_tts_generator.stop()
                print("   ✓ TTS generator stopped")
            except Exception as e:
                print(f"   ⚠️ Error stopping TTS generator: {e}")
            self.active_tts_generator = None

        # Stop interrupt monitor
        if self.interrupt_monitor:
            try:
                self.interrupt_monitor.stop()
                print("   ✓ Interrupt monitor stopped")
            except Exception as e:
                print(f"   ⚠️ Error stopping interrupt monitor: {e}")

        # Reset state
        self.state = "idle"
        self.audio_buffer = []
        self.is_processing = False
        self.speech_started = False
        self.silence_duration = 0.0

        # Publish False to ROS2 (playback ended due to disconnect)
        ros2_pub = get_ros2_publisher()
        if ros2_pub:
            ros2_pub.set_playing(False)
            print("   ✓ ROS2 playback status set to False")

        print("🧹 Cleanup complete")

    def process_audio_chunk(self, audio_data, ws):
        """Process audio chunk with VAD-based speech detection"""
        try:
            self.frame_count += 1

            # Update state to listening if idle
            if self.state == "idle":
                self.state = "listening"
                self._send_state(ws, "listening")
                print("🎧 State → LISTENING (VAD-based speech detection active)")

            # INTERRUPT DETECTION: If AI is speaking/transcribing, add audio to monitor queue
            if self.state in ["speaking", "transcribing"]:
                # Add audio chunk to background monitor queue (VAD interrupt detection)
                self.interrupt_monitor.add_audio_chunk(audio_data)

                # Log audio chunks received during speaking (every 20 frames)
                if self.frame_count % 20 == 0:
                    print(f"    📡 Audio chunk queued for VAD interrupt monitoring during {self.state}")

                # Don't process audio further during speaking/transcribing
                return

            # Convert audio chunk to numpy array for VAD
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # WebSocket sends 1365 samples per chunk, but VAD needs 512 samples
            # Process in 512-sample chunks
            chunk_size = self.vad_config.chunk_size  # 512 samples

            # Track VAD probabilities across sub-chunks for verbose output
            chunk_vad_probs = []

            for i in range(0, len(audio_np), chunk_size):
                chunk = audio_np[i:i + chunk_size]

                # Skip if chunk is too small
                if len(chunk) < chunk_size:
                    # Pad with zeros to reach 512 samples
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                # Use VAD to detect speech
                is_speech, vad_prob = self.vad.is_speech(chunk)
                chunk_vad_probs.append(vad_prob)

                # Verbose logging for each 512-sample chunk
                if self.vad_config.verbose and self.frame_count % 5 == 0 and i == 0:
                    print(f"  [VAD CHUNK] Frame #{self.frame_count} | Chunk {i//chunk_size + 1}/{len(audio_np)//chunk_size + 1} | "
                          f"VAD prob: {vad_prob:.3f} | is_speech: {is_speech} | "
                          f"speech_started: {self.speech_started} | silence: {self.silence_duration:.1f}s")

                if is_speech:
                    # SPEECH DETECTED!
                    self.silence_duration = 0  # Reset silence counter

                    if not self.speech_started:
                        # Speech just STARTED - begin recording NOW
                        self.speech_started = True
                        self.audio_buffer = []  # Clear any old buffer
                        avg_prob = sum(chunk_vad_probs) / len(chunk_vad_probs)
                        print(f"🎤 Speech STARTED! (VAD avg: {avg_prob:.3f}, max: {max(chunk_vad_probs):.3f}) - Recording...")

                    # Add original chunk to buffer (we buffer the WebSocket chunks, not 512-sample chunks)
                    if i == 0:  # Only add once per WebSocket chunk
                        self.audio_buffer.append(audio_data)

                else:
                    # SILENCE detected
                    if self.speech_started:
                        # We were recording, now silence
                        self.silence_duration += self.frame_duration

                        # Still add frames during silence (to capture end of words)
                        if i == 0:  # Only add once per WebSocket chunk
                            self.audio_buffer.append(audio_data)

                        # Log silence progress every 0.5s
                        if int(self.silence_duration * 10) % 5 == 0 and i == 0:
                            avg_prob = sum(chunk_vad_probs) / len(chunk_vad_probs)
                            print(f"🔇 Silence: {self.silence_duration:.1f}s (need {self.silence_threshold}s, "
                                  f"VAD avg: {avg_prob:.3f}, min: {min(chunk_vad_probs):.3f})")

                        # Check if silence threshold reached (1.5s of silence)
                        if self.silence_duration >= self.silence_threshold and not self.is_processing:
                            avg_prob = sum(chunk_vad_probs) / len(chunk_vad_probs)
                            print(f"✅ Speech ENDED! (silence={self.silence_duration:.1f}s, final VAD: {avg_prob:.3f}) - Sending to STT...")
                            print(f"   📊 Buffer size: {len(self.audio_buffer)} chunks = {len(self.audio_buffer) * 1365 / 16000:.2f}s of audio")

                            # Copy buffer before resetting (avoid race condition)
                            audio_buffer_copy = self.audio_buffer.copy()

                            # Reset VAD state for next utterance BEFORE starting thread
                            self.speech_started = False
                            self.silence_duration = 0
                            self.audio_buffer = []

                            # Process in a separate thread so WebSocket can continue receiving audio
                            processing_thread = threading.Thread(
                                target=self._process_buffered_audio_thread,
                                args=(ws, audio_buffer_copy),
                                daemon=True
                            )
                            processing_thread.start()
                            return  # Exit early after starting processing

        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
            self._send_error(ws, str(e))

    def _process_buffered_audio_thread(self, ws, audio_buffer):
        """Thread wrapper for processing buffered audio"""
        if self.is_processing or len(audio_buffer) == 0:
            return

        self.is_processing = True

        try:
            # Update state to transcribing
            self.state = "transcribing"
            self._send_state(ws, "transcribing")
            print("📝 State → TRANSCRIBING")

            # Convert buffer to AudioData for transcription
            audio_bytes = b''.join(audio_buffer)
            audio_size = len(audio_bytes)

            # Create AudioData object (16kHz, 16-bit mono)
            audio_data = sr.AudioData(audio_bytes, 16000, 2)

            # Get STT backend from config
            stt_backend = self.config.get('stt', {}).get('backend', 'google')
            print(f"🎯 Sending {audio_size} bytes to {stt_backend.upper()} STT...")

            # Transcribe using configured backend
            try:
                if stt_backend == 'whisper_server':
                    transcript = transcribe_with_whisper_server(audio_data, self.config)
                    if transcript is None:
                        raise sr.UnknownValueError("Whisper server returned no result")
                else:
                    # Default to Google
                    transcript = self.recognizer.recognize_google(audio_data)
                print(f"✅ Transcription: '{transcript}'")

                # Send user transcript immediately to web interface
                self._send_transcript(ws, f"You: {transcript}", replace=True)

                # Update state to speaking (will start streaming LLM + TTS)
                self.state = "speaking"
                self._send_state(ws, "speaking")
                print("🗣️ State → SPEAKING (streaming pipeline)")

                # Small delay to ensure state update is sent
                time.sleep(0.1)

                # Start background interrupt monitor
                self.interrupt_monitor.reset()
                self.interrupt_monitor.start()

                try:
                    # Use new streaming pipeline: LLM → TTS → Audio (all parallel)
                    llm_response, was_interrupted = self._stream_llm_and_speak(transcript, ws)

                    if llm_response:
                        # Send final transcript update (use same prefixes as streaming: You/INDU)
                        self._send_transcript(ws, f"You: {transcript}\n\nINDU: {llm_response}", replace=True)

                        # Save conversation to JSON file
                        self._save_conversation(transcript, llm_response)

                        if was_interrupted:
                            print("🎤 User interrupted - ready for new input")

                except Exception as e:
                    print(f"❌ Streaming pipeline failed: {e}")
                    import traceback
                    traceback.print_exc()

                    # Fallback: try old method
                    try:
                        llm_response = ""
                        for chunk in self._call_ollama(transcript):
                            llm_response += chunk
                        self._send_transcript(ws, f"You: {transcript}\n\nINDU: {llm_response}", replace=True)
                        self._save_conversation(transcript, llm_response)
                        self._generate_and_play_tts_chunked(llm_response, ws)
                    except Exception as fallback_e:
                        print(f"❌ Fallback also failed: {fallback_e}")
                        self._send_transcript(ws, f"\n\nAI: I'm having trouble responding right now.")

                finally:
                    # Stop interrupt monitor
                    self.interrupt_monitor.stop()

                # Back to listening state
                self.state = "listening"
                self._send_state(ws, "listening")
                print("🎧 State → LISTENING (ready for next utterance)")

            except sr.UnknownValueError:
                self._send_error(ws, "Could not understand audio")
                self.state = "listening"
                self._send_state(ws, "listening")
            except sr.RequestError as e:
                self._send_error(ws, f"STT API error: {e}")
                self.state = "listening"
                self._send_state(ws, "listening")

        except Exception as e:
            print(f"❌ Error in audio processing pipeline: {e}")
            self._send_error(ws, str(e))
            self.state = "listening"
            self._send_state(ws, "listening")
        finally:
            self.is_processing = False

    def _send_state(self, ws, state):
        """Send state update to client"""
        try:
            ws.send(json.dumps({
                'type': 'state',
                'state': state
            }))
        except Exception as e:
            print(f"⚠️  Error sending state: {e}")

    def _send_transcript(self, ws, text, replace=False):
        """Send transcript update to client"""
        try:
            ws.send(json.dumps({
                'type': 'transcript',
                'text': text,
                'replace': replace  # If True, replace entire transcript; if False, append
            }))
        except Exception as e:
            print(f"⚠️  Error sending transcript: {e}")

    def _query_rag(self, user_message):
        """Query RAG agent (synchronous, returns full response)"""
        if not self.rag_agent:
            return None

        try:
            print(f"🧠 Querying RAG Agent...")
            response = self.rag_agent.chat(user_message)
            print(f"✅ RAG response: {response[:100]}..." if len(response) > 100 else f"✅ RAG response: {response}")
            return response
        except Exception as e:
            print(f"❌ RAG query failed: {e}")
            return None

    def _call_ollama(self, user_message):
        """Stream response from Ollama LLM"""
        try:
            # Add user message to history
            self.messages.append({
                "role": "user",
                "content": user_message
            })

            # Keep only last N messages
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

            payload = {
                "model": self.llm_model,
                "messages": self.messages,
                "stream": True,
                "options": {
                    "temperature": self.llm_temperature
                }
            }

            print(f"🧠 Querying LLM: {self.llm_model} @ {self.ollama_url} (temp={self.llm_temperature})")
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=300)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue

            # Add assistant response to history
            self.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            raise
        except Exception as e:
            print(f"❌ LLM error: {e}")
            raise

    def _check_interrupted(self):
        """Check if interrupt flag is set (from background monitor)"""
        interrupted = self.interrupt_monitor.is_interrupted()
        # Log when interrupt flag is True (only once per check to avoid spam)
        if interrupted:
            print(f"    ✓ _check_interrupted: background monitor detected interrupt!")
        return interrupted


    def _stream_llm_and_speak(self, user_message, ws):
        """Stream LLM response while generating TTS and playing audio in parallel.

        Pipeline: LLM tokens → Buffer → TTS Generator → Audio Player
        All stages run in parallel for minimal latency.
        """
        print("\n🚀 Starting streaming LLM + TTS pipeline...")

        # Initialize pipeline components
        audio_player = AudioPlayer(interrupt_monitor=self.interrupt_monitor, completion_threshold=0.75)
        tts_generator = TTSGenerator(
            audio_player,
            backend=self.tts_backend,
            voice=self.tts_voice,
            rate=self.tts_rate,
            volume=self.tts_volume,
            host=self.tts_host,
            port=self.tts_port,
            seed=self.tts_seed,
            cfg_scale=self.tts_cfg_scale
        )

        # Link tts_generator to audio_player for last-chunk detection
        audio_player.tts_generator = tts_generator

        # Store references for cleanup on disconnect
        self.active_audio_player = audio_player
        self.active_tts_generator = tts_generator

        # Start worker threads
        audio_player.start()
        tts_generator.start()

        # Track full response for history
        full_response = ""
        buffer = ""
        chunks_sent = 0
        first_token = True
        start_time = time.time()
        last_transcript_update = 0  # Track when we last updated transcript

        try:
            # Add user message to history
            self.messages.append({"role": "user", "content": user_message})
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]

            # Use RAG if available, otherwise stream from Ollama
            if self.rag_agent:
                # RAG mode: Get full response synchronously, then process for TTS
                print(f"🧠 Using RAG Agent for query...")
                rag_response = self._query_rag(user_message)

                if rag_response:
                    full_response = rag_response
                    ttft = time.time() - start_time
                    print(f"  ⚡ RAG response time: {ttft:.3f}s")

                    # Clean and chunk the response for TTS
                    clean_response = clean_text_for_tts(full_response)
                    tts_chunks = split_text_into_chunks(clean_response, self.chunk_size_first, self.chunk_size_rest)

                    if tts_chunks:
                        # Queue ALL chunks to TTS generator first (they generate in sequence)
                        for i, chunk in enumerate(tts_chunks):
                            tts_generator.add_text(chunk)
                        print(f"  📝 Queued {len(tts_chunks)} TTS chunks")

                    # Update transcript
                    self._send_transcript(ws, f"You: {user_message}\n\nINDU: {full_response}", replace=True)

                    # Add to conversation history
                    self.messages.append({"role": "assistant", "content": full_response})

                    # Signal TTS done and wait for playback
                    tts_generator.mark_done()

                    print("⏳ Waiting for RAG audio playback...")
                    while not audio_player.is_done() or not tts_generator.is_done():
                        if audio_player.interrupted:
                            break
                        time.sleep(0.05)

                    total_time = time.time() - start_time
                    print(f"✅ RAG pipeline complete ({total_time:.2f}s)")

                    # Clear active references (playback complete)
                    self.active_audio_player = None
                    self.active_tts_generator = None

                    # Ensure ROS2 playback status is False
                    ros2_pub = get_ros2_publisher()
                    if ros2_pub:
                        ros2_pub.set_playing(False)

                    return full_response, audio_player.interrupted
                else:
                    print("⚠️ RAG returned empty, falling back to direct Ollama")

            # Fallback: Stream from LLM directly (no RAG)
            payload = {
                "model": self.llm_model,
                "messages": self.messages,
                "stream": True,
                "options": {
                    "temperature": self.llm_temperature
                }
            }

            print(f"🧠 Streaming from LLM: {self.llm_model} (temp={self.llm_temperature})")
            response = requests.post(self.ollama_url, json=payload, stream=True, timeout=300)
            response.raise_for_status()

            for line in response.iter_lines():
                # Check for interrupt
                if audio_player.interrupted:
                    print("🛑 Pipeline interrupted, stopping LLM stream")
                    break

                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            token = chunk["message"]["content"]

                            if first_token:
                                ttft = time.time() - start_time
                                print(f"  ⚡ TTFT: {ttft:.3f}s")
                                first_token = False

                            # Add token to buffers
                            buffer += token
                            full_response += token

                            # Stream transcript update to web interface (every 100ms or on sentence end)
                            current_time = time.time()
                            if current_time - last_transcript_update > 0.1 or token in '.!?\n':
                                self._send_transcript(ws, f"You: {user_message}\n\nINDU: {full_response}", replace=True)
                                last_transcript_update = current_time

                            # Check if buffer ready for TTS
                            chunk_size = self.chunk_size_first if chunks_sent == 0 else self.chunk_size_rest

                            if len(buffer) >= chunk_size or (len(buffer) > 30 and token in '.!?\n'):
                                clean_buffer = clean_text_for_tts(buffer.strip())
                                if clean_buffer:
                                    chunks_sent += 1
                                    tts_generator.add_text(clean_buffer)
                                buffer = ""

                    except json.JSONDecodeError:
                        continue

            # Handle remaining buffer
            if buffer.strip() and not audio_player.interrupted:
                clean_buffer = clean_text_for_tts(buffer.strip())
                if clean_buffer:
                    chunks_sent += 1
                    tts_generator.add_text(clean_buffer)

            # Signal TTS generator that we're done
            tts_generator.mark_done()

            # Add to conversation history
            self.messages.append({"role": "assistant", "content": full_response})

            llm_time = time.time() - start_time
            print(f"\n✅ LLM complete ({llm_time:.2f}s) - {len(full_response)} chars, {chunks_sent} chunks")

            # Wait for TTS and audio to finish
            print("⏳ Waiting for audio playback to complete...")
            while not audio_player.is_done() or not tts_generator.is_done():
                if audio_player.interrupted:
                    break
                time.sleep(0.1)

            total_time = time.time() - start_time
            print(f"✅ Pipeline complete ({total_time:.2f}s total)")

            # Clear active references (playback complete)
            self.active_audio_player = None
            self.active_tts_generator = None

            # Ensure ROS2 playback status is False
            ros2_pub = get_ros2_publisher()
            if ros2_pub:
                ros2_pub.set_playing(False)

            return full_response, audio_player.interrupted

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            # Clear active references on error too
            self.active_audio_player = None
            self.active_tts_generator = None
            # Ensure ROS2 playback status is False on error
            ros2_pub = get_ros2_publisher()
            if ros2_pub:
                ros2_pub.set_playing(False)
            import traceback
            traceback.print_exc()
            return "", False

        finally:
            # Cleanup
            tts_generator.stop()
            audio_player.stop()

    def _generate_and_play_tts_chunked_OLD(self, text, ws):
        """OLD METHOD - kept for reference. Use _stream_llm_and_speak instead."""
        pass

    def _generate_and_play_tts_chunked(self, text, ws):
        """Generate TTS in chunks and play on SERVER speakers with interrupt detection"""
        # Clean text before TTS
        text = clean_text_for_tts(text)

        # Split into chunks (50 chars first, 150 rest)
        chunks = split_text_into_chunks(text, self.chunk_size_first, self.chunk_size_rest)

        if len(chunks) == 0:
            return False

        print(f"🔊 Speaking ({len(chunks)} chunks)...")

        # Pre-generate chunks using asyncio
        audio_paths = [None] * len(chunks)
        generation_tasks = {}

        async def generate_chunk_task(idx, chunk_text):
            """Generate a single chunk and store result"""
            try:
                audio_paths[idx] = await generate_tts_chunk(
                    chunk_text,
                    backend=self.tts_backend,
                    voice=self.tts_voice,
                    rate=self.tts_rate,
                    volume=self.tts_volume
                )
            except Exception as e:
                print(f"\n❌ Error generating chunk {idx+1}: {e}")
                audio_paths[idx] = None

        async def generate_chunks_async():
            """Async function to generate all chunks with pre-buffering"""
            # Pre-buffer first 3 chunks
            for i in range(min(3, len(chunks))):
                generation_tasks[i] = asyncio.create_task(generate_chunk_task(i, chunks[i]))

            for i in range(len(chunks)):
                # Check for interrupts before processing chunk
                if self._check_interrupted():
                    remaining = len(chunks) - i
                    print(f"\n🎤 Interrupted! Discarding {remaining} remaining chunks")
                    return True

                # Wait for current chunk to be ready
                if i in generation_tasks:
                    await generation_tasks[i]

                audio_path = audio_paths[i]

                if not audio_path:
                    print(f"  ⚠️  Chunk {i+1} generation failed, skipping...")
                    continue

                # Start generating next chunk(s) in parallel
                for j in range(i + 3, min(i + 4, len(chunks))):
                    if j not in generation_tasks:
                        generation_tasks[j] = asyncio.create_task(generate_chunk_task(j, chunks[j]))

                print(f"  ▶ Playing chunk {i+1}/{len(chunks)} on SERVER speakers", flush=True)

                # Play audio on SERVER with interrupt checking and echo cancellation
                interrupted = play_audio(
                    audio_path,
                    check_interrupt_callback=self._check_interrupted,
                    interrupt_monitor=self.interrupt_monitor
                )

                if interrupted:
                    remaining = len(chunks) - (i + 1)
                    print(f"\n🎤 Interrupted! Discarding {remaining} remaining chunks")
                    return True

                # Small buffer between chunks
                time.sleep(0.05)

            print("\n✅ Done speaking\n")
            return False

        # Run the async generation and playback
        asyncio.run(generate_chunks_async())

    def _save_conversation(self, user_input, ai_response):
        """Save conversation to JSON file with timestamp"""
        try:
            from datetime import datetime

            # Create conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "ai": ai_response
            }

            # Load existing conversations or create new list
            conversations = []
            if CONVERSATIONS_FILE.exists():
                try:
                    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Support both list and dict formats
                        if isinstance(data, list):
                            conversations = data
                        elif isinstance(data, dict) and 'conversations' in data:
                            conversations = data['conversations']
                except Exception as e:
                    print(f"⚠️  Error loading conversations file: {e}")
                    conversations = []

            # Append new conversation
            conversations.append(conversation_entry)

            # Save to file
            with open(CONVERSATIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "conversations": conversations,
                    "total_count": len(conversations)
                }, f, indent=2, ensure_ascii=False)

            print(f"💾 Conversation saved to {CONVERSATIONS_FILE} (total: {len(conversations)})")

        except Exception as e:
            print(f"❌ Error saving conversation: {e}")

    def _send_error(self, ws, error):
        """Send error message to client"""
        try:
            ws.send(json.dumps({
                'type': 'error',
                'error': error
            }))
        except Exception as e:
            print(f"⚠️  Error sending error message: {e}")


# Global streaming assistant instance
streaming_assistant = WebStreamingAssistant()

# Initialize ROS2 publisher for playback status (publishes continuously at 10Hz)
init_ros2_publisher()


# ============================================================
# WEB ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve streaming voice assistant page"""
    return render_template('stream.html')


@app.route('/config')
def config_page():
    """Serve configuration page"""
    return render_template('config.html')


@app.route('/stream')
def stream_page():
    """Serve streaming voice assistant page"""
    return render_template('stream.html')


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/stream', methods=['POST'])
def stream_audio():
    """Receive audio stream from client"""
    try:
        audio_data = request.data
        # TODO: Process audio through voice assistant
        # For now, just acknowledge receipt
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config = load_config()
        return jsonify({
            'status': 'success',
            'config': config
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


def deep_merge(base, updates):
    """Deep merge updates into base config, preserving unmodified fields"""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@app.route('/api/config', methods=['POST'])
def save_config_api():
    """Save configuration"""
    try:
        config_data = request.json
        if not config_data:
            return jsonify({
                'status': 'error',
                'error': 'No configuration data provided'
            }), 400

        # Load existing config and merge with updates (preserves fields not in UI)
        existing_config = load_config()
        merged_config = deep_merge(existing_config, config_data)

        # Save merged configuration
        if save_config(merged_config):
            return jsonify({
                'status': 'success',
                'message': 'Configuration saved successfully. Restart assistant to apply changes.'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to save configuration'
            }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/voices/edge', methods=['GET'])
def get_edge_voices():
    """Get available Edge TTS voices with optional language filter (from cache)"""
    try:
        # Get language filter from query params (e.g., ?lang=hi-IN or ?lang=en-IN)
        language_filter = request.args.get('lang', None)

        # Load voices from cache file
        cache_file = Path(__file__).parent / "edge_voices_cache.json"

        # If cache doesn't exist, generate it
        if not cache_file.exists():
            import asyncio
            print("⚠️  Voice cache not found, fetching from Edge TTS...")
            voices = asyncio.run(edge_tts.list_voices())

            # Build cache
            voices_data = []
            for voice in voices:
                language = voice.get('LocaleName') or voice.get('FriendlyName') or voice.get('Locale', 'Unknown')
                voices_data.append({
                    'name': voice.get('ShortName', 'Unknown'),
                    'gender': voice.get('Gender', 'Unknown'),
                    'locale': voice.get('Locale', 'Unknown'),
                    'language': language
                })

            # Save cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'voices': voices_data, 'count': len(voices_data)}, f, indent=2)
        else:
            # Load from cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                voices_data = cache_data.get('voices', [])

        # Filter voices if requested
        voices_list = []
        locales_set = set()

        for voice in voices_data:
            locale = voice['locale']
            locales_set.add(locale)

            # Apply language filter if provided
            if language_filter:
                # Support both full locale (hi-IN) and language prefix (hi)
                if not (locale.startswith(language_filter) or locale == language_filter):
                    continue

            voices_list.append(voice)

        # Get available locales for the filter dropdown
        available_locales = sorted(list(locales_set))

        return jsonify({
            'status': 'success',
            'voices': voices_list,
            'count': len(voices_list),
            'available_locales': available_locales,
            'filter_applied': language_filter,
            'from_cache': cache_file.exists()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/models/ollama', methods=['GET'])
def get_ollama_models():
    """Get available Ollama models (from cache or live API)"""
    try:
        config = load_config()
        host = config.get('llm', {}).get('host', '172.16.6.19')
        port = config.get('llm', {}).get('port', 11434)

        # Load models from cache file
        cache_file = Path(__file__).parent / "ollama_models_cache.json"
        models = []
        from_cache = False

        # Try cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    models = cache_data.get('models', [])
                    from_cache = True
            except Exception as e:
                print(f"⚠️  Error loading model cache: {e}")

        # If cache is empty or doesn't exist, try live API
        if not models:
            try:
                url = f"http://{host}:{port}/api/tags"
                response = requests.get(url, timeout=5)
                response.raise_for_status()

                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                from_cache = False
            except requests.exceptions.ConnectionError:
                # If both cache and API fail, return error
                if not from_cache:
                    return jsonify({
                        'status': 'error',
                        'error': f'Cannot connect to Ollama at {host}:{port}. Create ollama_models_cache.json to add models manually.'
                    }), 503
            except Exception as e:
                if not from_cache:
                    return jsonify({
                        'status': 'error',
                        'error': str(e)
                    }), 500

        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models),
            'from_cache': from_cache
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Config server running'
    })


@app.route('/api/calibrate', methods=['POST'])
def trigger_calibration():
    """Run audio calibration via web UI"""
    try:
        result = run_calibration()

        if result['success']:
            return jsonify({
                'status': 'success',
                'message': result['message'],
                'echo_scale': result.get('echo_scale')
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result['error']
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ============================================================
# WEBSOCKET ENDPOINTS
# ============================================================

@sock.route('/ws/stream')
def stream_websocket(ws):
    """WebSocket endpoint for audio streaming"""
    print("📡 New WebSocket connection established")

    try:
        # Send initial state
        ws.send(json.dumps({
            'type': 'state',
            'state': 'idle'
        }))

        # Process incoming audio chunks
        while True:
            # Receive audio data from client
            data = ws.receive()

            if data is None:
                break

            # Process binary audio data
            if isinstance(data, bytes):
                # Process audio chunk synchronously
                streaming_assistant.process_audio_chunk(data, ws)

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        print("📡 WebSocket connection closed")
        # Clean up - stop playback, reset state, publish ROS2 False
        streaming_assistant.cleanup()


# ============================================================
# BACKGROUND THREAD RUNNER
# ============================================================

def run_config_server(host='127.0.0.1', port=8080, use_ssl=False):
    """Run Flask server (blocking)"""
    try:
        if use_ssl:
            # Check if SSL certificates exist
            cert_file = Path(__file__).parent / "cert.pem"
            key_file = Path(__file__).parent / "key.pem"

            if cert_file.exists() and key_file.exists():
                ssl_context = (str(cert_file), str(key_file))
                print(f"🔒 HTTPS enabled - SSL certificates found")
                app.run(host=host, port=port, debug=False, use_reloader=False,
                       threaded=True, ssl_context=ssl_context)
            else:
                print(f"⚠️  SSL certificates not found, falling back to HTTP")
                app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        else:
            app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"❌ Config server error: {e}")


def start_config_server_thread(host='127.0.0.1', port=8080, use_ssl=False):
    """Start config server in daemon thread"""
    thread = threading.Thread(
        target=run_config_server,
        args=(host, port, use_ssl),
        daemon=True,
        name="ConfigServerThread"
    )
    thread.start()
    return thread


# ============================================================
# STANDALONE SERVER (for testing)
# ============================================================

if __name__ == "__main__":
    import socket

    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("=" * 70)
    print("BR_AI_N Voice Assistant - Configuration Server")
    print("=" * 70)
    print(f"\n📡 Server URLs:")
    print(f"   http://127.0.0.1:8080/config  (Configuration)")
    print(f"   http://{local_ip}:8080/stream  (Streaming Voice Assistant)")
    print(f"\n   Or with HTTPS (for remote microphone access):")
    print(f"   https://{local_ip}:8080/stream")
    print(f"\n⚠️  For HTTPS: Accept the security warning in your browser")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")

    # Run with SSL for remote microphone access
    run_config_server(host='0.0.0.0', port=8080, use_ssl=True)
