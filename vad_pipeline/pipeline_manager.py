"""
VAD Pipeline Manager
Main orchestrator for the VAD-based recording/playback pipeline
"""

import threading
import time
import numpy as np
from enum import Enum
from typing import Optional

from .config import VADConfig
from .vad_processor import VADProcessor
from .audio_input import AudioInputMonitor
from .audio_recorder import AudioRecorder
from .audio_player import AudioPlayer


class PipelineState(Enum):
    """Pipeline state machine states"""
    LISTENING = "listening"   # Monitoring for speech start
    RECORDING = "recording"   # Capturing user speech
    PLAYING = "playing"       # Playing back audio + monitoring for interrupts


class VADPipelineManager:
    """Main pipeline orchestrator with state machine"""

    def __init__(self, config: Optional[VADConfig] = None):
        """
        Initialize VAD pipeline manager

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or VADConfig()

        # Initialize components
        print("Initializing VAD Pipeline...")

        self.vad = VADProcessor(
            threshold=self.config.vad_threshold,
            frames_needed=self.config.frames_needed,
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
            interrupt_threshold=self.config.interrupt_threshold,
            interrupt_frame_duration=self.config.interrupt_frame_duration,
            interrupt_frames_needed=self.config.interrupt_frames_needed,
            verbose=self.config.verbose
        )

        self.audio_monitor = AudioInputMonitor(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
            channels=self.config.channels,
            queue_size=self.config.audio_queue_size,
            verbose=self.config.verbose
        )

        self.recorder = AudioRecorder(
            vad_processor=self.vad,
            sample_rate=self.config.sample_rate,
            silence_timeout=self.config.silence_timeout,
            max_duration=self.config.max_recording_duration,
            save_recordings=self.config.save_recordings,
            recordings_dir=self.config.recordings_dir,
            verbose=self.config.verbose
        )

        self.player = AudioPlayer(
            sample_rate=self.config.sample_rate,
            verbose=self.config.verbose
        )

        # State machine
        self.state = PipelineState.LISTENING
        self.state_lock = threading.Lock()

        # Control flags
        self.is_running = False
        self.current_recording = None

        # Statistics
        self.loop_count = 0
        self.state_transitions = []

        print("VAD Pipeline initialized")

    def run(self):
        """
        Run the pipeline (infinite loop)

        Main state machine loop:
        LISTENING → RECORDING → PLAYING → LISTENING
        with interrupt capability during PLAYING
        """
        print("\n" + "="*60)
        print("VAD PIPELINE STARTED")
        print("="*60)
        print("\nStates:")
        print("  LISTENING: Waiting for you to speak...")
        print("  RECORDING: Recording your speech...")
        print("  PLAYING:   Playing back your recording (interrupt anytime)")
        print("\nPress Ctrl+C to stop\n")
        print("="*60 + "\n")

        # Start audio input monitor
        self.audio_monitor.start()
        self.is_running = True

        try:
            while self.is_running:
                self.loop_count += 1

                if self.state == PipelineState.LISTENING:
                    self._handle_listening_state()

                elif self.state == PipelineState.RECORDING:
                    self._handle_recording_state()

                elif self.state == PipelineState.PLAYING:
                    self._handle_playing_state()

                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nShutting down pipeline...")
        finally:
            self.shutdown()

    def _handle_listening_state(self):
        """Handle LISTENING state - monitor for speech start"""
        # Get audio chunk
        chunk = self.audio_monitor.get_chunk(timeout=0.1)
        if chunk is None:
            return

        # Check for speech with multi-frame confirmation
        speech_confirmed, prob = self.vad.detect_speech_start(chunk)

        if speech_confirmed:
            if self.config.verbose:
                print(f"\n[LISTENING] Speech detected! (prob={prob:.3f})")

            # Transition to RECORDING
            self._transition_to_recording()

    def _handle_recording_state(self):
        """Handle RECORDING state - record until silence detected"""
        # Record from audio monitor (blocking until silence or timeout)
        recording = self.recorder.record_from_monitor(self.audio_monitor)

        if recording is not None and len(recording) > 0:
            self.current_recording = recording

            # Transition to PLAYING
            self._transition_to_playing()
        else:
            # No recording captured, go back to LISTENING
            if self.config.verbose:
                print("[RECORDING] No audio captured, returning to LISTENING")
            self._transition_to_listening()

    def _handle_playing_state(self):
        """Handle PLAYING state - play back recording while monitoring for interrupts"""
        if self.current_recording is None:
            self._transition_to_listening()
            return

        # Start playback (non-blocking)
        playback_started = self.player.play(
            self.current_recording,
            on_complete=lambda: self._on_playback_complete(),
            on_interrupt=lambda: self._on_playback_interrupt()
        )

        if not playback_started:
            self._transition_to_listening()
            return

        # Monitor for interrupts while playing (strict detection: VAD == 1.0)
        while self.player.is_active():
            # Get audio chunk
            chunk = self.audio_monitor.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Check for interrupt (strict: VAD == 1.0 across 3 frames of 0.5s each)
            interrupt_confirmed, prob = self.vad.detect_interrupt(chunk)

            if interrupt_confirmed:
                if self.config.verbose:
                    print(f"\n[PLAYING] INTERRUPT detected! (VAD == 1.0 for 3 frames of 0.5s)")

                # Stop playback
                self.player.stop()

                # Discard current recording
                self.current_recording = None

                # Transition to RECORDING
                self._transition_to_recording()
                return

        # If we get here, playback completed without interrupt
        # (callback will handle transition)

    def _on_playback_complete(self):
        """Callback when playback completes normally"""
        if self.config.verbose:
            print("[PLAYING] Playback completed")

        # Discard recording
        self.current_recording = None

        # Transition to LISTENING
        self._transition_to_listening()

    def _on_playback_interrupt(self):
        """Callback when playback is interrupted"""
        if self.config.verbose:
            print("[PLAYING] Playback interrupted")

    def _transition_to_listening(self):
        """Transition to LISTENING state"""
        with self.state_lock:
            old_state = self.state
            self.state = PipelineState.LISTENING
            self._log_transition(old_state, self.state)

            # Reset VAD state
            self.vad.reset()

            # Clear audio queue (fresh start)
            self.audio_monitor.clear_queue()

            print(f"\n{'='*60}")
            print(f"STATE: LISTENING - Waiting for speech...")
            print(f"{'='*60}\n")

    def _transition_to_recording(self):
        """Transition to RECORDING state"""
        with self.state_lock:
            old_state = self.state
            self.state = PipelineState.RECORDING
            self._log_transition(old_state, self.state)

            # Reset VAD state
            self.vad.reset()

            print(f"\n{'='*60}")
            print(f"STATE: RECORDING - Speak now...")
            print(f"{'='*60}\n")

    def _transition_to_playing(self):
        """Transition to PLAYING state"""
        with self.state_lock:
            old_state = self.state
            self.state = PipelineState.PLAYING
            self._log_transition(old_state, self.state)

            # Reset interrupt detection state
            self.vad.reset_interrupt_detection()

            duration = len(self.current_recording) / self.config.sample_rate
            print(f"\n{'='*60}")
            print(f"STATE: PLAYING - Playing back {duration:.2f}s recording...")
            print(f"         (speak clearly to interrupt - requires VAD == 1.0)")
            print(f"{'='*60}\n")

    def _log_transition(self, old_state: PipelineState, new_state: PipelineState):
        """Log state transition"""
        transition = {
            'timestamp': time.time(),
            'from': old_state.value if old_state else None,
            'to': new_state.value,
            'loop_count': self.loop_count
        }
        self.state_transitions.append(transition)

        if self.config.verbose:
            print(f"[STATE] {old_state.value if old_state else 'INIT'} → {new_state.value}")

    def shutdown(self):
        """Shutdown the pipeline gracefully"""
        self.is_running = False

        # Stop components
        if self.player.is_active():
            self.player.stop()

        self.audio_monitor.stop()

        # Print statistics
        print("\n" + "="*60)
        print("PIPELINE STATISTICS")
        print("="*60)
        print(f"Loop iterations: {self.loop_count}")
        print(f"State transitions: {len(self.state_transitions)}")
        print(f"\nAudio Monitor: {self.audio_monitor.get_stats()}")
        print(f"\nRecorder: {self.recorder.get_stats()}")
        print(f"\nPlayer: {self.player.get_stats()}")
        print(f"\nVAD: {self.vad.get_stats()}")
        print("="*60)
        print("\nPipeline stopped. Goodbye!")

    def get_current_state(self) -> PipelineState:
        """Get current pipeline state"""
        return self.state

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        return {
            'state': self.state.value,
            'loop_count': self.loop_count,
            'transitions': len(self.state_transitions),
            'audio_monitor': self.audio_monitor.get_stats(),
            'recorder': self.recorder.get_stats(),
            'player': self.player.get_stats(),
            'vad': self.vad.get_stats()
        }
