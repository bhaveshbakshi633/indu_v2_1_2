"""
Audio Recorder
Records audio from input monitor and detects end of speech
"""

import numpy as np
import soundfile as sf
import time
import os
from typing import Optional
from .vad_processor import VADProcessor
from .audio_input import AudioInputMonitor


class AudioRecorder:
    """Records audio with silence detection"""

    def __init__(self, vad_processor: VADProcessor, sample_rate: int = 16000,
                 silence_timeout: float = 1.5, max_duration: float = 30.0,
                 save_recordings: bool = False, recordings_dir: str = "./recordings",
                 verbose: bool = False):
        """
        Initialize audio recorder

        Args:
            vad_processor: VAD processor instance for silence detection
            sample_rate: Audio sample rate
            silence_timeout: Seconds of silence before stopping recording
            max_duration: Maximum recording duration in seconds
            save_recordings: Save recordings to WAV files
            recordings_dir: Directory for saved recordings
            verbose: Enable verbose logging
        """
        self.vad = vad_processor
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.max_duration = max_duration
        self.save_recordings = save_recordings
        self.recordings_dir = recordings_dir
        self.verbose = verbose

        # Recording state
        self.buffer = []
        self.is_recording = False
        self.recording_count = 0

        # Create recordings directory if needed
        if self.save_recordings and not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)

    def record_from_monitor(self, audio_monitor: AudioInputMonitor) -> Optional[np.ndarray]:
        """
        Record audio from input monitor until silence detected

        Args:
            audio_monitor: Audio input monitor to record from

        Returns:
            Recorded audio as NumPy array, or None if failed
        """
        self.buffer = []
        self.is_recording = True
        start_time = time.time()
        silence_start = None
        chunk_duration = audio_monitor.chunk_size / audio_monitor.sample_rate

        if self.verbose:
            print("Recording started...")

        try:
            while self.is_recording:
                # Get next audio chunk
                chunk = audio_monitor.get_chunk(timeout=0.1)

                if chunk is None:
                    continue

                # Add to buffer
                self.buffer.append(chunk)

                # Check for speech/silence
                is_speech, prob = self.vad.is_speech(chunk)

                if self.verbose and len(self.buffer) % 10 == 0:
                    duration = len(self.buffer) * chunk_duration
                    print(f"  Recording... {duration:.1f}s | VAD: {prob:.3f} | Speech: {is_speech}")

                if is_speech:
                    # Reset silence timer
                    silence_start = None
                else:
                    # Start or continue silence timer
                    if silence_start is None:
                        silence_start = time.time()
                    else:
                        silence_duration = time.time() - silence_start
                        if silence_duration >= self.silence_timeout:
                            if self.verbose:
                                print(f"Silence detected after {silence_duration:.1f}s, stopping recording")
                            break

                # Check max duration
                elapsed = time.time() - start_time
                if elapsed >= self.max_duration:
                    if self.verbose:
                        print(f"Max recording duration ({self.max_duration}s) reached")
                    break

        except KeyboardInterrupt:
            if self.verbose:
                print("Recording interrupted by user")
            self.is_recording = False
            return None

        self.is_recording = False

        # Combine buffer chunks
        if not self.buffer:
            if self.verbose:
                print("No audio recorded")
            return None

        recording = np.concatenate(self.buffer)
        duration = len(recording) / self.sample_rate

        if self.verbose:
            print(f"Recording finished: {duration:.2f}s, {len(recording)} samples")

        # Save if enabled
        if self.save_recordings:
            self._save_recording(recording)

        self.recording_count += 1

        return recording

    def _save_recording(self, recording: np.ndarray):
        """Save recording to WAV file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}_{self.recording_count:03d}.wav"
        filepath = os.path.join(self.recordings_dir, filename)

        try:
            sf.write(filepath, recording, self.sample_rate)
            if self.verbose:
                print(f"Recording saved to {filepath}")
        except Exception as e:
            print(f"Error saving recording: {e}")

    def stop_recording(self):
        """Force stop current recording"""
        self.is_recording = False

    def clear_buffer(self):
        """Clear the recording buffer"""
        self.buffer = []

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds"""
        if not self.buffer:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self.buffer)
        return total_samples / self.sample_rate

    def get_stats(self) -> dict:
        """Get recording statistics"""
        return {
            'is_recording': self.is_recording,
            'buffer_duration': self.get_buffer_duration(),
            'buffer_chunks': len(self.buffer),
            'recording_count': self.recording_count,
            'silence_timeout': self.silence_timeout,
            'max_duration': self.max_duration
        }
