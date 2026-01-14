"""
VAD Processor using Silero VAD
Handles voice activity detection with multi-frame confirmation
"""

import torch
import numpy as np
from typing import Tuple
import time


class VADProcessor:
    """Voice Activity Detection processor using Silero VAD model"""

    def __init__(self, threshold: float = 0.5, frames_needed: int = 3, sample_rate: int = 16000, chunk_size: int = 512,
                 interrupt_threshold: float = 1.0, interrupt_frame_duration: float = 0.5, interrupt_frames_needed: int = 3,
                 verbose: bool = False):
        """
        Initialize VAD processor

        Args:
            threshold: Confidence threshold for speech detection (0.0-1.0)
            frames_needed: Number of consecutive frames needed to confirm speech
            sample_rate: Audio sample rate (must be 16000 for Silero)
            chunk_size: Chunk size (must be 512 for 16kHz, 256 for 8kHz)
            interrupt_threshold: Threshold for interrupt detection (1.0 = 100% confidence)
            interrupt_frame_duration: Duration of each interrupt frame in seconds
            interrupt_frames_needed: Number of frames needed to confirm interrupt
            verbose: Enable verbose logging
        """
        self.threshold = threshold
        self.frames_needed = frames_needed
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.interrupt_threshold = interrupt_threshold
        self.interrupt_frame_duration = interrupt_frame_duration
        self.interrupt_frames_needed = interrupt_frames_needed
        self.verbose = verbose

        # Validate chunk size for Silero VAD
        expected_chunk = 512 if sample_rate == 16000 else 256
        if chunk_size != expected_chunk:
            raise ValueError(f"Silero VAD requires {expected_chunk} samples for {sample_rate}Hz audio")

        # Load Silero VAD model
        print("Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.eval()

        # Get utility functions
        (self.get_speech_timestamps, _, self.read_audio, *_) = utils

        # Frame confirmation buffers
        self.recent_frames = []

        # Interrupt detection state
        self.interrupt_frame_buffer = []  # Stores VAD probs for current frame
        self.interrupt_frames_confirmed = 0  # Count of consecutive frames at 1.0
        self.chunks_per_interrupt_frame = int(sample_rate * interrupt_frame_duration / chunk_size)

        print(f"VAD model loaded. Threshold: {threshold}, Frames needed: {frames_needed}")
        print(f"Interrupt detection: {interrupt_frames_needed} frames of {interrupt_frame_duration}s at {interrupt_threshold} confidence")
        print(f"  ({self.chunks_per_interrupt_frame} chunks per interrupt frame)")

    def is_speech(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio chunk contains speech

        Args:
            audio_chunk: NumPy array of audio samples (mono, 16kHz)

        Returns:
            Tuple of (is_speech: bool, probability: float)
        """
        # Convert to torch tensor
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize to [-1, 1] if needed
        if audio_chunk.max() > 1.0 or audio_chunk.min() < -1.0:
            audio_chunk = audio_chunk / 32768.0

        audio_tensor = torch.from_numpy(audio_chunk)

        # Run VAD inference
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        is_speech = speech_prob >= self.threshold

        return is_speech, speech_prob

    def detect_speech_start(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect speech start with multi-frame confirmation

        Args:
            audio_chunk: NumPy array of audio samples

        Returns:
            Tuple of (speech_confirmed: bool, probability: float)
        """
        is_speech, prob = self.is_speech(audio_chunk)

        # Add to recent frames buffer
        self.recent_frames.append(is_speech)

        # Keep only last N frames
        if len(self.recent_frames) > self.frames_needed:
            self.recent_frames.pop(0)

        # Check if all recent frames indicate speech
        speech_confirmed = (
            len(self.recent_frames) == self.frames_needed
            and all(self.recent_frames)
        )

        return speech_confirmed, prob

    def detect_silence(self, timeout: float = 1.5) -> Tuple[bool, list]:
        """
        Track silence detection state

        Args:
            timeout: Duration of silence needed (in seconds)

        Returns:
            Tuple of (is_silence: bool, silence_frames: list)
        """
        # This is a stateful operation that should be called with each frame
        # The caller should track the silence duration
        # This method just tracks the recent non-speech frames
        return False, []

    def detect_interrupt(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Detect interrupt with strict requirements (VAD == 1.0 across frames of 0.5s)

        Args:
            audio_chunk: NumPy array of audio samples

        Returns:
            Tuple of (interrupt_confirmed: bool, probability: float)
        """
        # Get VAD probability for this chunk
        is_speech, prob = self.is_speech(audio_chunk)

        # Add to current frame buffer
        self.interrupt_frame_buffer.append(prob)

        if self.verbose:
            # Show progress every 5 chunks
            if len(self.interrupt_frame_buffer) % 5 == 0:
                avg_prob = sum(self.interrupt_frame_buffer) / len(self.interrupt_frame_buffer)
                print(f"  [INTERRUPT CHECK] Chunk {len(self.interrupt_frame_buffer)}/{self.chunks_per_interrupt_frame} | "
                      f"Current VAD: {prob:.3f} | Frame Avg: {avg_prob:.3f} | "
                      f"Frames confirmed: {self.interrupt_frames_confirmed}/{self.interrupt_frames_needed}")

        # Check if we've completed a frame (0.5 seconds worth of chunks)
        if len(self.interrupt_frame_buffer) >= self.chunks_per_interrupt_frame:
            # Get statistics for verbose output
            avg_prob = sum(self.interrupt_frame_buffer) / len(self.interrupt_frame_buffer)
            max_prob = max(self.interrupt_frame_buffer)
            min_prob = min(self.interrupt_frame_buffer)
            count_above_thresh = sum(1 for p in self.interrupt_frame_buffer if p >= self.interrupt_threshold)

            # Check if at least 67% of chunks in this frame have VAD >= interrupt_threshold
            required_chunks = int(len(self.interrupt_frame_buffer) * 0.67)  # 67% threshold (~10/15 chunks)
            all_max_confidence = count_above_thresh >= required_chunks

            if self.verbose:
                print(f"\n  [FRAME COMPLETE] Stats: avg={avg_prob:.3f}, max={max_prob:.3f}, min={min_prob:.3f}, "
                      f"chunks>={self.interrupt_threshold}={count_above_thresh}/{len(self.interrupt_frame_buffer)} (need {required_chunks})")

            if all_max_confidence:
                # This frame passed, increment counter
                self.interrupt_frames_confirmed += 1
                if self.verbose:
                    print(f"  [FRAME PASSED] {count_above_thresh}/{len(self.interrupt_frame_buffer)} chunks >= {self.interrupt_threshold} (67%+ rule)! Frames confirmed: {self.interrupt_frames_confirmed}/{self.interrupt_frames_needed}")
            else:
                # This frame failed, reset counter
                if self.verbose and self.interrupt_frames_confirmed > 0:
                    print(f"  [FRAME FAILED] Not all chunks >= {self.interrupt_threshold}, resetting counter (was {self.interrupt_frames_confirmed})")
                self.interrupt_frames_confirmed = 0

            # Clear buffer for next frame
            self.interrupt_frame_buffer = []

            # Check if we have enough consecutive frames above threshold
            if self.interrupt_frames_confirmed >= self.interrupt_frames_needed:
                if self.verbose:
                    print(f"  [INTERRUPT CONFIRMED!] {self.interrupt_frames_needed} frames at VAD >= {self.interrupt_threshold}")
                return True, avg_prob

        return False, prob

    def reset_interrupt_detection(self):
        """Reset interrupt detection state"""
        self.interrupt_frame_buffer = []
        self.interrupt_frames_confirmed = 0

    def reset(self):
        """Reset the frame confirmation buffer"""
        self.recent_frames = []
        self.reset_interrupt_detection()

    def get_stats(self) -> dict:
        """Get current VAD statistics"""
        return {
            'threshold': self.threshold,
            'frames_needed': self.frames_needed,
            'recent_frames': self.recent_frames.copy(),
            'recent_speech_count': sum(self.recent_frames),
            'interrupt_frames_confirmed': self.interrupt_frames_confirmed,
            'interrupt_buffer_size': len(self.interrupt_frame_buffer)
        }
