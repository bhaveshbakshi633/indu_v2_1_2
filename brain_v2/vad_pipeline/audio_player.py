"""
Audio Player
Plays back audio with interrupt capability
"""

import sounddevice as sd
import numpy as np
import threading
import time
from typing import Optional, Callable


class AudioPlayer:
    """Plays audio with interrupt capability"""

    def __init__(self, sample_rate: int = 16000, verbose: bool = False):
        """
        Initialize audio player

        Args:
            sample_rate: Audio sample rate
            verbose: Enable verbose logging
        """
        self.sample_rate = sample_rate
        self.verbose = verbose

        # Playback state
        self.is_playing = threading.Event()
        self.interrupt_event = threading.Event()
        self.playback_thread = None
        self.current_stream = None

        # Statistics
        self.playback_count = 0
        self.interrupt_count = 0

    def play(self, audio_data: np.ndarray, on_complete: Optional[Callable] = None,
             on_interrupt: Optional[Callable] = None) -> bool:
        """
        Play audio data (non-blocking)

        Args:
            audio_data: Audio data as NumPy array
            on_complete: Callback when playback completes normally
            on_interrupt: Callback when playback is interrupted

        Returns:
            True if playback started successfully, False otherwise
        """
        if self.is_playing.is_set():
            if self.verbose:
                print("Already playing, cannot start new playback")
            return False

        # Reset interrupt flag
        self.interrupt_event.clear()

        # Start playback in background thread
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(audio_data, on_complete, on_interrupt),
            daemon=True
        )
        self.playback_thread.start()

        return True

    def _playback_worker(self, audio_data: np.ndarray,
                        on_complete: Optional[Callable],
                        on_interrupt: Optional[Callable]):
        """Worker thread for audio playback"""
        self.is_playing.set()
        self.playback_count += 1

        duration = len(audio_data) / self.sample_rate

        if self.verbose:
            print(f"Starting playback: {duration:.2f}s, {len(audio_data)} samples")

        try:
            # Play audio (blocking call)
            sd.play(audio_data, self.sample_rate, blocking=False)
            self.current_stream = sd.get_stream()

            # Wait for playback to complete or interrupt
            start_time = time.time()
            while sd.get_stream().active:
                # Check for interrupt
                if self.interrupt_event.is_set():
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"Playback interrupted after {elapsed:.2f}s")

                    # Stop playback
                    sd.stop()
                    self.interrupt_count += 1

                    # Call interrupt callback
                    if on_interrupt:
                        on_interrupt()

                    self.is_playing.clear()
                    return

                time.sleep(0.01)  # 10ms polling interval

            # Playback completed normally
            if self.verbose:
                print(f"Playback completed: {duration:.2f}s")

            # Call completion callback
            if on_complete:
                on_complete()

        except Exception as e:
            print(f"Error during playback: {e}")

        finally:
            self.is_playing.clear()
            self.current_stream = None

    def stop(self):
        """Interrupt current playback"""
        if not self.is_playing.is_set():
            return

        if self.verbose:
            print("Stopping playback...")

        # Signal interrupt
        self.interrupt_event.set()

        # Wait for playback thread to finish (with timeout)
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

    def is_active(self) -> bool:
        """Check if currently playing"""
        return self.is_playing.is_set()

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for playback to complete

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if playback completed, False if timeout or interrupted
        """
        if not self.is_playing.is_set():
            return True

        if self.playback_thread:
            self.playback_thread.join(timeout=timeout)

        return not self.is_playing.is_set()

    def get_stats(self) -> dict:
        """Get playback statistics"""
        return {
            'is_playing': self.is_playing.is_set(),
            'playback_count': self.playback_count,
            'interrupt_count': self.interrupt_count,
            'sample_rate': self.sample_rate
        }
