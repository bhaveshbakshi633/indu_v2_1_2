"""
Audio Input Monitor
Continuously monitors microphone input and feeds audio chunks to a queue
"""

import sounddevice as sd
import numpy as np
import queue
import threading
from typing import Optional


class AudioInputMonitor:
    """Monitors microphone input in a background thread"""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512,
                 channels: int = 1, queue_size: int = 100, verbose: bool = False):
        """
        Initialize audio input monitor

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Number of samples per chunk
            channels: Number of audio channels (1=mono, 2=stereo)
            queue_size: Maximum chunks in queue
            verbose: Enable verbose logging
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.verbose = verbose

        # Thread-safe queue for audio chunks
        self.audio_queue = queue.Queue(maxsize=queue_size)

        # Control flags
        self.is_running = threading.Event()
        self.monitor_thread = None
        self.stream = None

        # Statistics
        self.chunks_processed = 0
        self.queue_overflows = 0

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for sounddevice input stream (runs in real-time thread)

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status and self.verbose:
            print(f"Audio input status: {status}")

        # Convert to mono if needed
        if self.channels == 1 and indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]

        # Convert to float32
        audio_data = audio_data.astype(np.float32)

        # Try to add to queue (non-blocking)
        try:
            self.audio_queue.put_nowait(audio_data.copy())
            self.chunks_processed += 1
        except queue.Full:
            # Queue is full, drop the oldest chunk
            self.queue_overflows += 1
            if self.verbose and self.queue_overflows % 10 == 0:
                print(f"Warning: Audio queue overflow (dropped {self.queue_overflows} chunks)")

    def start(self):
        """Start monitoring microphone in background"""
        if self.is_running.is_set():
            print("Audio monitor already running")
            return

        self.is_running.set()

        # Print available audio devices for debugging
        if self.verbose:
            print("\nAvailable audio devices:")
            print(sd.query_devices())
            print(f"\nDefault input device: {sd.query_devices(kind='input')}")

        # Start input stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"Audio input monitor started (sample_rate={self.sample_rate}Hz, chunk_size={self.chunk_size})")

        except Exception as e:
            print(f"Error starting audio input: {e}")
            self.is_running.clear()
            raise

    def stop(self):
        """Stop monitoring microphone"""
        if not self.is_running.is_set():
            return

        self.is_running.clear()

        # Stop and close stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        print(f"Audio input monitor stopped (processed {self.chunks_processed} chunks, {self.queue_overflows} overflows)")

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get next audio chunk from queue

        Args:
            timeout: Maximum time to wait for chunk (seconds)

        Returns:
            Audio chunk as NumPy array, or None if timeout
        """
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            return chunk
        except queue.Empty:
            return None

    def get_queue_size(self) -> int:
        """Get current number of chunks in queue"""
        return self.audio_queue.qsize()

    def clear_queue(self):
        """Clear all chunks from queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def is_active(self) -> bool:
        """Check if monitor is actively running"""
        return self.is_running.is_set()

    def get_stats(self) -> dict:
        """Get monitoring statistics"""
        return {
            'is_running': self.is_running.is_set(),
            'chunks_processed': self.chunks_processed,
            'queue_overflows': self.queue_overflows,
            'queue_size': self.get_queue_size(),
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size
        }
