"""
Configuration dataclass for VAD Pipeline
"""

from dataclasses import dataclass


@dataclass
class VADConfig:
    """Configuration for VAD Pipeline System"""

    # Core audio settings
    sample_rate: int = 16000
    """Audio sample rate in Hz (must be 16kHz for Silero VAD)"""

    chunk_size: int = 512
    """Audio chunk size in samples (32ms at 16kHz, required for Silero VAD)"""

    channels: int = 1
    """Number of audio channels (1 = mono)"""

    # VAD detection settings
    vad_threshold: float = 0.5
    """VAD confidence threshold (0.0-1.0). Higher = less sensitive. Tune 0.3-0.7"""

    frames_needed: int = 3
    """Number of consecutive frames needed to confirm speech (reduces false positives)"""

    # Interrupt detection settings (stricter than regular speech detection)
    interrupt_threshold: float = 0.95
    """VAD threshold for interrupt detection during playback (0.95 = 95% confidence)"""

    interrupt_frame_duration: float = 0.5
    """Duration of each interrupt detection frame in seconds"""

    interrupt_frames_needed: int = 1
    """Number of consecutive frames needed to confirm interrupt"""

    # Recording settings
    silence_timeout: float = 1.5
    """Seconds of silence before ending recording"""

    max_recording_duration: float = 30.0
    """Maximum recording duration in seconds"""

    # Buffer settings
    audio_queue_size: int = 100
    """Max audio chunks in queue (100 chunks = 6 seconds of buffering)"""

    # Debug settings
    verbose: bool = False
    """Enable verbose logging"""

    save_recordings: bool = False
    """Save recordings to WAV files for debugging"""

    recordings_dir: str = "./recordings"
    """Directory to save recordings if save_recordings=True"""

    def __post_init__(self):
        """Validate configuration"""
        if self.sample_rate != 16000:
            raise ValueError("Silero VAD requires 16kHz sample rate")

        if self.chunk_size != 512:
            raise ValueError("Silero VAD requires 512 samples (32ms) for 16kHz audio")

        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("VAD threshold must be between 0.0 and 1.0")

        if self.frames_needed < 1:
            raise ValueError("frames_needed must be at least 1")
