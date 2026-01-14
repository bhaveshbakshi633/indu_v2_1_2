"""
VAD Pipeline System
A voice activity detection based pipeline for recording, playback, and interrupt handling.
"""

from .config import VADConfig
from .vad_processor import VADProcessor
from .audio_input import AudioInputMonitor
from .audio_recorder import AudioRecorder
from .audio_player import AudioPlayer
from .pipeline_manager import VADPipelineManager, PipelineState

__all__ = [
    'VADConfig',
    'VADProcessor',
    'AudioInputMonitor',
    'AudioRecorder',
    'AudioPlayer',
    'VADPipelineManager',
    'PipelineState',
]

__version__ = '1.0.0'
