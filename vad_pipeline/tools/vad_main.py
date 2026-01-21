#!/usr/bin/env python3
"""
VAD Pipeline - Main Entry Point
Run the VAD-based recording/playback pipeline
"""

import argparse
from vad_pipeline import VADPipelineManager, VADConfig


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="VAD Pipeline - Voice Activity Detection Recording/Playback System"
    )

    # Configuration arguments
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='VAD threshold (0.0-1.0, default: 0.5)'
    )

    parser.add_argument(
        '--silence-timeout',
        type=float,
        default=1.5,
        help='Silence timeout in seconds (default: 1.5)'
    )

    parser.add_argument(
        '--max-duration',
        type=float,
        default=30.0,
        help='Maximum recording duration in seconds (default: 30.0)'
    )

    parser.add_argument(
        '--frames-needed',
        type=int,
        default=3,
        help='Frames needed for speech confirmation (default: 3)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--save-recordings',
        action='store_true',
        help='Save recordings to WAV files'
    )

    parser.add_argument(
        '--recordings-dir',
        type=str,
        default='./recordings',
        help='Directory for saved recordings (default: ./recordings)'
    )

    args = parser.parse_args()

    # Create configuration
    config = VADConfig(
        vad_threshold=args.threshold,
        silence_timeout=args.silence_timeout,
        max_recording_duration=args.max_duration,
        frames_needed=args.frames_needed,
        verbose=args.verbose,
        save_recordings=args.save_recordings,
        recordings_dir=args.recordings_dir
    )

    # Print configuration
    print("\nVAD Pipeline Configuration:")
    print(f"  Sample Rate: {config.sample_rate} Hz")
    print(f"  Chunk Size: {config.chunk_size} samples ({config.chunk_size/config.sample_rate*1000:.1f}ms)")
    print(f"  VAD Threshold: {config.vad_threshold}")
    print(f"  Frames Needed: {config.frames_needed}")
    print(f"  Silence Timeout: {config.silence_timeout}s")
    print(f"  Max Duration: {config.max_recording_duration}s")
    print(f"  Verbose: {config.verbose}")
    print(f"  Save Recordings: {config.save_recordings}")
    if config.save_recordings:
        print(f"  Recordings Dir: {config.recordings_dir}")
    print()

    # Create and run pipeline
    pipeline = VADPipelineManager(config)
    pipeline.run()


if __name__ == "__main__":
    main()
