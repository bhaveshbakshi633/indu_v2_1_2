# stop_fast_path.py
# Emergency STOP detection - LLM pipeline se pehle hi catch karo
# Ye module STT output ko intercept karke STOP keywords check karta hai

import logging
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StopType(Enum):
    """STOP ka type - different urgency levels"""
    EMERGENCY = "emergency"      # "emergency", "emergency stop"
    IMMEDIATE = "immediate"      # "stop", "halt", "freeze"
    CASUAL = "casual"            # "ruk", "bas"

@dataclass
class StopDetectionResult:
    """STOP detection ka result"""
    is_stop: bool
    stop_type: Optional[StopType]
    matched_keyword: Optional[str]
    original_transcript: str

# STOP keywords - priority order me
# Emergency keywords sabse pehle check honge
EMERGENCY_KEYWORDS = [
    "emergency",
    "emergency stop",
]

# Immediate stop keywords
IMMEDIATE_KEYWORDS = [
    "stop",
    "halt",
    "freeze",
    "stop now",
    "stop it",
    "stop stop",
]

# Hindi/casual stop keywords
CASUAL_KEYWORDS = [
    "ruk",
    "ruk jao",
    "bas",
    "band karo",
    "rukh",
    "ruko",
]

# Combined list - check order: emergency -> immediate -> casual
ALL_STOP_KEYWORDS = {
    StopType.EMERGENCY: EMERGENCY_KEYWORDS,
    StopType.IMMEDIATE: IMMEDIATE_KEYWORDS,
    StopType.CASUAL: CASUAL_KEYWORDS,
}


def detect_stop(transcript: str) -> StopDetectionResult:
    """
    STT output me STOP keyword detect karo

    Args:
        transcript: Raw STT output text

    Returns:
        StopDetectionResult with detection info
    """
    if not transcript:
        return StopDetectionResult(
            is_stop=False,
            stop_type=None,
            matched_keyword=None,
            original_transcript=transcript
        )

    lower = transcript.lower().strip()

    # Priority order me check karo
    for stop_type, keywords in ALL_STOP_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower:
                logger.warning(f"[STOP FAST-PATH] Detected '{keyword}' in transcript: '{transcript}'")
                return StopDetectionResult(
                    is_stop=True,
                    stop_type=stop_type,
                    matched_keyword=keyword,
                    original_transcript=transcript
                )

    return StopDetectionResult(
        is_stop=False,
        stop_type=None,
        matched_keyword=None,
        original_transcript=transcript
    )


class StopFastPath:
    """
    STOP Fast-Path handler
    Ye class STT output ko intercept karti hai aur STOP keywords pe
    emergency_stop publish karti hai - LLM pipeline se pehle
    """

    def __init__(
        self,
        on_emergency_stop: Optional[Callable[[], None]] = None,
        on_cancel_pending: Optional[Callable[[], None]] = None
    ):
        """
        Args:
            on_emergency_stop: Callback to publish emergency stop to G1
            on_cancel_pending: Callback to cancel pending pipeline operations
        """
        self._on_emergency_stop = on_emergency_stop
        self._on_cancel_pending = on_cancel_pending
        self._enabled = True

    def enable(self):
        """Fast-path enable karo"""
        self._enabled = True
        logger.info("[STOP FAST-PATH] Enabled")

    def disable(self):
        """Fast-path disable karo (testing ke liye)"""
        self._enabled = False
        logger.warning("[STOP FAST-PATH] Disabled - USE WITH CAUTION")

    def process_transcript(self, transcript: str) -> StopDetectionResult:
        """
        STT output process karo - STOP detect hone pe actions trigger karo

        Args:
            transcript: Raw STT output

        Returns:
            StopDetectionResult - caller ko batata hai ki STOP tha ya nahi
        """
        if not self._enabled:
            return StopDetectionResult(
                is_stop=False,
                stop_type=None,
                matched_keyword=None,
                original_transcript=transcript
            )

        result = detect_stop(transcript)

        if result.is_stop:
            # Emergency stop publish karo
            if self._on_emergency_stop:
                logger.warning(f"[STOP FAST-PATH] Triggering emergency stop - type: {result.stop_type}")
                self._on_emergency_stop()

            # Pending operations cancel karo
            if self._on_cancel_pending:
                logger.info("[STOP FAST-PATH] Cancelling pending operations")
                self._on_cancel_pending()

        return result


# ROS2 integration ke liye helper - voice_bridge.py me use hoga
def create_stop_fast_path_with_ros2(voice_bridge) -> StopFastPath:
    """
    ROS2 VoiceBridge ke saath StopFastPath create karo

    Args:
        voice_bridge: VoiceBridge instance jo /emergency_stop publish karega

    Returns:
        Configured StopFastPath instance
    """
    def emergency_stop_callback():
        if hasattr(voice_bridge, 'send_emergency_stop'):
            voice_bridge.send_emergency_stop()
        else:
            logger.error("[STOP FAST-PATH] voice_bridge.send_emergency_stop() not available")

    def cancel_pending_callback():
        if hasattr(voice_bridge, 'cancel_pending'):
            voice_bridge.cancel_pending()

    return StopFastPath(
        on_emergency_stop=emergency_stop_callback,
        on_cancel_pending=cancel_pending_callback
    )


# === TEST ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_phrases = [
        "stop",
        "please stop now",
        "emergency stop",
        "ruk jao",
        "can you wave at me?",
        "walk forward",
        "halt",
        "freeze freeze freeze",
        "bas kar",
        "hello there",
    ]

    print("=== STOP Fast-Path Test ===\n")

    for phrase in test_phrases:
        result = detect_stop(phrase)
        status = "🛑 STOP" if result.is_stop else "✅ PASS"
        print(f"{status}: '{phrase}'")
        if result.is_stop:
            print(f"   Type: {result.stop_type}, Keyword: '{result.matched_keyword}'")
        print()
