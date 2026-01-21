# intent_reasoner.py
# Voice command intent parser - STT output se action ya conversation identify karo
# Ye module LLM se pehle check karta hai ki user ne action bola ya conversation

import re
import json
import logging
import requests
import string
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(Enum):
    ACTION = "action"
    CONVERSATION = "conversation"
    QUERY = "query"

@dataclass
class IntentResult:
    """Intent parsing ka result"""
    intent_type: IntentType
    action_name: Optional[str]
    confidence: float
    requires_confirmation: bool
    reason: str
    original_transcript: str


# Action patterns - direct keyword matching for high confidence
# Format: {keyword: (ACTION_NAME, confidence)}
ACTION_PATTERNS = {
    # System actions - HIGH priority
    "initialize": ("INIT", 0.95),
    "init": ("INIT", 0.95),
    "boot up": ("INIT", 0.95),
    "start up": ("INIT", 0.90),
    "chalu karo": ("INIT", 0.95),
    "shuru karo": ("INIT", 0.95),

    "ready": ("READY", 0.90),
    "get ready": ("READY", 0.95),
    "ready mode": ("READY", 0.95),
    "taiyaar": ("READY", 0.90),

    "damp": ("DAMP", 0.95),
    "damp mode": ("DAMP", 0.95),
    "damping": ("DAMP", 0.90),
    "relax": ("DAMP", 0.85),
    "aaraam": ("DAMP", 0.90),

    "zero torque": ("ZERO_TORQUE", 0.95),
    "limp mode": ("ZERO_TORQUE", 0.90),
    "disable": ("ZERO_TORQUE", 0.85),

    # Posture actions
    "stand up": ("STANDUP", 0.95),
    "get up": ("STANDUP", 0.90),
    "utho": ("STANDUP", 0.95),
    "khade ho": ("STANDUP", 0.95),

    "sit down": ("SIT", 0.95),
    "sit": ("SIT", 0.85),
    "baith": ("SIT", 0.95),

    "squat": ("SQUAT", 0.95),
    "crouch": ("SQUAT", 0.90),

    # Motion actions
    "walk forward": ("FORWARD", 0.95),
    "go forward": ("FORWARD", 0.95),
    "move forward": ("FORWARD", 0.95),
    "go ahead": ("FORWARD", 0.90),
    "aage": ("FORWARD", 0.90),

    "walk back": ("BACKWARD", 0.95),
    "go back": ("BACKWARD", 0.95),
    "step back": ("BACKWARD", 0.95),
    "move back": ("BACKWARD", 0.95),
    "peeche": ("BACKWARD", 0.90),

    "turn left": ("LEFT", 0.95),
    "go left": ("LEFT", 0.90),
    "left": ("LEFT", 0.80),

    "turn right": ("RIGHT", 0.95),
    "go right": ("RIGHT", 0.90),
    "right": ("RIGHT", 0.80),

    # Gestures
    "wave": ("WAVE", 0.95),
    "wave at me": ("WAVE", 0.95),
    "say hi": ("WAVE", 0.90),
    "hello": ("WAVE", 0.70),  # lower confidence - could be greeting
    "bye": ("WAVE", 0.80),

    "shake hand": ("SHAKE_HAND", 0.95),
    "handshake": ("SHAKE_HAND", 0.95),
    "haath milao": ("SHAKE_HAND", 0.95),

    "hug": ("HUG", 0.95),
    "hug me": ("HUG", 0.95),
    "give me a hug": ("HUG", 0.95),

    "high five": ("HIGH_FIVE", 0.95),
    "give me five": ("HIGH_FIVE", 0.95),

    "shake head": ("HEADSHAKE", 0.95),
    "say no": ("HEADSHAKE", 0.90),

    "heart": ("HEART", 0.90),
    "make heart": ("HEART", 0.95),
    "love": ("HEART", 0.70),
}

# Question words that indicate conversation, not action
QUESTION_WORDS = [
    "what", "who", "where", "when", "why", "how",
    "which", "whose", "whom",
    "is it", "are you", "do you", "does it",
    "tell me about", "explain", "describe",
    "kya", "kaun", "kahan", "kab", "kyun", "kaise",
]

# Confirmation/Rejection patterns - flexible matching
# Instead of exact words, use patterns that capture intent
POSITIVE_PATTERNS = [
    # Direct yes
    "yes", "yeah", "yep", "yup", "ya", "ye",
    # Affirmative
    "ok", "okay", "sure", "confirm", "confirmed", "proceed", "go", "do it",
    "please", "go ahead", "alright", "right", "correct", "absolutely",
    "definitely", "certainly", "affirmative", "aye", "yea",
    # Hindi/Hinglish
    "haan", "han", "ha", "theek", "thik", "sahi", "bilkul", "zaroor",
    # Casual
    "fine", "cool", "done", "let's go", "let's do it", "why not"
]

NEGATIVE_PATTERNS = [
    # Direct no
    "no", "nope", "nah", "na",
    # Rejection
    "cancel", "stop", "don't", "dont", "never", "negative", "abort",
    "wait", "hold", "not now", "later", "skip",
    # Hindi/Hinglish
    "nahi", "mat", "ruk", "ruko", "band", "rehne do",
    # Casual
    "forget it", "nevermind", "never mind"
]


def is_question(transcript: str) -> bool:
    """Check if transcript is a question"""
    lower = transcript.lower().strip()

    # Check for question mark
    if "?" in transcript:
        return True

    # Check for question words at start
    for qw in QUESTION_WORDS:
        if lower.startswith(qw) or f" {qw} " in f" {lower} ":
            return True

    return False


def is_confirmation(transcript: str) -> bool:
    """Check if transcript is a confirmation - flexible pattern matching"""
    # Clean up - remove punctuation, lowercase
    lower = transcript.lower().strip()
    cleaned = lower.strip(string.punctuation)

    # Check each positive pattern
    for pattern in POSITIVE_PATTERNS:
        # Exact match
        if cleaned == pattern:
            return True
        # Pattern at start: "yes please", "sure thing"
        if cleaned.startswith(pattern + " ") or cleaned.startswith(pattern + ","):
            return True
        # Pattern at end: "I said yes", "that's okay"
        if cleaned.endswith(" " + pattern):
            return True
        # Pattern anywhere (as whole word)
        if f" {pattern} " in f" {cleaned} ":
            return True

    return False


def is_rejection(transcript: str) -> bool:
    """Check if transcript is a rejection - flexible pattern matching"""
    # Clean up - remove punctuation, lowercase
    lower = transcript.lower().strip()
    cleaned = lower.strip(string.punctuation)

    # Check each negative pattern
    for pattern in NEGATIVE_PATTERNS:
        # Exact match
        if cleaned == pattern:
            return True
        # Pattern at start: "no thanks", "cancel that"
        if cleaned.startswith(pattern + " ") or cleaned.startswith(pattern + ","):
            return True
        # Pattern at end: "I said no"
        if cleaned.endswith(" " + pattern):
            return True
        # Pattern anywhere (as whole word)
        if f" {pattern} " in f" {cleaned} ":
            return True

    return False


def match_action_pattern(transcript: str) -> Optional[Tuple[str, float]]:
    """
    Direct pattern matching for action detection
    Returns (action_name, confidence) or None
    """
    lower = transcript.lower().strip()

    # Try exact match first
    if lower in ACTION_PATTERNS:
        return ACTION_PATTERNS[lower]

    # Try substring match (sorted by length - longer patterns first)
    sorted_patterns = sorted(ACTION_PATTERNS.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if pattern in lower:
            action, conf = ACTION_PATTERNS[pattern]
            # Reduce confidence for substring match
            return (action, conf * 0.9)

    return None


def parse_intent_local(transcript: str) -> IntentResult:
    """
    Local intent parsing without LLM - fast pattern matching
    """
    lower = transcript.lower().strip()

    # Check if it's a question - always conversation
    if is_question(transcript):
        return IntentResult(
            intent_type=IntentType.CONVERSATION,
            action_name=None,
            confidence=0.95,
            requires_confirmation=False,
            reason="Detected question - treating as conversation",
            original_transcript=transcript
        )

    # Check for direct action pattern match
    match = match_action_pattern(transcript)
    if match:
        action_name, confidence = match

        # Determine if confirmation is needed based on action
        needs_confirm = action_name in [
            "INIT", "DAMP", "ZERO_TORQUE",  # HIGH risk
            "STANDUP", "SIT", "SQUAT", "READY",  # MEDIUM risk
            "FORWARD", "BACKWARD", "LEFT", "RIGHT"  # Motion
        ]

        return IntentResult(
            intent_type=IntentType.ACTION,
            action_name=action_name,
            confidence=confidence,
            requires_confirmation=needs_confirm,
            reason=f"Matched action pattern '{action_name}'",
            original_transcript=transcript
        )

    # Default to conversation
    return IntentResult(
        intent_type=IntentType.CONVERSATION,
        action_name=None,
        confidence=0.8,
        requires_confirmation=False,
        reason="No action pattern matched - treating as conversation",
        original_transcript=transcript
    )


def parse_intent_with_llm(transcript: str, ollama_host: str = "172.16.4.226",
                          ollama_port: int = 11434, model: str = "llama3.1:8b") -> IntentResult:
    """
    LLM-based intent parsing - uses Ollama for flexible understanding
    """
    # Skip LLM for simple responses (save latency)
    if is_question(transcript):
        return IntentResult(
            intent_type=IntentType.CONVERSATION,
            action_name=None,
            confidence=0.95,
            requires_confirmation=False,
            reason="Question detected",
            original_transcript=transcript
        )

    # NOTE: Confirmation/rejection words are handled by IntentReasoner.process()
    # before this function is called. If we reach here with "yes"/"no", it means
    # we're NOT awaiting confirmation, so treat as normal input for LLM.

    # Use LLM for intent parsing
    system_prompt = """You are an intent parser for a humanoid robot. Given user speech, determine if they want the robot to perform a physical action or have a conversation.

VALID ACTIONS (use EXACTLY these names):
- INIT: boot up, initialize, start up, turn on
- READY: get ready, ready mode, prepare
- STANDUP: stand up, get up, rise
- SIT: sit down, take a seat
- SQUAT: squat, crouch, duck
- DAMP: damp mode, relax, limp
- ZERO_TORQUE: disable motors, power off
- FORWARD: walk forward, go ahead, move forward
- BACKWARD: walk back, go back, step back
- LEFT: turn left, rotate left
- RIGHT: turn right, rotate right
- STOP: stop, halt, freeze
- WAVE: wave, wave hand, say hi, hello gesture
- SHAKE_HAND: shake hands, handshake
- HUG: hug, embrace
- HIGH_FIVE: high five, give me five
- HEADSHAKE: shake head, say no with head
- HEART: make heart shape, heart gesture

RULES:
1. Physical action request → type: "action", action: "ACTION_NAME"
2. Question or conversation → type: "conversation", action: null
3. If unsure → type: "conversation" (safer default)
4. Confidence: 0.9+ for clear intent, 0.7-0.9 for likely, below 0.7 for unsure

OUTPUT ONLY VALID JSON (no markdown, no extra text):
{"type": "action" or "conversation", "action": "ACTION_NAME" or null, "confidence": 0.0-1.0, "reason": "brief reason"}"""

    try:
        url = f"http://{ollama_host}:{ollama_port}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }

        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 200:
            result = response.json()
            content = result.get("message", {}).get("content", "")

            # Parse JSON from response
            try:
                # Clean up response (remove markdown if present)
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                parsed = json.loads(content)

                intent_type = IntentType.ACTION if parsed.get("type") == "action" else IntentType.CONVERSATION
                action_name = parsed.get("action")
                confidence = float(parsed.get("confidence", 0.5))
                reason = parsed.get("reason", "LLM classification")

                # Validate action name
                valid_actions = ["INIT", "READY", "STANDUP", "SIT", "SQUAT", "DAMP", "ZERO_TORQUE",
                               "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP",
                               "WAVE", "SHAKE_HAND", "HUG", "HIGH_FIVE", "HEADSHAKE", "HEART"]

                # Normalize action name to uppercase
                if action_name:
                    action_name = action_name.upper()

                if action_name and action_name not in valid_actions:
                    action_name = None
                    intent_type = IntentType.CONVERSATION

                needs_confirm = action_name in [
                    "INIT", "DAMP", "ZERO_TORQUE",
                    "STANDUP", "SIT", "SQUAT", "READY",
                    "FORWARD", "BACKWARD", "LEFT", "RIGHT"
                ] if action_name else False

                return IntentResult(
                    intent_type=intent_type,
                    action_name=action_name,
                    confidence=confidence,
                    requires_confirmation=needs_confirm,
                    reason=reason,
                    original_transcript=transcript
                )

            except json.JSONDecodeError:
                logger.warning(f"[INTENT] Failed to parse LLM JSON: {content}")

    except Exception as e:
        logger.warning(f"[INTENT] LLM call failed: {e}")

    # Fallback: try local pattern matching if LLM fails
    local_result = parse_intent_local(transcript)
    if local_result.confidence >= 0.8:
        return local_result

    # Default to conversation if everything fails
    return IntentResult(
        intent_type=IntentType.CONVERSATION,
        action_name=None,
        confidence=0.5,
        requires_confirmation=False,
        reason="LLM unavailable - defaulting to conversation",
        original_transcript=transcript
    )


class IntentReasoner:
    """
    Intent Reasoner class - manages intent parsing with state
    Tracks pending confirmations and conversation context
    """

    def __init__(self, ollama_host: str = "172.16.4.226", ollama_port: int = 11434,
                 model: str = "llama3.1:8b", use_llm: bool = False):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.model = model
        self.use_llm = use_llm

        # State
        self.pending_action: Optional[str] = None
        self.pending_transcript: Optional[str] = None
        self.awaiting_confirmation = False

    def process(self, transcript: str) -> IntentResult:
        """
        Process transcript and return intent
        Handles confirmation state
        """
        # Debug - state check
        print(f"[DEBUG] process() called: transcript='{transcript}', awaiting={self.awaiting_confirmation}, pending={self.pending_action}")

        # PRIORITY 1: Check if awaiting confirmation (instant response, no LLM)
        if self.awaiting_confirmation and self.pending_action:
            if is_confirmation(transcript):
                action = self.pending_action
                self.clear_pending()
                return IntentResult(
                    intent_type=IntentType.ACTION,
                    action_name=action,
                    confidence=1.0,
                    requires_confirmation=False,
                    reason="User confirmed action",
                    original_transcript=transcript
                )
            elif is_rejection(transcript):
                self.clear_pending()
                return IntentResult(
                    intent_type=IntentType.CONVERSATION,
                    action_name=None,
                    confidence=1.0,
                    requires_confirmation=False,
                    reason="User cancelled action",
                    original_transcript=transcript
                )
            else:
                # Unclear - cancel pending and continue to parse normally
                self.clear_pending()

        # Parse intent
        if self.use_llm:
            result = parse_intent_with_llm(transcript, self.ollama_host,
                                           self.ollama_port, self.model)
        else:
            result = parse_intent_local(transcript)

        # If action needs confirmation, store it
        if result.intent_type == IntentType.ACTION and result.requires_confirmation:
            self.pending_action = result.action_name
            self.pending_transcript = transcript
            self.awaiting_confirmation = True

        return result

    def clear_pending(self):
        """Clear pending confirmation state"""
        self.pending_action = None
        self.pending_transcript = None
        self.awaiting_confirmation = False

    def get_confirmation_prompt(self) -> Optional[str]:
        """Get confirmation prompt for pending action"""
        if not self.pending_action:
            return None

        prompts = {
            "INIT": "Run full boot sequence? Say yes to confirm.",
            "READY": "Switch to ready mode? Say yes to confirm.",
            "STANDUP": "Stand up? Say yes to confirm.",
            "SIT": "Sit down? Say yes to confirm.",
            "SQUAT": "Squat down? Say yes to confirm.",
            "DAMP": "Enter damp mode? Robot will go limp. Say yes to confirm.",
            "ZERO_TORQUE": "Zero torque mode? Robot may fall! Say yes to confirm.",
            "FORWARD": "Walk forward? Say yes to confirm.",
            "BACKWARD": "Walk backward? Say yes to confirm.",
            "LEFT": "Turn left? Say yes to confirm.",
            "RIGHT": "Turn right? Say yes to confirm.",
        }
        return prompts.get(self.pending_action, f"Execute {self.pending_action}? Say yes to confirm.")


# === TEST ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_phrases = [
        "initialize",
        "init",
        "boot up",
        "get ready",
        "stand up",
        "wave at me",
        "walk forward",
        "what can you do?",
        "tell me about yourself",
        "hello",
        "yes",
        "haan",
        "no",
        "turn left",
        "damp mode",
    ]

    print("=== Intent Reasoner Test (Local) ===\n")

    reasoner = IntentReasoner(use_llm=False)

    for phrase in test_phrases:
        result = reasoner.process(phrase)

        if result.intent_type == IntentType.ACTION:
            status = f"🤖 ACTION: {result.action_name}"
            if result.requires_confirmation:
                status += " (needs confirm)"
                print(f"   Prompt: {reasoner.get_confirmation_prompt()}")
                reasoner.clear_pending()  # Clear for next test
        else:
            status = "💬 CONVERSATION"

        print(f"{status}: '{phrase}' (conf: {result.confidence:.2f})")
        print(f"   Reason: {result.reason}")
        print()
