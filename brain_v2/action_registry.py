# action_registry.py
# G1 Robot Action Registry - saari actions ki definitions with metadata
# Risk levels: LOW (immediate), MEDIUM (confirmation needed), HIGH (extra validation)

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Set

class RiskLevel(Enum):
    LOW = "low"          # gestures - immediate execution
    MEDIUM = "medium"    # locomotion - needs confirmation
    HIGH = "high"        # system commands - extra validation

class BodyPart(Enum):
    ARM = "arm"
    LEGS = "legs"
    WAIST = "waist"
    FULL = "full"
    NONE = "none"

class ActionType(Enum):
    GESTURE = "gesture"
    MOTION = "motion"
    POSTURE = "posture"
    SYSTEM = "system"
    MODE = "mode"
    QUERY = "query"

@dataclass
class Action:
    name: str
    action_type: ActionType
    risk_level: RiskLevel
    body_part: BodyPart
    required_fsm: Set[int]  # FSM states jahan ye action valid hai
    api_id: Optional[str]   # loco_client API ID ya custom
    example_phrases: List[str]
    confirmation_prompt: Optional[str] = None  # MEDIUM/HIGH risk ke liye
    rejection_message: Optional[str] = None    # FSM mismatch ke liye

# FSM State Constants - g1_loco_client.hpp se
FSM_ZERO_TORQUE = 0
FSM_DAMP = 1
FSM_SQUAT = 2
FSM_SIT = 3
FSM_STANDUP = 4
FSM_START = 500
FSM_READY = 801

# Valid FSM sets - commonly used combinations
FSM_ANY = {FSM_ZERO_TORQUE, FSM_DAMP, FSM_SQUAT, FSM_SIT, FSM_STANDUP, FSM_START, FSM_READY}
FSM_STANDING = {FSM_START, FSM_READY}
FSM_READY_ONLY = {FSM_READY}
FSM_CAN_STANDUP = {FSM_DAMP, FSM_STANDUP}

# === ACTION REGISTRY ===
# Ye dictionary main source of truth hai - Gatekeeper isse check karega

ACTION_REGISTRY = {
    # --- GESTURES (LOW Risk) ---
    "WAVE": Action(
        name="WAVE",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.ARM,
        required_fsm=FSM_READY_ONLY,
        api_id="7106:0",  # WaveHand(false) - task_id 0
        example_phrases=["wave", "wave at me", "say hi", "hello", "bye bye"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "SHAKE_HAND": Action(
        name="SHAKE_HAND",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.ARM,
        required_fsm=FSM_READY_ONLY,
        api_id="7106:2",  # ShakeHand(0) - task_id 2
        example_phrases=["shake hand", "handshake", "nice to meet you", "let's shake hands"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "HUG": Action(
        name="HUG",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.ARM,
        required_fsm=FSM_READY_ONLY,
        api_id="7108:hug",  # ExecuteCustomAction("hug")
        example_phrases=["hug", "give me a hug", "embrace", "hug me"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "HIGH_FIVE": Action(
        name="HIGH_FIVE",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.ARM,
        required_fsm=FSM_READY_ONLY,
        api_id="7108:high_five",
        example_phrases=["high five", "give me five", "up top"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "HEADSHAKE": Action(
        name="HEADSHAKE",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.WAIST,
        required_fsm=FSM_READY_ONLY,
        api_id="7108:headshake",
        example_phrases=["shake head", "say no", "disagree", "nope"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "HEART": Action(
        name="HEART",
        action_type=ActionType.GESTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.ARM,
        required_fsm=FSM_READY_ONLY,
        api_id="custom:heart",
        example_phrases=["heart", "make heart", "love", "show love"],
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    # --- MOTION (MEDIUM Risk - Confirmation Required) ---
    "FORWARD": Action(
        name="FORWARD",
        action_type=ActionType.MOTION,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_STANDING,
        api_id="7105",  # Move(vx=0.3, vy=0, vyaw=0)
        example_phrases=["walk forward", "go ahead", "move forward", "go forward"],
        confirmation_prompt="Walk forward? Say yes to confirm.",
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "BACKWARD": Action(
        name="BACKWARD",
        action_type=ActionType.MOTION,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_STANDING,
        api_id="7105",  # Move(vx=-0.3, vy=0, vyaw=0)
        example_phrases=["walk back", "go back", "step back", "move backward"],
        confirmation_prompt="Walk backward? Say yes to confirm.",
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "LEFT": Action(
        name="LEFT",
        action_type=ActionType.MOTION,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_STANDING,
        api_id="7105",  # Move(vx=0, vy=0, vyaw=0.3)
        example_phrases=["turn left", "go left", "rotate left"],
        confirmation_prompt="Turn left? Say yes to confirm.",
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "RIGHT": Action(
        name="RIGHT",
        action_type=ActionType.MOTION,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_STANDING,
        api_id="7105",  # Move(vx=0, vy=0, vyaw=-0.3)
        example_phrases=["turn right", "go right", "rotate right"],
        confirmation_prompt="Turn right? Say yes to confirm.",
        rejection_message="I need to stand up first. Say 'stand up'."
    ),

    "STOP": Action(
        name="STOP",
        action_type=ActionType.MOTION,
        risk_level=RiskLevel.LOW,  # STOP is always safe - no confirmation needed
        body_part=BodyPart.LEGS,
        required_fsm=FSM_ANY,
        api_id="7105:stop",  # StopMove()
        example_phrases=["stop", "halt", "freeze", "ruk jao", "bas"],
    ),

    # --- POSTURE (MEDIUM Risk) ---
    "STANDUP": Action(
        name="STANDUP",
        action_type=ActionType.POSTURE,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.FULL,
        required_fsm=FSM_CAN_STANDUP,
        api_id="7101:4",  # SetFsmId(4)
        example_phrases=["stand up", "get up", "utho", "khade ho jao"],
        confirmation_prompt="Stand up? Say yes to confirm.",
        rejection_message="I'm already standing or can't stand up from this state."
    ),

    "SIT": Action(
        name="SIT",
        action_type=ActionType.POSTURE,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.FULL,
        required_fsm=FSM_READY_ONLY,
        api_id="7101:3",  # SetFsmId(3)
        example_phrases=["sit down", "take a seat", "baith jao"],
        confirmation_prompt="Sit down? Say yes to confirm.",
        rejection_message="Can't sit down safely from this state."
    ),

    "SQUAT": Action(
        name="SQUAT",
        action_type=ActionType.POSTURE,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_READY_ONLY,
        api_id="7101:2",  # SetFsmId(2)
        example_phrases=["squat", "crouch", "jhuko"],
        confirmation_prompt="Squat down? Say yes to confirm.",
        rejection_message="Can't squat from this state."
    ),

    "HIGH_STAND": Action(
        name="HIGH_STAND",
        action_type=ActionType.POSTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_READY_ONLY,
        api_id="7104:high",  # HighStand()
        example_phrases=["stand tall", "full height", "seedha khade ho"],
    ),

    "LOW_STAND": Action(
        name="LOW_STAND",
        action_type=ActionType.POSTURE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.LEGS,
        required_fsm=FSM_READY_ONLY,
        api_id="7104:low",  # LowStand()
        example_phrases=["stand low", "lower", "neecha ho jao"],
    ),

    # --- SYSTEM (HIGH Risk) ---
    "INIT": Action(
        name="INIT",
        action_type=ActionType.SYSTEM,
        risk_level=RiskLevel.HIGH,
        body_part=BodyPart.FULL,
        required_fsm=FSM_ANY,  # Can init from any state
        api_id="orchestrator:init",  # Full boot sequence
        example_phrases=["initialize", "init", "boot up", "start up", "chalu karo", "shuru karo"],
        confirmation_prompt="Run full boot sequence? DAMP, stand up, then ready. Say yes to confirm.",
    ),

    "READY": Action(
        name="READY",
        action_type=ActionType.SYSTEM,
        risk_level=RiskLevel.MEDIUM,
        body_part=BodyPart.FULL,
        required_fsm={FSM_STANDUP, FSM_START},  # Can go ready from standup(4) or start(500)
        api_id="7101:801",  # SetFsmId(801)
        example_phrases=["ready", "get ready", "ready mode", "taiyaar ho jao"],
        confirmation_prompt="Switch to ready mode? Say yes to confirm.",
        rejection_message="Stand up first, then I can switch to ready mode."
    ),

    "DAMP": Action(
        name="DAMP",
        action_type=ActionType.SYSTEM,
        risk_level=RiskLevel.HIGH,
        body_part=BodyPart.FULL,
        required_fsm=FSM_ANY,
        api_id="7101:1",  # SetFsmId(1)
        example_phrases=["damp mode", "relax", "damping", "aaraam karo"],
        confirmation_prompt="Enter damp mode? Robot will go limp. Say yes to confirm.",
    ),

    "ZERO_TORQUE": Action(
        name="ZERO_TORQUE",
        action_type=ActionType.SYSTEM,
        risk_level=RiskLevel.HIGH,
        body_part=BodyPart.FULL,
        required_fsm=FSM_ANY,
        api_id="7101:0",  # SetFsmId(0)
        example_phrases=["disable", "limp mode", "zero torque", "power off motors"],
        confirmation_prompt="Zero torque mode? Robot may fall! Say yes to confirm.",
    ),

    # --- MODE (Special) ---
    "TALK_MODE": Action(
        name="TALK_MODE",
        action_type=ActionType.MODE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.NONE,
        required_fsm=FSM_ANY,
        api_id="mode:talk",
        example_phrases=["talk mode", "let's chat", "conversation mode", "baat karte hain"],
    ),

    "STOP_TALKING": Action(
        name="STOP_TALKING",
        action_type=ActionType.MODE,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.NONE,
        required_fsm=FSM_ANY,
        api_id="mode:stop_talk",
        example_phrases=["stop talking", "shut up", "be quiet", "chup ho jao"],
    ),

    # --- QUERY (Information requests) ---
    "BATTERY": Action(
        name="BATTERY",
        action_type=ActionType.QUERY,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.NONE,
        required_fsm=FSM_ANY,
        api_id="query:battery",
        example_phrases=["battery", "battery level", "how much charge", "kitni battery hai"],
    ),

    "ABOUT_SELF": Action(
        name="ABOUT_SELF",
        action_type=ActionType.QUERY,
        risk_level=RiskLevel.LOW,
        body_part=BodyPart.NONE,
        required_fsm=FSM_ANY,
        api_id="query:about",
        example_phrases=["who are you", "about yourself", "introduce yourself", "tum kaun ho"],
    ),
}

# === UTILITY FUNCTIONS ===

def get_action(action_name: str) -> Optional[Action]:
    """Registry se action fetch karo"""
    return ACTION_REGISTRY.get(action_name.upper())

def is_valid_action(action_name: str) -> bool:
    """Check karo ki action registry me hai ya nahi"""
    return action_name.upper() in ACTION_REGISTRY

def get_all_action_names() -> List[str]:
    """Saari action names ki list"""
    return list(ACTION_REGISTRY.keys())

def get_actions_by_risk(risk_level: RiskLevel) -> List[Action]:
    """Risk level ke hisaab se actions filter karo"""
    return [a for a in ACTION_REGISTRY.values() if a.risk_level == risk_level]

def get_actions_by_type(action_type: ActionType) -> List[Action]:
    """Action type ke hisaab se filter karo"""
    return [a for a in ACTION_REGISTRY.values() if a.action_type == action_type]

def requires_confirmation(action_name: str) -> bool:
    """Check karo ki action ko confirmation chahiye ya nahi"""
    action = get_action(action_name)
    if not action:
        return True  # unknown action - safer to require confirmation
    return action.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]

def is_fsm_compatible(action_name: str, current_fsm: int) -> bool:
    """Check karo ki current FSM state me ye action valid hai ya nahi"""
    action = get_action(action_name)
    if not action:
        return False
    return current_fsm in action.required_fsm

def get_valid_actions_str() -> str:
    """Intent Reasoner ke liye valid actions ki comma-separated string"""
    return ", ".join(ACTION_REGISTRY.keys())


# === DEBUG/TEST ===
if __name__ == "__main__":
    print("=== G1 Action Registry ===\n")

    print("LOW Risk (Immediate):")
    for a in get_actions_by_risk(RiskLevel.LOW):
        print(f"  - {a.name}: {a.example_phrases[0]}")

    print("\nMEDIUM Risk (Confirmation):")
    for a in get_actions_by_risk(RiskLevel.MEDIUM):
        print(f"  - {a.name}: {a.example_phrases[0]}")

    print("\nHIGH Risk (Extra Validation):")
    for a in get_actions_by_risk(RiskLevel.HIGH):
        print(f"  - {a.name}: {a.example_phrases[0]}")

    print(f"\nTotal actions: {len(ACTION_REGISTRY)}")
    print(f"\nValid actions string:\n{get_valid_actions_str()}")
