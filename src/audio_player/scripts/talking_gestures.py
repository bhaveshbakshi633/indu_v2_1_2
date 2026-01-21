#!/usr/bin/env python3
"""
Talking Gestures Node - Triggers explain1-5 gestures while robot is speaking

Features:
- Subscribes to /g1/tts/speaking for speaking status
- Subscribes to /g1/tts/speech_info for text + is_confirmation
- SKIPS gestures for confirmation prompts (is_confirmation=True)
- DETECTS "namaste" in text and does namaste gesture first
- Cycles through explain1-5 while speaking normally
"""

import threading
import time
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from unitree_api.msg import Request


# Explain actions list - cycle through while speaking
EXPLAIN_ACTIONS = ["explain1", "explain2", "explain3", "explain4", "explain5"]

# Namaste actions - use namaste1 by default
NAMASTE_ACTION = "namaste1"
NAMASTE_DURATION = 3.5  # seconds to wait for namaste to complete

# API ID for custom actions
API_ID_ARM_CUSTOM_ACTION = 7108

# Time between explain actions (seconds)
ACTION_INTERVAL = 3.5


class TalkingGesturesNode(Node):
    """Node jo speaking ke time explain gestures trigger karta hai"""

    def __init__(self):
        super().__init__('talking_gestures')

        # Publisher for arm custom actions
        self.arm_request_pub = self.create_publisher(
            Request,
            '/api/arm/request',
            10
        )

        # Subscriber for speaking status
        self.speaking_sub = self.create_subscription(
            Bool,
            '/g1/tts/speaking',
            self.speaking_callback,
            10
        )

        # Subscriber for speech info (text + is_confirmation)
        self.speech_info_sub = self.create_subscription(
            String,
            '/g1/tts/speech_info',
            self.speech_info_callback,
            10
        )

        # Subscriber for gesture mode (from brain_v2 via audio_receiver)
        self.gesture_mode_sub = self.create_subscription(
            String,
            '/g1/gesture/mode',
            self.gesture_mode_callback,
            10
        )

        # State variables
        self.is_speaking = False
        self.gesture_thread = None
        self.stop_gestures = threading.Event()
        self.action_index = 0
        self.request_id = 0

        # Gesture mode - DEFAULT DISABLED (only brain_v2 conversations enable)
        self.gesture_enabled = False
        self.namaste_pending = False

        # Speech info state (for confirmation detection)
        self.current_text = ""
        self.is_confirmation = False

        self.get_logger().info('Talking Gestures Node initialized')
        self.get_logger().info(f'Explain actions: {EXPLAIN_ACTIONS}')
        self.get_logger().info(f'Namaste action: {NAMASTE_ACTION}')
        self.get_logger().info(f'Interval: {ACTION_INTERVAL}s')

    def speech_info_callback(self, msg):
        """Speech info receive karo - sirf confirmation detection ke liye"""
        try:
            data = json.loads(msg.data)
            self.current_text = data.get('text', '')
            self.is_confirmation = data.get('is_confirmation', False)

            self.get_logger().info(
                f"Speech info: confirmation={self.is_confirmation}, "
                f"text='{self.current_text[:50]}...'"
            )

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Speech info JSON error: {e}")
        except Exception as e:
            self.get_logger().error(f"Speech info error: {e}")

    def gesture_mode_callback(self, msg):
        """Gesture mode receive karo - brain_v2 se conversation ke liye aata hai"""
        try:
            data = json.loads(msg.data)
            self.gesture_enabled = data.get('enable', False)
            self.namaste_pending = data.get('namaste', False)
            self.get_logger().info(
                f"Gesture mode: enabled={self.gesture_enabled}, namaste={self.namaste_pending}"
            )
        except Exception as e:
            self.get_logger().error(f"Gesture mode parse error: {e}")

    def speaking_callback(self, msg):
        """Speaking status change handle karo"""
        was_speaking = self.is_speaking
        self.is_speaking = msg.data

        if self.is_speaking and not was_speaking:
            # Speaking started
            self.get_logger().info(
                f"Speaking started - gesture_enabled={self.gesture_enabled}, "
                f"confirmation={self.is_confirmation}"
            )

            # SKIP if gestures not enabled (system announcements, etc.)
            if not self.gesture_enabled:
                self.get_logger().info("Gestures disabled - skipping")
                return

            # SKIP for confirmation prompts
            if self.is_confirmation:
                self.get_logger().info("Confirmation prompt - skipping gestures")
                return

            # Start gesture handling
            self.start_gesture_loop()

        elif not self.is_speaking and was_speaking:
            # Speaking stopped - stop gesture loop
            self.get_logger().info('Speaking stopped - stopping gesture loop')
            self.stop_gesture_loop()

            # Reset ALL state for next speech
            self.gesture_enabled = False
            self.namaste_pending = False
            self.current_text = ""
            self.is_confirmation = False

    def start_gesture_loop(self):
        """Gesture loop start karo background thread me"""
        self.stop_gestures.clear()
        self.action_index = 0

        if self.gesture_thread is not None and self.gesture_thread.is_alive():
            return  # Already running

        self.gesture_thread = threading.Thread(target=self._gesture_loop, daemon=True)
        self.gesture_thread.start()

    def stop_gesture_loop(self):
        """Gesture loop stop karo"""
        self.stop_gestures.set()

    def _gesture_loop(self):
        """Background thread - namaste (if needed) then explain actions cycle karo"""

        # Step 1: Namaste pehle (agar text me "namaste" hai)
        if self.namaste_pending:
            self.get_logger().info(f"Doing namaste first: {NAMASTE_ACTION}")
            self.execute_custom_action(NAMASTE_ACTION)

            # Wait for namaste to complete
            for _ in range(int(NAMASTE_DURATION * 10)):
                if self.stop_gestures.is_set():
                    return
                time.sleep(0.1)

            self.namaste_pending = False
            self.get_logger().info("Namaste complete, starting explain loop")

        # Step 2: Explain actions loop
        while not self.stop_gestures.is_set():
            # Current action execute karo
            action_name = EXPLAIN_ACTIONS[self.action_index]
            self.execute_custom_action(action_name)

            # Next action
            self.action_index = (self.action_index + 1) % len(EXPLAIN_ACTIONS)

            # Wait before next action
            # Use small intervals to check stop flag frequently
            for _ in range(int(ACTION_INTERVAL * 10)):
                if self.stop_gestures.is_set():
                    break
                time.sleep(0.1)

    def execute_custom_action(self, action_name):
        """API 7108 se custom action execute karo"""
        try:
            self.request_id += 1

            req = Request()
            req.header.identity.id = self.request_id
            req.header.identity.api_id = API_ID_ARM_CUSTOM_ACTION

            # JSON parameter with action name
            params = {"action_name": action_name}
            req.parameter = json.dumps(params)

            self.arm_request_pub.publish(req)
            self.get_logger().info(f'Executed: {action_name}')

        except Exception as e:
            self.get_logger().error(f'Action error: {e}')


def main(args=None):
    rclpy.init(args=args)

    try:
        node = TalkingGesturesNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
