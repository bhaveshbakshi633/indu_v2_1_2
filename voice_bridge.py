# voice_bridge.py
# ROS2 bridge for voice-driven robot control
# brain_v2 (172.16.6.19) se G1 (172.16.2.242) tak communication

import logging
import threading
import time
import json
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ROS2 availability check - gracefully handle if not installed
ROS2_AVAILABLE = False
rclpy = None
Node = None
Bool = None
String = None
QoSProfile = None
ReliabilityPolicy = None
DurabilityPolicy = None

try:
    import rclpy as _rclpy
    from rclpy.node import Node as _Node
    from rclpy.qos import QoSProfile as _QoSProfile
    from rclpy.qos import ReliabilityPolicy as _ReliabilityPolicy
    from rclpy.qos import DurabilityPolicy as _DurabilityPolicy
    from std_msgs.msg import Bool as _Bool, String as _String

    # Assign to module-level variables
    rclpy = _rclpy
    Node = _Node
    Bool = _Bool
    String = _String
    QoSProfile = _QoSProfile
    ReliabilityPolicy = _ReliabilityPolicy
    DurabilityPolicy = _DurabilityPolicy

    ROS2_AVAILABLE = True
    logger.info("[VOICE_BRIDGE] ROS2 (rclpy) available")
except ImportError as e:
    logger.warning(f"[VOICE_BRIDGE] ROS2 (rclpy) not available: {e}")
    logger.warning("[VOICE_BRIDGE] Running in MOCK mode - no actual G1 communication")


@dataclass
class VoiceActionRequest:
    """Voice action request data structure"""
    transcript: str           # Raw STT text
    action_name: str          # Parsed action from LLM
    confidence: float         # LLM confidence 0.0-1.0
    confirmed: bool           # User confirmed (MEDIUM/HIGH risk)
    timestamp_ns: int         # Request creation time


class VoiceBridgeMock:
    """
    Mock implementation jab ROS2 available nahi hai
    Testing aur development ke liye useful
    """

    def __init__(self):
        self._pending_operations = []
        self._response_callback = None
        print("[VOICE_BRIDGE] ⚠️ Running in MOCK mode - no actual ROS2/G1 communication")

    def start(self):
        print("[VOICE_BRIDGE_MOCK] Started (mock mode)")

    def stop(self):
        print("[VOICE_BRIDGE_MOCK] Stopped")

    def send_emergency_stop(self):
        """Mock emergency stop - just logs"""
        print("[VOICE_BRIDGE_MOCK] 🛑 EMERGENCY STOP (mock - not sent to G1)")

    def send_action_request(self, request: VoiceActionRequest):
        """Mock action request - just logs"""
        print(f"[VOICE_BRIDGE_MOCK] 🤖 Action request: {request.action_name} (conf: {request.confidence}) - NOT sent to G1")
        self._pending_operations.append(request)

    def cancel_pending(self):
        """Cancel pending operations"""
        count = len(self._pending_operations)
        self._pending_operations.clear()
        print(f"[VOICE_BRIDGE_MOCK] Cancelled {count} pending operations")

    def set_response_callback(self, callback: Callable):
        """Set callback for gatekeeper responses"""
        self._response_callback = callback

    def is_connected(self) -> bool:
        return False


# Only define ROS2 class if ROS2 is available
if ROS2_AVAILABLE:
    class VoiceBridgeROS2(Node):
        """
        ROS2 Node for voice-driven robot control
        Topics:
          - Publishes: /emergency_stop (Bool), /voice/action_request (String/JSON)
          - Subscribes: /voice/gatekeeper_response (String/JSON)
        """

        def __init__(self):
            super().__init__('voice_bridge')

            # QoS profile for reliable communication
            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=10
            )

            # Emergency stop - highest priority, reliable delivery
            emergency_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=1
            )

            # Publishers
            self._emergency_stop_pub = self.create_publisher(
                Bool,
                '/emergency_stop',
                emergency_qos
            )

            # Action request publisher - JSON in String message
            self._action_request_pub = self.create_publisher(
                String,
                '/voice/action_request',
                reliable_qos
            )

            # Response subscriber
            self._response_callback = None
            self._response_sub = self.create_subscription(
                String,
                '/voice/gatekeeper_response',
                self._on_gatekeeper_response,
                reliable_qos
            )

            # State tracking
            self._pending_operations = []
            self._connected = True
            self._last_stop_time = 0

            self.get_logger().info('[VOICE_BRIDGE] ROS2 node initialized')

        def send_emergency_stop(self):
            """
            Emergency stop publish karo - DAMP immediately
            Rate limiting: min 100ms between stops (prevent spam)
            """
            current_time = time.time()
            if current_time - self._last_stop_time < 0.1:
                return  # Rate limit

            self._last_stop_time = current_time

            msg = Bool()
            msg.data = True
            self._emergency_stop_pub.publish(msg)
            self.get_logger().warn('[VOICE_BRIDGE] 🛑 EMERGENCY STOP sent to G1')

        def send_action_request(self, request: VoiceActionRequest):
            """
            Action request G1 ko bhejo
            JSON format me serialize karke String msg me
            """
            msg_data = {
                'transcript': request.transcript,
                'action_name': request.action_name,
                'confidence': request.confidence,
                'confirmed': request.confirmed,
                'timestamp_ns': request.timestamp_ns
            }

            msg = String()
            msg.data = json.dumps(msg_data)
            self._action_request_pub.publish(msg)
            self._pending_operations.append(request)

            self.get_logger().info(
                f'[VOICE_BRIDGE] Action request sent: {request.action_name} '
                f'(conf: {request.confidence:.2f})'
            )

        def cancel_pending(self):
            """Pending operations cancel karo"""
            count = len(self._pending_operations)
            self._pending_operations.clear()
            self.get_logger().info(f'[VOICE_BRIDGE] Cancelled {count} pending operations')

        def set_response_callback(self, callback: Callable):
            """Gatekeeper response ke liye callback set karo"""
            self._response_callback = callback

        def _on_gatekeeper_response(self, msg: String):
            """Gatekeeper response handle karo"""
            try:
                response = json.loads(msg.data)
                self.get_logger().info(
                    f'[VOICE_BRIDGE] Gatekeeper response: {response.get("action_name")} - '
                    f'{"APPROVED" if response.get("approved") else "REJECTED"}'
                )

                if self._response_callback:
                    self._response_callback(response)

            except json.JSONDecodeError as e:
                self.get_logger().error(f'[VOICE_BRIDGE] Invalid response JSON: {e}')

        def is_connected(self) -> bool:
            return self._connected


class VoiceBridge:
    """
    Voice Bridge wrapper - ROS2 available ho toh use karo, nahi toh mock
    Thread-safe spinning ke saath
    """

    def __init__(self):
        self._node = None
        self._spin_thread = None
        self._running = False
        self._use_ros2 = ROS2_AVAILABLE

        if not self._use_ros2:
            self._mock = VoiceBridgeMock()

    def start(self):
        """Bridge start karo - ROS2 spinning shuru karo"""
        if self._use_ros2:
            try:
                rclpy.init()
                self._node = VoiceBridgeROS2()
                self._running = True

                # Separate thread me spin karo
                self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
                self._spin_thread.start()

                print("[VOICE_BRIDGE] ✅ ROS2 bridge started - connected to G1")
            except Exception as e:
                print(f"[VOICE_BRIDGE] ❌ Failed to start ROS2: {e}")
                self._use_ros2 = False
                self._mock = VoiceBridgeMock()
                self._mock.start()
        else:
            self._mock.start()

    def stop(self):
        """Bridge stop karo - cleanup"""
        self._running = False

        if self._use_ros2 and self._node:
            self._node.destroy_node()
            rclpy.shutdown()
            print("[VOICE_BRIDGE] ROS2 bridge stopped")
        elif not self._use_ros2:
            self._mock.stop()

    def _spin_loop(self):
        """ROS2 spin loop - separate thread me"""
        while self._running and self._node:
            try:
                rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception as e:
                print(f"[VOICE_BRIDGE] Spin error: {e}")
                break

    def send_emergency_stop(self):
        """Emergency stop bhejo"""
        if self._use_ros2 and self._node:
            self._node.send_emergency_stop()
        else:
            self._mock.send_emergency_stop()

    def send_action_request(self, request: VoiceActionRequest):
        """Action request bhejo"""
        if self._use_ros2 and self._node:
            self._node.send_action_request(request)
        else:
            self._mock.send_action_request(request)

    def cancel_pending(self):
        """Pending operations cancel karo"""
        if self._use_ros2 and self._node:
            self._node.cancel_pending()
        else:
            self._mock.cancel_pending()

    def set_response_callback(self, callback: Callable):
        """Gatekeeper response callback set karo"""
        if self._use_ros2 and self._node:
            self._node.set_response_callback(callback)
        else:
            self._mock.set_response_callback(callback)

    def is_connected(self) -> bool:
        """Check if connected to ROS2 network"""
        if self._use_ros2 and self._node:
            return self._node.is_connected()
        return False


# Singleton instance
_voice_bridge_instance: Optional[VoiceBridge] = None


def get_voice_bridge() -> VoiceBridge:
    """Get or create singleton VoiceBridge instance"""
    global _voice_bridge_instance
    if _voice_bridge_instance is None:
        _voice_bridge_instance = VoiceBridge()
    return _voice_bridge_instance


# === TEST ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== Voice Bridge Test ===\n")

    bridge = get_voice_bridge()
    bridge.start()

    print(f"ROS2 available: {ROS2_AVAILABLE}")
    print(f"Connected: {bridge.is_connected()}")

    # Test emergency stop
    print("\nSending emergency stop...")
    bridge.send_emergency_stop()

    # Test action request
    print("\nSending action request...")
    request = VoiceActionRequest(
        transcript="wave at me",
        action_name="WAVE",
        confidence=0.95,
        confirmed=False,
        timestamp_ns=int(time.time() * 1e9)
    )
    bridge.send_action_request(request)

    # Wait a bit
    time.sleep(2)

    bridge.stop()
    print("\nTest complete")
