#!/usr/bin/env python3
# HTTP-to-ROS2 bridge for voice commands
# HTTP se action aata hai, orchestrator ko ActionCommand publish karta hai

import rclpy
from rclpy.node import Node
from orchestrator_msgs.msg import ActionCommand
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)
ros_node = None

# Allowed actions - whitelist for safety
ALLOWED_ACTIONS = [
    "INIT", "READY", "DAMP", "ZERO_TORQUE",
    "STANDUP", "SIT", "SQUAT", "HIGH_STAND", "LOW_STAND",
    "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP",
    "WAVE", "SHAKE_HAND", "HUG", "HIGH_FIVE", "HEADSHAKE", "HEART",
    "TALK_MODE", "STOP_TALKING"
]

# Action name mapping (lowercase for orchestrator)
ACTION_MAP = {
    "INIT": "init",
    "READY": "ready",
    "DAMP": "damp",
    "ZERO_TORQUE": "zero_torque",
    "STANDUP": "standup",
    "SIT": "sit",
    "SQUAT": "squat",
    "HIGH_STAND": "high_stand",
    "LOW_STAND": "low_stand",
    "FORWARD": "forward",
    "BACKWARD": "backward",
    "LEFT": "left",
    "RIGHT": "right",
    "STOP": "stop",
    "WAVE": "wave",
    "SHAKE_HAND": "shake_hand",
    "HUG": "hug",
    "HIGH_FIVE": "high_five",
    "HEADSHAKE": "headshake",
    "HEART": "heart",
    "TALK_MODE": "talk_mode",
    "STOP_TALKING": "stop_talking"
}

@app.route('/action', methods=['POST'])
def handle_action():
    """HTTP endpoint for voice action requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "reason": "No JSON body"}), 400
        
        action = data.get('action', '').upper()
        source = data.get('source', 'unknown')
        priority = data.get('priority', 5)
        
        if action not in ALLOWED_ACTIONS:
            return jsonify({
                "status": "rejected", 
                "reason": f"Unknown action: {action}"
            }), 400
        
        # Publish ActionCommand to orchestrator
        if ros_node:
            msg = ActionCommand()
            msg.action_name = ACTION_MAP.get(action, action.lower())
            msg.parameters = [f"source={source}"]
            msg.priority = priority
            
            ros_node.action_pub.publish(msg)
            ros_node.get_logger().info(f'Action: {msg.action_name} from {source}')
            return jsonify({"status": "accepted", "action": action})
        else:
            return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503
            
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "ros2": ros_node is not None,
        "topic": "/orchestrator/action_command"
    })

class HttpBridgeNode(Node):
    """ROS2 node that publishes action commands to orchestrator"""
    
    def __init__(self):
        super().__init__('http_action_bridge')
        # Publish ActionCommand to orchestrator
        self.action_pub = self.create_publisher(
            ActionCommand, 
            '/orchestrator/action_command', 
            10
        )
        self.get_logger().info('HTTP Action Bridge ready')
        self.get_logger().info('  HTTP: http://0.0.0.0:5051/action')
        self.get_logger().info('  ROS2: /orchestrator/action_command')

def run_flask():
    """Run Flask HTTP server in background"""
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app.run(host='0.0.0.0', port=5051, threaded=True)

def main(args=None):
    global ros_node
    rclpy.init(args=args)
    ros_node = HttpBridgeNode()
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    ros_node.get_logger().info('Flask server started on port 5051')
    
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
