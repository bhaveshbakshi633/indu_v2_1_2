#!/usr/bin/env python3
# HTTP-to-ROS2 bridge for voice commands + SLAM navigation
# HTTP se action aata hai, orchestrator ko ActionCommand publish karta hai
# SLAM commands via /api/slam_operate/request topic

import rclpy
from rclpy.node import Node
from orchestrator_msgs.msg import ActionCommand
from unitree_api.msg import Request
from std_msgs.msg import String
from flask import Flask, request, jsonify
import threading
import json
import os
import time

app = Flask(__name__)
ros_node = None

# File paths for persistent storage (mounted volumes)
MAPS_DIR = "/maps"
WAYPOINTS_DIR = "/waypoints"
WAYPOINTS_FILE = os.path.join(WAYPOINTS_DIR, "waypoints.json")
DEFAULT_MAP_FILE = "/maps/default_map.pcd"

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

# SLAM API IDs
SLAM_API = {
    "START_MAPPING": 1801,
    "END_MAPPING": 1802,
    "START_RELOCATION": 1804,
    "POSE_NAV": 1102,
    "PAUSE_NAV": 1201,
    "RESUME_NAV": 1202,
}


def load_waypoints():
    """Load waypoints from JSON file"""
    try:
        if os.path.exists(WAYPOINTS_FILE):
            with open(WAYPOINTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WAYPOINT] Error loading waypoints: {e}")
    return {}


def save_waypoints(waypoints):
    """Save waypoints to JSON file"""
    try:
        os.makedirs(WAYPOINTS_DIR, exist_ok=True)
        with open(WAYPOINTS_FILE, 'w') as f:
            json.dump(waypoints, f, indent=2)
        return True
    except Exception as e:
        print(f"[WAYPOINT] Error saving waypoints: {e}")
        return False


# ============================================================
# ACTION ENDPOINTS (existing)
# ============================================================

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
        "topic": "/orchestrator/action_command",
        "slam": {
            "current_pose": ros_node.current_pose if ros_node else None,
            "mapping_active": ros_node.mapping_active if ros_node else False,
            "navigation_active": ros_node.navigation_active if ros_node else False
        }
    })


# ============================================================
# SLAM ENDPOINTS (new)
# ============================================================

@app.route('/slam/start_mapping', methods=['POST'])
def start_mapping():
    """Start SLAM mapping mode"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    try:
        msg = Request()
        msg.header.identity.api_id = SLAM_API["START_MAPPING"]
        msg.parameter = json.dumps({"data": {"slam_type": "indoor"}})

        ros_node.slam_pub.publish(msg)
        ros_node.mapping_active = True
        ros_node.navigation_active = False
        ros_node.get_logger().info('[SLAM] Mapping started')

        return jsonify({
            "status": "accepted",
            "message": "Mapping started. Walk around to create map."
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/stop_mapping', methods=['POST'])
def stop_mapping():
    """Stop mapping and save map"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    try:
        # Get map filename from request or use default
        data = request.get_json() or {}
        map_name = data.get('name', 'default_map')
        map_path = f"/maps/{map_name}.pcd"

        msg = Request()
        msg.header.identity.api_id = SLAM_API["END_MAPPING"]
        msg.parameter = json.dumps({"data": {"address": map_path}})

        ros_node.slam_pub.publish(msg)
        ros_node.mapping_active = False
        ros_node.current_map = map_path
        ros_node.get_logger().info(f'[SLAM] Mapping stopped, saved to {map_path}')

        return jsonify({
            "status": "accepted",
            "message": f"Map saved to {map_path}",
            "map_path": map_path
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/relocate', methods=['POST'])
def start_relocation():
    """Start navigation mode (relocate on map)"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    try:
        # Get map path from request or use current/default
        data = request.get_json() or {}
        map_path = data.get('map_path', ros_node.current_map or DEFAULT_MAP_FILE)

        # Check if map exists
        if not os.path.exists(map_path):
            return jsonify({
                "status": "error",
                "reason": f"Map not found: {map_path}"
            }), 404

        # Initial pose (0,0,0 with identity quaternion)
        msg = Request()
        msg.header.identity.api_id = SLAM_API["START_RELOCATION"]
        msg.parameter = json.dumps({
            "data": {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,
                "address": map_path
            }
        })

        ros_node.slam_pub.publish(msg)
        ros_node.navigation_active = True
        ros_node.mapping_active = False
        ros_node.current_map = map_path
        ros_node.get_logger().info(f'[SLAM] Navigation started with map {map_path}')

        return jsonify({
            "status": "accepted",
            "message": f"Navigation started with map {map_path}",
            "map_path": map_path
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/goto', methods=['POST'])
def goto_pose():
    """Navigate to a specific pose (x, y, yaw)"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    if not ros_node.navigation_active:
        return jsonify({
            "status": "error",
            "reason": "Navigation not active. Start relocation first."
        }), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "reason": "No pose data"}), 400

        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        z = float(data.get('z', 0.0))
        q_x = float(data.get('q_x', 0.0))
        q_y = float(data.get('q_y', 0.0))
        q_z = float(data.get('q_z', 0.0))
        q_w = float(data.get('q_w', 1.0))

        msg = Request()
        msg.header.identity.api_id = SLAM_API["POSE_NAV"]
        msg.parameter = json.dumps({
            "data": {
                "targetPose": {
                    "x": x, "y": y, "z": z,
                    "q_x": q_x, "q_y": q_y, "q_z": q_z, "q_w": q_w
                },
                "mode": 1
            }
        })

        ros_node.slam_pub.publish(msg)
        ros_node.get_logger().info(f'[SLAM] Navigating to ({x}, {y})')

        return jsonify({
            "status": "accepted",
            "message": f"Navigating to ({x:.2f}, {y:.2f})",
            "target": {"x": x, "y": y, "z": z}
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/goto_waypoint', methods=['POST'])
def goto_waypoint():
    """Navigate to a saved waypoint by name"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    if not ros_node.navigation_active:
        return jsonify({
            "status": "error",
            "reason": "Navigation not active. Start relocation first."
        }), 400

    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "reason": "Waypoint name required"}), 400

        name = data['name'].lower().strip()
        waypoints = load_waypoints()

        if name not in waypoints:
            return jsonify({
                "status": "error",
                "reason": f"Waypoint '{name}' not found",
                "available": list(waypoints.keys())
            }), 404

        wp = waypoints[name]

        msg = Request()
        msg.header.identity.api_id = SLAM_API["POSE_NAV"]
        msg.parameter = json.dumps({
            "data": {
                "targetPose": {
                    "x": wp['x'], "y": wp['y'], "z": wp.get('z', 0.0),
                    "q_x": wp.get('q_x', 0.0), "q_y": wp.get('q_y', 0.0),
                    "q_z": wp.get('q_z', 0.0), "q_w": wp.get('q_w', 1.0)
                },
                "mode": 1
            }
        })

        ros_node.slam_pub.publish(msg)
        ros_node.get_logger().info(f'[SLAM] Navigating to waypoint: {name}')

        return jsonify({
            "status": "accepted",
            "message": f"Navigating to {name}",
            "waypoint": wp
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/pause', methods=['POST'])
def pause_navigation():
    """Pause current navigation"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    try:
        msg = Request()
        msg.header.identity.api_id = SLAM_API["PAUSE_NAV"]
        msg.parameter = json.dumps({"data": {}})

        ros_node.slam_pub.publish(msg)
        ros_node.get_logger().info('[SLAM] Navigation paused')

        return jsonify({
            "status": "accepted",
            "message": "Navigation paused"
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/resume', methods=['POST'])
def resume_navigation():
    """Resume paused navigation"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    try:
        msg = Request()
        msg.header.identity.api_id = SLAM_API["RESUME_NAV"]
        msg.parameter = json.dumps({"data": {}})

        ros_node.slam_pub.publish(msg)
        ros_node.get_logger().info('[SLAM] Navigation resumed')

        return jsonify({
            "status": "accepted",
            "message": "Navigation resumed"
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/slam/status', methods=['GET'])
def slam_status():
    """Get current SLAM status"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    return jsonify({
        "status": "ok",
        "mapping_active": ros_node.mapping_active,
        "navigation_active": ros_node.navigation_active,
        "current_map": ros_node.current_map,
        "current_pose": ros_node.current_pose,
        "last_arrival_status": ros_node.last_arrival_status
    })


# ============================================================
# WAYPOINT ENDPOINTS
# ============================================================

@app.route('/waypoint/save', methods=['POST'])
def save_waypoint():
    """Save current position as a named waypoint"""
    if not ros_node:
        return jsonify({"status": "error", "reason": "ROS2 not ready"}), 503

    if not ros_node.current_pose:
        return jsonify({
            "status": "error",
            "reason": "No pose data available. Start mapping or navigation first."
        }), 400

    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "reason": "Waypoint name required"}), 400

        name = data['name'].lower().strip()
        if not name:
            return jsonify({"status": "error", "reason": "Invalid waypoint name"}), 400

        waypoints = load_waypoints()
        waypoints[name] = ros_node.current_pose.copy()
        waypoints[name]['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

        if save_waypoints(waypoints):
            ros_node.get_logger().info(f'[WAYPOINT] Saved: {name} at {ros_node.current_pose}')
            return jsonify({
                "status": "accepted",
                "message": f"Waypoint '{name}' saved",
                "waypoint": waypoints[name]
            })
        else:
            return jsonify({"status": "error", "reason": "Failed to save waypoint"}), 500

    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/waypoint/list', methods=['GET'])
def list_waypoints():
    """List all saved waypoints"""
    try:
        waypoints = load_waypoints()
        return jsonify({
            "status": "ok",
            "count": len(waypoints),
            "waypoints": waypoints
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/waypoint/delete', methods=['POST'])
def delete_waypoint():
    """Delete a waypoint by name"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "reason": "Waypoint name required"}), 400

        name = data['name'].lower().strip()
        waypoints = load_waypoints()

        if name not in waypoints:
            return jsonify({
                "status": "error",
                "reason": f"Waypoint '{name}' not found"
            }), 404

        del waypoints[name]

        if save_waypoints(waypoints):
            return jsonify({
                "status": "accepted",
                "message": f"Waypoint '{name}' deleted"
            })
        else:
            return jsonify({"status": "error", "reason": "Failed to save changes"}), 500

    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


@app.route('/map/list', methods=['GET'])
def list_maps():
    """List all saved maps"""
    try:
        maps = []
        if os.path.exists(MAPS_DIR):
            for f in os.listdir(MAPS_DIR):
                if f.endswith('.pcd'):
                    path = os.path.join(MAPS_DIR, f)
                    maps.append({
                        "name": f.replace('.pcd', ''),
                        "path": path,
                        "size": os.path.getsize(path),
                        "modified": time.ctime(os.path.getmtime(path))
                    })
        return jsonify({
            "status": "ok",
            "count": len(maps),
            "maps": maps
        })
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500


# ============================================================
# ROS2 NODE
# ============================================================

class HttpBridgeNode(Node):
    """ROS2 node that publishes action commands and SLAM requests"""

    def __init__(self):
        super().__init__('http_action_bridge')

        # Publish ActionCommand to orchestrator
        self.action_pub = self.create_publisher(
            ActionCommand,
            '/orchestrator/action_command',
            10
        )

        # Publish SLAM requests
        self.slam_pub = self.create_publisher(
            Request,
            '/api/slam_operate/request',
            10
        )

        # Subscribe to SLAM info for current pose
        self.create_subscription(
            String,
            '/slam_info',
            self._slam_info_callback,
            10
        )

        # Subscribe to SLAM key info for arrival status
        self.create_subscription(
            String,
            '/slam_key_info',
            self._slam_key_info_callback,
            10
        )

        # State tracking
        self.current_pose = None
        self.mapping_active = False
        self.navigation_active = False
        self.current_map = None
        self.last_arrival_status = None

        self.get_logger().info('HTTP Action Bridge ready (with SLAM)')
        self.get_logger().info('  HTTP: http://0.0.0.0:5051')
        self.get_logger().info('  Actions: /action, /slam/*, /waypoint/*')
        self.get_logger().info('  ROS2: /orchestrator/action_command, /api/slam_operate/request')

    def _slam_info_callback(self, msg):
        """Process SLAM info messages for current pose"""
        try:
            data = json.loads(msg.data)
            if data.get('type') == 'pos_info':
                pose = data.get('data', {}).get('currentPose', {})
                if pose:
                    self.current_pose = {
                        'x': pose.get('x', 0.0),
                        'y': pose.get('y', 0.0),
                        'z': pose.get('z', 0.0),
                        'q_x': pose.get('q_x', 0.0),
                        'q_y': pose.get('q_y', 0.0),
                        'q_z': pose.get('q_z', 0.0),
                        'q_w': pose.get('q_w', 1.0)
                    }
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().warning(f'SLAM info parse error: {e}')

    def _slam_key_info_callback(self, msg):
        """Process SLAM key info for task completion"""
        try:
            data = json.loads(msg.data)
            if data.get('type') == 'task_result':
                self.last_arrival_status = data.get('data', {}).get('is_arrived', False)
                if self.last_arrival_status:
                    self.get_logger().info('[SLAM] Navigation complete - arrived at target')
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().warning(f'SLAM key info parse error: {e}')


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
