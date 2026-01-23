#!/usr/bin/env python3
"""
SLAM HTTP Server for G1 Robot
Runs on G1 host (port 5052) - wraps slam_control binary

Features:
- SLAM mapping and navigation control
- Persistent waypoint storage
- Map management (save, list, load)
- Current pose tracking

Endpoints:
  POST /slam/start_mapping
  POST /slam/stop_mapping     body: {"map_name": "office"} (saves to /home/unitree/maps/)
  POST /slam/relocate         body: {"map_name": "office"} (loads from /home/unitree/maps/)
  POST /slam/goto             body: {"x":..., "y":..., "z":..., "qw":..., "qx":..., "qy":..., "qz":...}
  POST /slam/goto_waypoint    body: {"name": "kitchen"}
  POST /slam/pause
  POST /slam/resume
  GET  /slam/status
  GET  /slam/pose             - Get current robot pose
  POST /waypoint/save         body: {"name": "kitchen"}
  GET  /waypoint/list
  POST /waypoint/delete       body: {"name": "kitchen"}
  GET  /map/list              - List saved maps
  POST /map/delete            body: {"name": "office"}
"""

import os
import json
import subprocess
import threading
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from pathlib import Path

# Paths
SLAM_BINARY = os.path.expanduser("~/slam_control/build/slam_control")
MAPS_DIR = os.path.expanduser("~/maps")
WAYPOINTS_FILE = os.path.join(MAPS_DIR, "waypoints.json")
DEFAULT_MAP_NAME = "default_map"

# Ensure directories exist
os.makedirs(MAPS_DIR, exist_ok=True)

# Thread lock for waypoints file
waypoints_lock = threading.Lock()


def load_waypoints():
    """Load waypoints from JSON file."""
    with waypoints_lock:
        if os.path.exists(WAYPOINTS_FILE):
            try:
                with open(WAYPOINTS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}


def save_waypoints(waypoints):
    """Save waypoints to JSON file."""
    with waypoints_lock:
        with open(WAYPOINTS_FILE, 'w') as f:
            json.dump(waypoints, f, indent=2)


def get_map_path(map_name):
    """Get full path for a map file."""
    return f"/home/unitree/{map_name}.pcd"


def list_maps():
    """List all saved maps (by checking what relocate accepts)."""
    # Maps are stored internally by SLAM, but we track names we've used
    map_registry = os.path.join(MAPS_DIR, "map_registry.json")
    if os.path.exists(map_registry):
        try:
            with open(map_registry, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def register_map(map_name):
    """Register a map name after successful save."""
    map_registry = os.path.join(MAPS_DIR, "map_registry.json")
    maps = list_maps()
    maps[map_name] = {
        "path": get_map_path(map_name),
        "created": datetime.now().isoformat()
    }
    with open(map_registry, 'w') as f:
        json.dump(maps, f, indent=2)


def unregister_map(map_name):
    """Remove a map from registry."""
    map_registry = os.path.join(MAPS_DIR, "map_registry.json")
    maps = list_maps()
    if map_name in maps:
        del maps[map_name]
        with open(map_registry, 'w') as f:
            json.dump(maps, f, indent=2)
        return True
    return False


def run_slam_command(args, timeout=30):
    """Run slam_control binary with given arguments."""
    cmd = [SLAM_BINARY] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Command timed out"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_current_pose():
    """Get current robot pose from SLAM."""
    result = run_slam_command(["pose"], timeout=5)
    if result["status"] == "success" and result["stdout"]:
        try:
            # Parse the pose JSON from slam_control output
            pose_data = json.loads(result["stdout"])
            # Extract currentPose if present
            if "data" in pose_data and "currentPose" in pose_data["data"]:
                return pose_data["data"]["currentPose"]
            elif "currentPose" in pose_data:
                return pose_data["currentPose"]
            return pose_data
        except json.JSONDecodeError:
            return None
    return None


class SlamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {args[0]}")

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            return json.loads(self.rfile.read(content_length))
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/slam/status':
            pose = get_current_pose()
            waypoints = load_waypoints()
            maps = list_maps()
            self.send_json({
                "status": "ok",
                "current_pose": pose,
                "waypoints": list(waypoints.keys()),
                "maps": list(maps.keys())
            })

        elif self.path == '/slam/pose':
            pose = get_current_pose()
            if pose:
                self.send_json({"status": "ok", "pose": pose})
            else:
                self.send_json({"status": "error", "message": "Could not get pose. Is navigation mode active?"}, 500)

        elif self.path == '/waypoint/list':
            waypoints = load_waypoints()
            self.send_json({
                "status": "ok",
                "waypoints": waypoints,
                "names": list(waypoints.keys())
            })

        elif self.path == '/map/list':
            maps = list_maps()
            self.send_json({
                "status": "ok",
                "maps": maps,
                "names": list(maps.keys())
            })

        else:
            self.send_json({"status": "error", "message": "Unknown endpoint"}, 404)

    def do_POST(self):
        body = self.get_body()

        if self.path == '/slam/start_mapping':
            result = run_slam_command(['start_mapping'])
            self.send_json(result)

        elif self.path == '/slam/stop_mapping':
            map_name = body.get('map_name', DEFAULT_MAP_NAME)
            map_path = get_map_path(map_name)
            result = run_slam_command(['stop_mapping', map_path], timeout=60)

            # Register map on success
            if result["status"] == "success":
                register_map(map_name)
                result["map_name"] = map_name
                result["map_path"] = map_path

            self.send_json(result)

        elif self.path == '/slam/relocate':
            map_name = body.get('map_name', DEFAULT_MAP_NAME)
            map_path = body.get('map_path') or get_map_path(map_name)
            result = run_slam_command(['relocate', map_path])
            if result["status"] == "success":
                result["map_name"] = map_name
            self.send_json(result)

        elif self.path == '/slam/goto':
            x = body.get('x', 0)
            y = body.get('y', 0)
            z = body.get('z', 0)
            qw = body.get('qw', 1)
            qx = body.get('qx', 0)
            qy = body.get('qy', 0)
            qz = body.get('qz', 0)
            result = run_slam_command([
                'goto', str(x), str(y), str(z), str(qw), str(qx), str(qy), str(qz)
            ])
            self.send_json(result)

        elif self.path == '/slam/goto_waypoint':
            name = body.get('name', '').lower().strip()
            # Remove trailing punctuation
            name = name.rstrip('.,!?;:')
            if not name:
                self.send_json({"status": "error", "message": "Waypoint name required"}, 400)
                return

            waypoints = load_waypoints()

            # Try exact match first, then fuzzy match (handle punctuation in stored names)
            matched_name = None
            if name in waypoints:
                matched_name = name
            else:
                # Try to find a match ignoring punctuation
                for wp_name in waypoints.keys():
                    wp_clean = wp_name.rstrip('.,!?;:')
                    if wp_clean == name:
                        matched_name = wp_name
                        break

            if not matched_name:
                # Get clean names for display
                clean_names = [n.rstrip('.,!?;:') for n in waypoints.keys()]
                self.send_json({
                    "status": "error",
                    "message": f"Waypoint '{name}' not found",
                    "available": clean_names
                }, 404)
                return

            wp = waypoints[matched_name]
            result = run_slam_command([
                'goto',
                str(wp.get('x', 0)),
                str(wp.get('y', 0)),
                str(wp.get('z', 0)),
                str(wp.get('qw', 1)),
                str(wp.get('qx', 0)),
                str(wp.get('qy', 0)),
                str(wp.get('qz', 0))
            ])
            result['waypoint'] = name
            self.send_json(result)

        elif self.path == '/slam/pause':
            result = run_slam_command(['pause'])
            self.send_json(result)

        elif self.path == '/slam/resume':
            result = run_slam_command(['resume'])
            self.send_json(result)

        elif self.path == '/waypoint/save':
            name = body.get('name', '').lower().strip()
            # Remove trailing punctuation from waypoint names
            name = name.rstrip('.,!?;:')
            if not name:
                self.send_json({"status": "error", "message": "Waypoint name required"}, 400)
                return

            # Get current pose
            pose = get_current_pose()
            if not pose:
                self.send_json({
                    "status": "error",
                    "message": "Could not get current pose. Make sure navigation mode is active (say 'start navigation' first)."
                }, 400)
                return

            # Save waypoint
            waypoints = load_waypoints()
            waypoints[name] = {
                "x": pose.get("x", 0),
                "y": pose.get("y", 0),
                "z": pose.get("z", 0),
                "qw": pose.get("q_w", pose.get("qw", 1)),
                "qx": pose.get("q_x", pose.get("qx", 0)),
                "qy": pose.get("q_y", pose.get("qy", 0)),
                "qz": pose.get("q_z", pose.get("qz", 0)),
                "saved_at": datetime.now().isoformat()
            }
            save_waypoints(waypoints)

            self.send_json({
                "status": "success",
                "message": f"Waypoint '{name}' saved",
                "waypoint": waypoints[name]
            })

        elif self.path == '/waypoint/delete':
            name = body.get('name', '').lower().strip()
            if not name:
                self.send_json({"status": "error", "message": "Waypoint name required"}, 400)
                return

            waypoints = load_waypoints()
            if name in waypoints:
                del waypoints[name]
                save_waypoints(waypoints)
                self.send_json({"status": "success", "message": f"Waypoint '{name}' deleted"})
            else:
                self.send_json({"status": "error", "message": f"Waypoint '{name}' not found"}, 404)

        elif self.path == '/map/delete':
            name = body.get('name', '').lower().strip()
            if not name:
                self.send_json({"status": "error", "message": "Map name required"}, 400)
                return

            if unregister_map(name):
                self.send_json({"status": "success", "message": f"Map '{name}' removed from registry"})
            else:
                self.send_json({"status": "error", "message": f"Map '{name}' not found"}, 404)

        else:
            self.send_json({"status": "error", "message": "Unknown endpoint"}, 404)


def main():
    port = 5052
    server = HTTPServer(('0.0.0.0', port), SlamHandler)

    print("=" * 60)
    print("SLAM HTTP Server for G1 Robot")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Binary: {SLAM_BINARY}")
    print(f"Maps dir: {MAPS_DIR}")
    print(f"Waypoints: {WAYPOINTS_FILE}")
    print("")
    print("Endpoints:")
    print("  SLAM:")
    print("    POST /slam/start_mapping")
    print("    POST /slam/stop_mapping     {map_name: 'office'}")
    print("    POST /slam/relocate         {map_name: 'office'}")
    print("    POST /slam/goto             {x, y, z, qw, qx, qy, qz}")
    print("    POST /slam/goto_waypoint    {name: 'kitchen'}")
    print("    POST /slam/pause")
    print("    POST /slam/resume")
    print("    GET  /slam/status")
    print("    GET  /slam/pose")
    print("  Waypoints:")
    print("    POST /waypoint/save         {name: 'kitchen'}")
    print("    GET  /waypoint/list")
    print("    POST /waypoint/delete       {name: 'kitchen'}")
    print("  Maps:")
    print("    GET  /map/list")
    print("    POST /map/delete            {name: 'office'}")
    print("=" * 60)
    print("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SLAM Server] Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
