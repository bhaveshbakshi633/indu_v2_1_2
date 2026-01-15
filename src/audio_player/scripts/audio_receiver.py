#!/usr/bin/env python3
"""
Audio Receiver - HTTP to ROS2 Bridge

Laptop se HTTP POST ke through PCM audio receive karta hai
aur /g1/tts/audio_output topic pe publish karta hai.

tts_audio_player (C++) is topic ko subscribe karke G1 speaker pe play karta hai.

Usage:
    ros2 run audio_player audio_receiver.py

Test:
    curl -X POST http://G1_IP:5050/play_audio --data-binary @test.pcm
"""

import threading
import time
from flask import Flask, request, jsonify

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import UInt8MultiArray, Bool, Empty

# Constants
HTTP_PORT = 5050
AUDIO_TOPIC = "/g1/tts/audio_output"
STATUS_TOPIC = "/g1/tts/speaking"
STOP_TOPIC = "/g1/tts/stop"


class AudioReceiverNode(Node):
    """ROS2 node jo audio publish karta hai"""

    def __init__(self):
        super().__init__('audio_receiver')

        # Publisher for audio data
        self.audio_pub = self.create_publisher(
            UInt8MultiArray,
            AUDIO_TOPIC,
            10
        )

        # Publisher for stop command
        self.stop_pub = self.create_publisher(
            Empty,
            STOP_TOPIC,
            10
        )

        # Subscriber for speaking status (from tts_audio_player)
        self.speaking_sub = self.create_subscription(
            Bool,
            STATUS_TOPIC,
            self.speaking_callback,
            10
        )

        self.is_speaking = False
        self.is_muted = False  # Mute flag - jab True ho tab audio forward nahi hoga
        self.last_audio_time = 0
        self.start_time = time.time()

        self.get_logger().info(f"Audio Receiver initialized")
        self.get_logger().info(f"Publishing audio to: {AUDIO_TOPIC}")
        self.get_logger().info(f"Listening for status on: {STATUS_TOPIC}")

    def speaking_callback(self, msg):
        """tts_audio_player se speaking status receive karo"""
        self.is_speaking = msg.data

    def publish_stop(self) -> bool:
        """Stop command publish karo to tts_audio_player"""
        try:
            self.stop_pub.publish(Empty())
            self.get_logger().info("Stop command published")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish stop: {e}")
            return False

    def set_mute(self, muted: bool):
        """Mute state set karo"""
        self.is_muted = muted
        self.get_logger().info(f"Mute state: {muted}")

    def publish_audio(self, pcm_data: bytes) -> bool:
        """PCM audio bytes ko ROS2 topic pe publish karo"""
        # Agar muted hai toh audio forward nahi karo
        if self.is_muted:
            self.get_logger().debug("Audio muted, skipping publish")
            return True  # Success return karo taaki sender ko pata na chale

        try:
            msg = UInt8MultiArray()
            msg.data = list(pcm_data)
            self.audio_pub.publish(msg)
            self.last_audio_time = time.time()
            self.get_logger().info(f"Published audio: {len(pcm_data)} bytes")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish audio: {e}")
            return False

    def get_status(self) -> dict:
        """Current status return karo"""
        return {
            "speaking": self.is_speaking,
            "muted": self.is_muted,
            "last_audio_time": self.last_audio_time,
            "uptime": time.time() - self.start_time
        }


# Flask app for HTTP endpoint
app = Flask(__name__)
ros_node = None


@app.route('/play_audio', methods=['POST'])
def play_audio():
    """
    PCM audio receive karo aur ROS2 pe publish karo

    Expected: Raw PCM bytes (16kHz, mono, 16-bit)
    """
    global ros_node

    if ros_node is None:
        return jsonify({"error": "ROS2 node not initialized"}), 503

    pcm_data = request.data
    if not pcm_data:
        return jsonify({"error": "No audio data received"}), 400

    # PCM data publish karo
    success = ros_node.publish_audio(pcm_data)

    if success:
        # Estimate duration (16kHz, 16-bit mono = 32000 bytes/sec)
        duration_sec = len(pcm_data) / 32000.0
        return jsonify({
            "success": True,
            "bytes_received": len(pcm_data),
            "estimated_duration_sec": round(duration_sec, 2)
        })
    else:
        return jsonify({"error": "Failed to publish audio"}), 500


@app.route('/status', methods=['GET'])
def status():
    """Current status return karo"""
    global ros_node

    if ros_node is None:
        return jsonify({"error": "ROS2 node not initialized"}), 503

    return jsonify(ros_node.get_status())


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "audio_receiver"})


@app.route('/stop', methods=['POST'])
def stop_audio():
    """Stop current audio playback immediately"""
    global ros_node

    if ros_node is None:
        return jsonify({"error": "ROS2 node not initialized"}), 503

    success = ros_node.publish_stop()

    if success:
        return jsonify({"success": True, "message": "Stop command sent"})
    else:
        return jsonify({"error": "Failed to send stop command"}), 500


@app.route('/mute', methods=['POST'])
def mute_audio():
    """Mute/unmute audio output"""
    global ros_node

    if ros_node is None:
        return jsonify({"error": "ROS2 node not initialized"}), 503

    # Request body se mute state lo, ya toggle karo
    data = request.get_json(silent=True) or {}
    if 'muted' in data:
        muted = bool(data['muted'])
    else:
        # Toggle current state
        muted = not ros_node.is_muted

    ros_node.set_mute(muted)

    return jsonify({"success": True, "muted": ros_node.is_muted})


def run_flask():
    """Flask server background me run karo"""
    app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True)


def main(args=None):
    global ros_node

    # Initialize ROS2
    rclpy.init(args=args)

    try:
        # Create ROS2 node
        ros_node = AudioReceiverNode()

        # Flask ko background thread me run karo
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

        ros_node.get_logger().info(f"HTTP server listening on port {HTTP_PORT}")
        ros_node.get_logger().info(f"Endpoints: POST /play_audio, /stop, /mute | GET /status, /health")

        # Multi-threaded executor use karo
        executor = MultiThreadedExecutor()
        executor.add_node(ros_node)
        executor.spin()

    except KeyboardInterrupt:
        pass
    finally:
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
