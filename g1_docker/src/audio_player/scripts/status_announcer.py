#!/usr/bin/env python3
"""
Status Announcer - v2.10
Continuous ROS2 status monitoring with voice announcements
Monitors arm_controller and orchestrator status, announces important state changes

v2.10: Added FSM state broadcast subscription for immediate announcements
"""

import os
import sys
import json
import yaml
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Int32

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from announcer_utils import get_announcer, announce


class StatusAnnouncer(Node):
    """
    ROS2 Node that monitors status topics and announces important state changes
    """

    def __init__(self):
        super().__init__('status_announcer')

        # Load config
        self.config = self._load_config()
        self.announcements = self.config.get('announcements', {})

        # State tracking - only announce on change
        self.prev_arm_state = None
        self.prev_boot_state = None
        self.prev_fsm_state = None
        self.startup_complete_announced = False

        # Announcement queue (to avoid blocking callbacks)
        self.announcement_queue = []
        self.queue_lock = threading.Lock()

        # Subscribe to status topics
        self.arm_status_sub = self.create_subscription(
            String,
            '/arm_ctrl/status',
            self.arm_status_callback,
            10
        )

        # v2.10: Subscribe to FSM state broadcast from orchestrator for immediate announcements
        self.fsm_state_sub = self.create_subscription(
            Int32,
            '/fsm_state',
            self.fsm_state_callback,
            10
        )

        # State mapping for announcements - v2.10: Added new arm states
        self.arm_state_messages = {
            'AT_HOME': 'Arm controller at home position',
            'MOVING': 'Arms moving',
            'TEACH': 'Teach mode activated',
            'IDLE': None,  # Don't announce idle
            'HOLDING': None,
            'ARM_DISABLED': 'Arms disabled',
            'ARM_DISABLING': 'Arms disabling',
            'ARM_ENABLING': 'Arms initializing',
            'TRANSITIONING': None,  # Don't announce transition
            'ERROR': 'Arm controller error detected'
        }

        self.boot_state_messages = {
            'READY': 'Robot ready',
            'HOLDING': None,
            'AT_HOME': 'Robot at home position',
            'STANDING_UP': 'Robot standing up',
            'MOVING_TO_HOME': 'Moving to home position',
            'ERROR': 'Robot error state',
            'DISCONNECTED': None,
            'INIT': None,
            'WAIT_LOWSTATE': None,
            'DAMPING': 'Robot in damping mode'
        }

        # v2.10: FSM state messages - announces state transitions
        self.fsm_state_messages = {
            801: 'Robot ready for arm control',
            1: 'Robot entering damp mode',
            0: 'Zero torque mode',
            4: 'Robot standing up'
        }

        # Timer for processing announcement queue
        self.announcement_timer = self.create_timer(0.5, self.process_announcements)

        # Timer for startup complete announcement
        self.startup_timer = self.create_timer(5.0, self.check_startup_complete)

        self.get_logger().info('Status Announcer v2.10 initialized')
        self.get_logger().info('Listening for /arm_ctrl/status and /fsm_state')

    def _load_config(self):
        """Load configuration"""
        config_path = '/ros2_ws/config/external_services.yaml'
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}

    def arm_status_callback(self, msg):
        """Process arm_controller status updates"""
        try:
            status = json.loads(msg.data)

            arm_state = status.get('arm_state')
            boot_state = status.get('boot_state')
            fsm = status.get('fsm')

            # Check for arm state change
            if arm_state != self.prev_arm_state:
                if arm_state in self.arm_state_messages:
                    message = self.arm_state_messages.get(arm_state)
                    if message:
                        self._queue_announcement(message)
                self.prev_arm_state = arm_state

            # Check for boot state change
            if boot_state != self.prev_boot_state:
                if boot_state in self.boot_state_messages:
                    message = self.boot_state_messages.get(boot_state)
                    if message:
                        self._queue_announcement(message)
                self.prev_boot_state = boot_state

            # FSM from status is tracked but announcements come from /fsm_state topic
            # for more immediate notification
            self.prev_fsm_state = fsm

        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.get_logger().error(f'Status callback error: {e}')

    def fsm_state_callback(self, msg):
        """v2.10: Process FSM state broadcast from orchestrator - immediate announcements"""
        try:
            fsm = msg.data

            # Only announce changes and known states
            if fsm != self.prev_fsm_state and fsm in self.fsm_state_messages:
                message = self.fsm_state_messages.get(fsm)
                if message:
                    self.get_logger().info(f'FSM state change: {self.prev_fsm_state} -> {fsm}')
                    self._queue_announcement(message)
                self.prev_fsm_state = fsm

        except Exception as e:
            self.get_logger().error(f'FSM callback error: {e}')

    def _queue_announcement(self, message):
        """Add message to announcement queue"""
        with self.queue_lock:
            # Avoid duplicate consecutive announcements
            if not self.announcement_queue or self.announcement_queue[-1] != message:
                self.announcement_queue.append(message)
                self.get_logger().info(f'Queued announcement: {message}')

    def process_announcements(self):
        """Process queued announcements (runs in timer callback)"""
        with self.queue_lock:
            if self.announcement_queue:
                message = self.announcement_queue.pop(0)
            else:
                return

        # Announce in background thread to not block ROS2
        threading.Thread(
            target=lambda: announce(message, wait=True),
            daemon=True
        ).start()

    def check_startup_complete(self):
        """Check if startup is complete and announce"""
        if self.startup_complete_announced:
            self.startup_timer.cancel()
            return

        # Check if we have received status and robot is ready
        if self.prev_fsm_state == 801 and self.prev_boot_state in ['READY', 'HOLDING', 'AT_HOME']:
            self.startup_complete_announced = True
            msg = self.announcements.get('system_ready', 'System startup complete. Ready for operation.')
            self._queue_announcement(msg)
            self.startup_timer.cancel()
            self.get_logger().info('Startup complete announced')


def main(args=None):
    print('=' * 60)
    print('STATUS ANNOUNCER')
    print('=' * 60)

    rclpy.init(args=args)

    try:
        node = StatusAnnouncer()

        # Use multi-threaded executor
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        print('Status Announcer running...')
        executor.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
