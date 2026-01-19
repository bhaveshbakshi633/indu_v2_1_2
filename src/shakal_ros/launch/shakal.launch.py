"""
Shakal ROS2 Launch File
Launches: face_recognition_node + enrollment_service_node
Paths computed dynamically from workspace root
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('shakal_ros')
    # Workspace root: share is at install/shakal_ros/share/shakal_ros
    # Go up 4 levels to get workspace root
    ws_root = os.path.abspath(os.path.join(pkg_share, '..', '..', '..', '..'))

    # Model and data paths relative to workspace
    detection_model = os.path.join(ws_root, 'models', 'face_detection.onnx')
    recognition_model = os.path.join(ws_root, 'models', 'face_recognition.onnx')
    database_path = os.path.join(ws_root, 'data', 'embeddings', 'database.bin')

    default_params = os.path.join(pkg_share, 'config', 'shakal_params.yaml')

    # Common parameters to override YAML paths
    path_overrides = {
        'models.detection_path': detection_model,
        'models.recognition_path': recognition_model,
        'recognition.database_path': database_path,
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to parameters file'
        ),

        # Face Recognition Node
        Node(
            package='shakal_ros',
            executable='face_recognition_node',
            name='face_recognition',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                path_overrides
            ],
            remappings=[
                ('~/faces', '/shakal/faces'),
                ('~/names', '/shakal/names'),
                ('~/debug_image', '/shakal/debug_image'),
            ]
        ),

        # Enrollment Service Node
        Node(
            package='shakal_ros',
            executable='enrollment_service_node',
            name='enrollment',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                path_overrides
            ],
            remappings=[
                ('~/enroll', '/shakal/enroll'),
                ('~/remove', '/shakal/remove'),
                ('~/list_persons', '/shakal/list_persons'),
            ]
        ),
    ])
