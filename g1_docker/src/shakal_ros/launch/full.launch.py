"""
Shakal ROS2 Full Launch File
Launches: face_recognition_node + enrollment_service_node + obsbot_service_node
Paths computed dynamically from workspace root
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('shakal_ros')
    default_params = os.path.join(pkg_share, 'config', 'shakal_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to parameters file'
        ),

        # Include base launch (face_recognition + enrollment)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'shakal.launch.py')
            ),
            launch_arguments={
                'params_file': LaunchConfiguration('params_file')
            }.items()
        ),

        # OBSBOT Control Service Node
        Node(
            package='shakal_ros',
            executable='obsbot_service_node',
            name='obsbot_control',
            output='screen',
            remappings=[
                ('~/set_fov', '/shakal/obsbot/set_fov'),
                ('~/set_zoom', '/shakal/obsbot/set_zoom'),
            ]
        ),
    ])
