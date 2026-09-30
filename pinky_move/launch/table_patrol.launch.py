"""Launch the Pinky Pro tabletop patrol controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create a configurable launch description for real or simulated time."""
    package_share = get_package_share_directory('pinky_move')
    default_config = os.path.join(
        package_share, 'config', 'table_patrol.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the Gazebo simulation clock',
        ),
        DeclareLaunchArgument(
            'config',
            default_value=default_config,
            description='Path to the patrol parameter YAML file',
        ),
        Node(
            package='pinky_move',
            executable='move_node',
            name='pinky_move',
            output='screen',
            parameters=[
                LaunchConfiguration('config'),
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
            ],
        ),
    ])
