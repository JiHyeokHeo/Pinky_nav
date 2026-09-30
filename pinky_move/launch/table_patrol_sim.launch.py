"""Launch the four-table Gazebo world and the patrol controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start Gazebo, spawn Pinky on the tabletop, and start disabled control."""
    sim_share = get_package_share_directory('pinky_gz_sim')
    move_share = get_package_share_directory('pinky_move')
    sim_launch = os.path.join(
        sim_share, 'launch', 'four_tables_sim.launch.xml')
    config = os.path.join(
        move_share, 'config', 'table_patrol_sim.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Start the Gazebo graphical client',
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(sim_launch),
            launch_arguments={
                'cam_tilt_deg': '0',
                'spawn_x': '0.0',
                'spawn_y': '0.0',
                'spawn_z': '0.90',
                'gui': LaunchConfiguration('gui'),
            }.items(),
        ),
        Node(
            package='pinky_move',
            executable='ir_adc_sim_node',
            name='ir_adc_sim',
            output='screen',
            parameters=[
                config,
                {'use_sim_time': True},
            ],
        ),
        Node(
            package='pinky_move',
            executable='move_node',
            name='pinky_move',
            output='screen',
            parameters=[
                config,
                {'use_sim_time': True},
            ],
        ),
    ])
