"""Launch Pinky2 hardware, the established camera stream, and lane autonomy."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Start Pinky2 with one /cmd_vel publisher, initially disabled."""
    bringup_share = get_package_share_directory('pinky_bringup')
    move_share = get_package_share_directory('pinky_move')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/pinky/models/lane_seg_best.pt',
        ),
        DeclareLaunchArgument(
            'python_executable',
            default_value='/usr/bin/python3',
        ),
        DeclareLaunchArgument('remote_inference', default_value='true'),
        DeclareLaunchArgument('remote_geometry', default_value='true'),
        DeclareLaunchArgument('remote_port', default_value='18765'),
        DeclareLaunchArgument('result_timeout', default_value='0.8'),
        DeclareLaunchArgument('start_hardware', default_value='true'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument(
            'camera_script', default_value='/home/pinky/wjchoi/cam_stream.py'),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(os.path.join(
                bringup_share, 'launch', 'bringup_robot.launch.xml')),
            condition=IfCondition(LaunchConfiguration('start_hardware')),
        ),
        ExecuteProcess(
            cmd=[LaunchConfiguration('python_executable'),
                 LaunchConfiguration('camera_script')],
            additional_env={
                'LD_PRELOAD': '/usr/local/lib/aarch64-linux-gnu/libpisp.so.1.0.7',
            },
            output='screen',
        ),
        Node(
            package='pinky_move',
            executable='lane_autonomy',
            name='lane_autonomy',
            prefix=[LaunchConfiguration('python_executable')],
            additional_env={'YOLO_OFFLINE': 'true'},
            output='screen',
            parameters=[
                os.path.join(move_share, 'config', 'lane_autonomy.yaml'),
                {
                    'model_path': LaunchConfiguration('model_path'),
                    'enabled': False,
                    'remote_inference': ParameterValue(LaunchConfiguration('remote_inference'), value_type=bool),
                    'remote_geometry': ParameterValue(LaunchConfiguration('remote_geometry'), value_type=bool),
                    'remote_port': ParameterValue(LaunchConfiguration('remote_port'), value_type=int),
                    'result_timeout': ParameterValue(LaunchConfiguration('result_timeout'), value_type=float),
                    'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                    'calibration_path': os.path.join(move_share, 'config', 'robot_floor_calibration.json'),
                },
            ],
        ),
    ])
