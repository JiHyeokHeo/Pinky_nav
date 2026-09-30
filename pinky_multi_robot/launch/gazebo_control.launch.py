"""Two-Pinky Gazebo/Nav2/control integration on isolated ROS Domain 152.

This launch intentionally does not start the production Domain 52 bridge or
action proxy: both simulated Nav2 servers live in one isolated DDS domain and
already expose /pinky1/navigate_to_pose and /pinky2/navigate_to_pose.
"""

import os
from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    workspace = Path(__file__).resolve().parents[2]
    control_package = workspace / 'pinky_multi_robot'
    sim_package = workspace / 'pinky_gz_sim'
    nav_package = workspace / 'pinky_navigation'
    domain = LaunchConfiguration('sim_domain')
    map_file = str(nav_package / 'map' / 'parking_passage_map.yaml')
    env = {'PYTHONPATH': str(control_package) + os.pathsep + os.environ.get('PYTHONPATH', '')}

    actions = [
        DeclareLaunchArgument('sim_domain', default_value='152'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('show_ui', default_value='true'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', domain),
        SetEnvironmentVariable('CYCLONEDDS_URI',
                               'file://' + str(control_package / 'config' / 'cyclonedds_sim.xml')),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(sim_package / 'launch' / 'parking_passage_sim.launch.xml')),
            launch_arguments={'gui': LaunchConfiguration('gui')}.items()),
    ]

    for robot in ('pinky1', 'pinky2'):
        nav_params = RewrittenYaml(
            source_file=str(nav_package / 'params' / f'{robot}_parking_passage_nav2_v2.yaml'),
            param_rewrites={'update_min_a': '0.0', 'update_min_d': '0.0'},
            root_key=robot, convert_types=True)
        actions.extend([
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(str(nav_package / 'launch' / 'gz_bringup_launch.xml')),
                launch_arguments={
                    'namespace': robot,
                    'use_composition': 'False',
                    'use_sim_time': 'True',
                    'map': map_file,
                    'params_file': nav_params,
                }.items()),
            ExecuteProcess(
                cmd=[sys.executable, str(control_package / 'pinky_multi_robot' / 'sim_odom_tf.py'),
                     '--ros-args', '-r', f'__ns:=/{robot}', '-r', '/tf:=tf'],
                additional_env=env, output='screen', name=f'{robot}_sim_odom_tf'),
            ExecuteProcess(
                cmd=[sys.executable, str(control_package / 'pinky_multi_robot' / 'localization_guard.py'),
                     '--ros-args', '-r', f'__ns:=/{robot}',
                     '-p', f'map_frame:={robot}/map',
                     '-p', f'base_frame:={robot}/base_footprint',
                     '-p', 'use_sim_time:=true',
                     '-r', '/tf:=tf', '-r', '/tf_static:=tf_static',
                     '-p', 'latch_ready_on_pose_timeout:=true'],
                additional_env=env, output='screen', name=f'{robot}_sim_localization_guard'),
        ])

    actions.extend([
        Node(package='pinky_multi_robot', executable='mission_action_server', output='screen',
             parameters=[{'allow_namespaced_map_frames': True}]),
        Node(package='pinky_multi_robot', executable='nav2_map_ui', output='screen',
             condition=IfCondition(LaunchConfiguration('show_ui')),
             arguments=['--simulation', '--simulation-domain', domain,
                        '--domain-id', domain, '--map-yaml', map_file]),
    ])
    return LaunchDescription(actions)
