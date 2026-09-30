"""Launch Pinky Pro hardware, camera, and YOLO lane autonomy."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start one and only one /cmd_vel controller."""
    bringup_share = get_package_share_directory('pinky_bringup')
    move_share = get_package_share_directory('pinky_move')
    bringup_launch = os.path.join(
        bringup_share, 'launch', 'bringup_robot.launch.xml')
    config = os.path.join(move_share, 'config', 'lane_autonomy.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=(
                '/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt'),
            description='Absolute path to the YOLO segmentation best.pt',
        ),
        DeclareLaunchArgument(
            'camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument(
            'pixel_format', default_value='mjpeg2rgb'),
        DeclareLaunchArgument(
            'python_executable',
            default_value='/home/tory/venv/omx/bin/python',
            description='Python interpreter containing ultralytics',
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(bringup_launch),
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='pinky_camera',
            output='screen',
            parameters=[{
                'video_device': LaunchConfiguration('camera_device'),
                'framerate': 30.0,
                'io_method': 'mmap',
                'pixel_format': LaunchConfiguration('pixel_format'),
                'image_width': 640,
                'image_height': 480,
                'camera_name': 'pinky_camera',
                'frame_id': 'front_camera_link',
            }],
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
            ],
        ),
        Node(
            package='pinky_move',
            executable='lane_autonomy',
            name='lane_autonomy',
            prefix=[LaunchConfiguration('python_executable')],
            output='screen',
            parameters=[
                config,
                {'model_path': LaunchConfiguration('model_path')},
            ],
        ),
    ])
