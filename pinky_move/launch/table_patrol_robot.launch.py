"""Launch Pinky Pro hardware, ADC IR, USB camera, and patrol."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start robot hardware and guards with patrol disabled."""
    bringup_share = get_package_share_directory('pinky_bringup')
    move_share = get_package_share_directory('pinky_move')
    bringup_launch = os.path.join(
        bringup_share, 'launch', 'bringup_robot.launch.xml')
    config = os.path.join(
        move_share, 'config', 'table_patrol.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_device',
            default_value='/dev/video0',
            description='USB camera device path',
        ),
        DeclareLaunchArgument(
            'pixel_format',
            default_value='mjpeg2rgb',
            description='usb_cam pixel format; try yuyv2rgb if unsupported',
        ),
        DeclareLaunchArgument(
            'start_ir_adc',
            default_value='true',
            description='Start the built-in Pinky Pro ADC/IR driver',
        ),
        DeclareLaunchArgument(
            'ir_i2c_interface',
            default_value='/dev/i2c-1',
            description='I2C interface used by pinky_sensor_adc',
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(bringup_launch),
        ),
        Node(
            package='pinky_sensor_adc',
            executable='main_node',
            name='pinky_sensor_adc',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_ir_adc')),
            parameters=[{
                'interface': LaunchConfiguration('ir_i2c_interface'),
                'rate': 20.0,
            }],
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
            executable='move_node',
            name='pinky_move',
            output='screen',
            parameters=[config],
        ),
    ])
