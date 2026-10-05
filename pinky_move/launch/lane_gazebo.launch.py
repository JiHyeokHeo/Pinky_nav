"""Isolated Gazebo camera/YOLO/physical-wheel lane tests; never hardware."""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    from pinky_move.lane_simulation import prepare_simulation
    value = lambda name: LaunchConfiguration(name).perform(context)
    domain = int(value('domain'))
    if domain in (20, 22, 52) or not 0 <= domain <= 232:
        raise ValueError('Use a separate simulation domain, not robot/control domains 20/22/52')
    directory = value('output') or tempfile.mkdtemp(prefix='pinky-lane-gazebo-')
    world, calibration = prepare_simulation(get_package_share_directory('pinky_description'),
        directory, value('course'), float(value('lane_width')),
        int(value('lane_count')), int(value('target_lane')), value('perception'))
    share = get_package_share_directory('pinky_move')
    env = {'ROS_DOMAIN_ID': str(domain), 'CYCLONEDDS_URI':
        '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>',
        'GZ_PARTITION': 'pinky_lane_sim_'+str(domain), 'YOLO_OFFLINE': 'true'}
    bridge = ['/lane_sim/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
              '/lane_sim/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
              '/lane_sim/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
              '/lane_sim/ground_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry',
              '/lane_sim/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
              '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    return [*(SetEnvironmentVariable(k,v) for k,v in env.items()),
        ExecuteProcess(cmd=['gz','sim','-s','-r','--headless-rendering','-v','2',world], output='screen'),
        ExecuteProcess(cmd=['gz','sim','-g'], condition=IfCondition(LaunchConfiguration('gui')), output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='lane_sim_bridge',
             arguments=bridge, output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='pinky_move', executable='lane_autonomy', namespace='lane_sim', name='lane_autonomy',
             prefix=[value('python_executable')], output='screen', parameters=[
                 os.path.join(share,'config','lane_autonomy.yaml'), dict(use_sim_time=True,
                    model_path=value('model_path'), calibration_path=calibration,
                    remote_inference=False, remote_geometry=False, enabled=False,
                    simulation_white_lane=value('perception') == 'opencv',
                    connected_geometry=value('connected_geometry').lower() == 'true',
                    metric_path_min_m=.08 if value('perception') == 'opencv' else .14,
                    metric_path_max_m=.70 if value('perception') == 'opencv' else .48,
                    metric_lookahead_m=.16 if value('perception') == 'opencv' else .22,
                    maximum_angular_speed=.6 if value('perception') == 'opencv' else .15,
                    image_topic='/lane_sim/camera/image_raw', image_compressed=False,
                    cmd_vel_topic='/lane_sim/cmd_vel', corner_odom_topic='/lane_sim/odom',
                    scan_topic='/lane_sim/scan', use_lidar_guard=False,
                    debug_image_topic='/lane_sim/debug_image', lane_width=float(value('lane_width')),
                    publish_debug_image=True, linear_speed=.10,
                    single_line_max_speed=.06 if value('perception') == 'opencv' else .03)])]


def generate_launch_description():
    defaults = dict(course='two_lines', lane_width='.20', lane_count='1', target_lane='0', perception='yolo',
        domain='172', gui='true', output='', connected_geometry='false',
        python_executable='/home/tory/venv/omx/bin/python',
        model_path='/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt')
    return LaunchDescription([*(DeclareLaunchArgument(k, default_value=v) for k,v in defaults.items()), OpaqueFunction(function=start)])
