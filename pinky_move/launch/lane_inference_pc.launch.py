"""PC runs segmentation and requested geometry; robot owns motor commands."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('python_executable', default_value='/home/tory/venv/omx/bin/python'),
        DeclareLaunchArgument('robot', default_value='pinky@192.168.45.22'),
        DeclareLaunchArgument('model_path', default_value='/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt'),
        DeclareLaunchArgument('port', default_value='18765'),
        DeclareLaunchArgument('imgsz', default_value='320'),
        ExecuteProcess(cmd=[LaunchConfiguration('python_executable'), '-m',
                            'pinky_move.lane_inference_worker',
                            '--robot', LaunchConfiguration('robot'),
                            '--model', LaunchConfiguration('model_path'),
                            '--port', LaunchConfiguration('port'),
                            '--imgsz', LaunchConfiguration('imgsz')], output='screen'),
    ])
