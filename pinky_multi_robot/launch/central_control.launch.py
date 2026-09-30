"""Start the Domain 52 bridge, guarded action proxy, mission server and UI.

The two localization guards run on the control PC but join robot Domains 20
and 22. Robot hardware and Nav2 bringup remain on their respective machines.
"""

import os
from pathlib import Path
import sys
from ament_index_python.packages import get_package_prefix, PackageNotFoundError

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, EmitEvent
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _existing_control_processes(module_root, bridge_config):
    """Catch an older launch's orphaned children before duplicating servers."""
    scripts = {
        str(module_root / 'localization_guard.py'),
        str(module_root / 'guarded_nav2_action_proxy.py'),
    }
    found = []
    for process_dir in Path('/proc').iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            argv = (process_dir / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(arg in scripts for arg in argv):
            found.append(process_dir.name)
        elif bridge_config in argv and any('domain_bridge' in arg for arg in argv):
            found.append(process_dir.name)
    return found


def generate_launch_description():
    package_root = Path(__file__).resolve().parents[1]
    module_root = package_root / 'pinky_multi_robot'
    source_pythonpath = str(package_root) + os.pathsep + os.environ.get('PYTHONPATH', '')
    common_env = {'PYTHONPATH': source_pythonpath}
    bridge_config = str(package_root / 'domain52_central_bridge.yaml')

    executables = {}
    for package, executable in [('domain_bridge', 'domain_bridge'),
                                ('pinky_multi_robot', 'mission_action_server'),
                                ('pinky_multi_robot', 'nav2_map_ui')]:
        try:
            prefix = get_package_prefix(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f'Preflight: source install/setup.bash; {package} unavailable. '
                               'No control processes started.') from error
        path = Path(prefix) / 'lib' / package / executable
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f'Preflight: executable unavailable: {path}')
        executables[executable] = str(path)
    for path in (Path(bridge_config), module_root / 'localization_guard.py',
                 module_root / 'guarded_nav2_action_proxy.py'):
        if not path.is_file():
            raise RuntimeError(f'Preflight: required file missing: {path}')

    def lifecycle(name):
        return dict(sigterm_timeout='5', sigkill_timeout='3',
                    on_exit=[EmitEvent(event=Shutdown(reason=f'{name} exited'))])

    duplicates = _existing_control_processes(module_root, bridge_config)
    if duplicates:
        raise RuntimeError(
            'Another central-control process is still running (PID: '
            + ', '.join(duplicates) + '). Stop its launch or orphaned children '
            'before starting a second copy.')

    return LaunchDescription([
        DeclareLaunchArgument('show_ui', default_value='true'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', '52'),
        ExecuteProcess(
            cmd=[executables['domain_bridge'], bridge_config],
            output='screen', name='amcl_pose_bridge', **lifecycle('bridge')),
        ExecuteProcess(
            cmd=[sys.executable, str(module_root / 'localization_guard.py'),
                 '--ros-args', '-p', 'auto_global_localization:=false'],
            additional_env={**common_env, 'ROS_DOMAIN_ID': '20'},
            output='screen', name='pinky1_localization_guard', **lifecycle('pinky1 guard')),
        ExecuteProcess(
            cmd=[sys.executable, str(module_root / 'localization_guard.py'),
                 '--ros-args', '-p', 'auto_global_localization:=false'],
            additional_env={**common_env, 'ROS_DOMAIN_ID': '22'},
            output='screen', name='pinky2_localization_guard', **lifecycle('pinky2 guard')),
        ExecuteProcess(
            cmd=[sys.executable, str(module_root / 'guarded_nav2_action_proxy.py')],
            additional_env=common_env,
            output='screen', name='guarded_nav2_action_proxy', **lifecycle('proxy')),
        Node(package='pinky_multi_robot', executable='mission_action_server',
             output='screen', **lifecycle('mission server')),
        Node(package='pinky_multi_robot', executable='nav2_map_ui',
             output='screen', condition=IfCondition(LaunchConfiguration('show_ui')),
             **lifecycle('UI')),
    ])
