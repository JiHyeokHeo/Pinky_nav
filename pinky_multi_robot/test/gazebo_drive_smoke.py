"""Drive both Gazebo robots through one real Nav2 mission on local Domain 152.

Start gazebo_control.launch.py first. Run this once from the initial spawn poses.
This script uses a localhost-only Cyclone DDS configuration and never joins
production Domains 20, 22, or 52.
"""

import importlib.util
import math
import os
from pathlib import Path
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
os.environ['CYCLONEDDS_URI'] = 'file://' + str(
    WORKSPACE / 'pinky_multi_robot' / 'config' / 'cyclonedds_sim.xml')
if importlib.util.find_spec('pinky_interfaces') is None:
    setup = WORKSPACE / 'install' / 'setup.bash'
    if not setup.is_file():
        raise SystemExit(f'Missing {setup}; build the workspace first.')
    if os.environ.get('PINKY_GAZEBO_BOOTSTRAPPED') == '1':
        raise SystemExit('pinky_interfaces is unavailable; build and source the workspace.')
    os.environ['PINKY_GAZEBO_BOOTSTRAPPED'] = '1'
    os.execv('/bin/bash', ['bash', '-c', 'source "$1" && exec "$2" "$3"',
                            'bash', str(setup), sys.executable, str(Path(__file__).resolve())])

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from pinky_multi_robot.localization_command_bus import LocalizationCommandBus


DOMAIN = 152
ROBOTS = ('pinky1', 'pinky2')


def spin_until(node, condition, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return condition()


def task(robot, x, y, yaw):
    result = RobotTask()
    result.robot_id = robot
    result.task_type = RobotTask.PARK
    result.parking_goal = PoseStamped()
    result.parking_goal.header.frame_id = f'{robot}/map'
    result.parking_goal.pose.position.x = x
    result.parking_goal.pose.position.y = y
    result.parking_goal.pose.orientation.z = math.sin(yaw / 2.0)
    result.parking_goal.pose.orientation.w = math.cos(yaw / 2.0)
    return result


def main():
    rclpy.init(domain_id=DOMAIN)
    node = Node('gazebo_drive_smoke')
    positions = {robot: None for robot in ROBOTS}
    states = {}
    for robot in ROBOTS:
        node.create_subscription(
            Odometry, f'/{robot}/odom',
            lambda msg, name=robot: positions.__setitem__(
                name, (msg.pose.pose.position.x, msg.pose.pose.position.y)), 10)
        node.create_subscription(
            String, f'/{robot}/localization_status',
            lambda msg, name=robot: states.__setitem__(name, msg.data), 10)
    client = ActionClient(node, ExecuteMultiRobotMission,
                          '/execute_multi_robot_mission')
    bus = LocalizationCommandBus(
        domains={robot: DOMAIN for robot in ROBOTS},
        namespaces={robot: robot for robot in ROBOTS})
    try:
        if not spin_until(node, lambda: client.server_is_ready() and
                          all(position is not None for position in positions.values()), 20.0):
            raise RuntimeError('Gazebo odom or mission server is not ready.')
        if not spin_until(node, lambda: all(
                bus._pose_publishers[robot].get_subscription_count() > 0
                for robot in ROBOTS), 5.0):
            raise RuntimeError('Simulated AMCL is not subscribing to /initialpose.')
        bus.set_initial_pose('pinky1', -0.35, 0.90, 0.0, 'pinky1/map')
        bus.set_initial_pose('pinky2', 0.65, -0.45, 1.57, 'pinky2/map')
        if not spin_until(node, lambda: all(states.get(robot) == 'READY'
                                            for robot in ROBOTS), 25.0):
            raise RuntimeError(f'Localization did not become READY: {states}')

        before = positions.copy()
        request = ExecuteMultiRobotMission.Goal()
        request.tasks = [task('pinky1', -0.41, 1.30, 0.0),
                         task('pinky2', 0.63, -0.03, 1.57)]
        future = client.send_goal_async(request)
        if not spin_until(node, future.done, 8.0):
            raise TimeoutError('Mission acceptance timed out.')
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError('Mission was rejected.')
        result_future = handle.get_result_async()
        if not spin_until(node, result_future.done, 60.0):
            handle.cancel_goal_async()
            raise TimeoutError('Mission did not finish; cancellation requested.')
        answer = result_future.result()
        displacement = {
            robot: math.hypot(positions[robot][0] - before[robot][0],
                              positions[robot][1] - before[robot][1])
            for robot in ROBOTS
        }
        print(f'Mission status={answer.status}, success={answer.result.success}; '
              + ', '.join(f'{robot} moved {distance:.3f} m'
                          for robot, distance in displacement.items()))
        if (answer.status != GoalStatus.STATUS_SUCCEEDED or
                not answer.result.success or
                any(distance < 0.05 for distance in displacement.values())):
            raise RuntimeError('Gazebo drive test failed or a robot did not move.')
        print('PASS: both simulated robots moved under one concurrent mission.')
    finally:
        bus.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
