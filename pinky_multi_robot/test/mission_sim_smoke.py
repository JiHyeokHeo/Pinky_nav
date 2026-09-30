"""Isolated ROS action-level simulation of two simultaneous Nav2 robots.

Run directly: python3 pinky_multi_robot/test/mission_sim_smoke.py
This never connects to robot Domains 20, 22, or central Domain 52.
"""

import importlib.util
import os
from pathlib import Path
import sys
import threading
import time

if importlib.util.find_spec('pinky_interfaces') is None:
    setup = Path(__file__).resolve().parents[2] / 'install' / 'setup.bash'
    if not setup.is_file():
        raise SystemExit(f'Missing {setup}; build pinky_interfaces first.')
    if os.environ.get('PINKY_SIM_BOOTSTRAPPED') == '1':
        raise SystemExit('pinky_interfaces is still unavailable after sourcing the workspace. '
                         'Build it with: colcon build --packages-select pinky_interfaces')
    os.environ['PINKY_SIM_BOOTSTRAPPED'] = '1'
    os.execv('/bin/bash', ['bash', '-c', 'source "$1" && exec "$2" "$3"',
                            'bash', str(setup), sys.executable, str(Path(__file__).resolve())])

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Path as NavPath
from rclpy.action import ActionClient, ActionServer, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from pinky_multi_robot.mission_action_server import MissionActionServer


def wait(future, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not future.done():
        raise TimeoutError('ROS action did not finish in time')
    return future.result()


class FakeNav2(Node):
    def __init__(self, robot, starts, y):
        super().__init__(f'{robot}_sim_nav2')
        self.robot = robot
        self.starts = starts
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, f'/{robot}/tracked_pose', 10)
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        pose.pose.pose.position.y = y
        pose.pose.pose.orientation.w = 1.0
        self.pose = pose
        self.create_timer(0.1, self.publish_pose)
        self.server = ActionServer(
            self, NavigateToPose, f'/{robot}/navigate_to_pose',
            execute_callback=self.execute,
            callback_group=ReentrantCallbackGroup())
        self.planner = ActionServer(
            self, ComputePathToPose, f'/{robot}/compute_path_to_pose',
            execute_callback=self.compute_path,
            callback_group=ReentrantCallbackGroup())

    def publish_pose(self):
        self.pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_pub.publish(self.pose)

    async def compute_path(self, goal_handle):
        request = goal_handle.request
        result = ComputePathToPose.Result()
        result.path = NavPath()
        result.path.header.frame_id = 'map'
        result.path.poses = [request.start, request.goal]
        goal_handle.succeed()
        return result

    async def execute(self, goal_handle):
        self.starts.append((self.robot, time.monotonic()))
        future = self.executor.create_future()
        timer = self.create_timer(0.8, lambda: future.set_result(True)
                                  if not future.done() else None,
                                  callback_group=ReentrantCallbackGroup())
        try:
            await future
        finally:
            self.destroy_timer(timer)
        result = NavigateToPose.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
        return result


def park(robot, x, y=0.0):
    task = RobotTask()
    task.robot_id = robot
    task.task_type = RobotTask.PARK
    task.parking_goal = PoseStamped()
    task.parking_goal.header.frame_id = 'map'
    task.parking_goal.pose.position.x = x
    task.parking_goal.pose.position.y = y
    task.parking_goal.pose.orientation.w = 1.0
    return task


def main():
    domain = 150 + os.getpid() % 50
    rclpy.init(domain_id=domain)
    starts = []
    server = MissionActionServer()
    nav1 = FakeNav2('pinky1', starts, 0.0)
    nav2 = FakeNav2('pinky2', starts, 1.0)
    client_node = Node('mission_sim_client')
    client = ActionClient(client_node, ExecuteMultiRobotMission,
                          '/execute_multi_robot_mission')
    executor = MultiThreadedExecutor(num_threads=6)
    for node in (server, nav1, nav2, client_node):
        executor.add_node(node)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    try:
        assert client.wait_for_server(timeout_sec=5.0), 'Mission server not ready'
        time.sleep(0.5)
        request = ExecuteMultiRobotMission.Goal()
        request.tasks = [park('pinky1', 1.0), park('pinky2', 2.0, 1.0)]
        handle = wait(client.send_goal_async(request))
        assert handle.accepted, 'Combined mission rejected'
        duplicate = wait(client.send_goal_async(request))
        assert not duplicate.accepted, 'Overlapping mission was accepted'
        outcome = wait(handle.get_result_async())
        assert outcome.status == GoalStatus.STATUS_SUCCEEDED, outcome.status
        assert outcome.result.success, outcome.result.message
        assert {name for name, _ in starts} == {'pinky1', 'pinky2'}, starts
        skew = abs(starts[0][1] - starts[1][1])
        assert skew < 0.4, f'Robot goals were not concurrent: {skew:.3f}s'
        starts.clear()
        crossing = ExecuteMultiRobotMission.Goal()
        crossing.tasks = [park('pinky1', 1.0, 1.0), park('pinky2', 1.0, 0.0)]
        crossing_handle = wait(client.send_goal_async(crossing))
        assert crossing_handle.accepted
        crossing_result = wait(crossing_handle.get_result_async())
        assert crossing_result.status == GoalStatus.STATUS_SUCCEEDED, crossing_result.result.message
        assert [name for name, _ in starts] == ['pinky1', 'pinky2'], starts
        delay = starts[1][1] - starts[0][1]
        assert delay >= 0.7, f'Crossing routes started together: {delay:.3f}s'
        starts.clear()
        blocked = ExecuteMultiRobotMission.Goal()
        blocked.tasks = [park('pinky1', 0.0, 1.0), park('pinky2', 1.0, 1.0)]
        blocked_handle = wait(client.send_goal_async(blocked))
        assert blocked_handle.accepted
        blocked_result = wait(blocked_handle.get_result_async())
        assert blocked_result.status == GoalStatus.STATUS_ABORTED
        assert 'no safe ordering' in blocked_result.result.message
        assert not starts, f'Unsafe mission moved a robot: {starts}'
        print(f'PASS: isolated ROS Domain {domain}; both goals began {skew:.3f}s apart; '
              f'crossing goals staggered {delay:.3f}s; unsafe staging rejected.')
    finally:
        executor.shutdown()
        for node in (client_node, nav2, nav1, server):
            node.destroy_node()
        rclpy.shutdown()
        worker.join(timeout=2.0)


if __name__ == '__main__':
    main()
