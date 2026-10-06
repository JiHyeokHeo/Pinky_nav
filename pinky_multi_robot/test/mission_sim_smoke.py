"""Isolated ROS action-level simulation of two simultaneous Nav2 robots.

Run directly: python3 pinky_multi_robot/test/mission_sim_smoke.py
This never connects to robot Domains 20, 22, or central Domain 52.
"""

import importlib.util
import json
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
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Path as NavPath
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from pinky_multi_robot.mission_action_server import MissionActionServer
from pinky_multi_robot.mission_view import VIEW_TOPIC, parse_view


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
        self.duration = .8
        self.finish_pose = None
        self.move_to_goal = False
        self.completed_at = None
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
            cancel_callback=lambda _: CancelResponse.ACCEPT,
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
        timer = self.create_timer(self.duration, lambda: future.set_result(True)
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
            if self.move_to_goal:
                self.pose.pose.pose = goal_handle.request.pose.pose
                self.publish_pose()
            if self.finish_pose is not None:
                self.pose.pose.pose.position.x, self.pose.pose.pose.position.y = self.finish_pose
                self.publish_pose()
            self.completed_at = time.monotonic()
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
    views = []
    subscription = client_node.create_subscription(String, VIEW_TOPIC,
        lambda message: views.append(parse_view(message.data)),
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
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
        time.sleep(.1)
        assert views and views[-1]['state'] == 'succeeded'
        assert set(views[-1]['paths']) == {'pinky1', 'pinky2'}
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
        # PARK/PARK now approaches together, then cancels/resends the yielding
        # robot. The old assertion incorrectly expected serial initial starts.
        assert {name for name, _ in starts} == {'pinky1', 'pinky2'}, starts
        assert [name for name, _ in starts].count('pinky1') == 1, starts
        second_starts = [stamp for name, stamp in starts if name == 'pinky2']
        # Timer may yield before the second send, or just after acceptance.
        # Both schedules are valid; neither may retry the completed priority.
        assert 1 <= len(second_starts) <= 2, starts
        delay = second_starts[-1]-next(stamp for name, stamp in starts if name == 'pinky1')
        assert delay >= 1.9, f'Yielding robot resumed before minimum wait: {delay:.3f}s'
        starts.clear()
        blocked = ExecuteMultiRobotMission.Goal()
        # Swapping the occupied start points blocks both possible orderings.
        blocked.tasks = [park('pinky1', 0.0, 1.0), park('pinky2', 0.0, 0.0)]
        blocked_handle = wait(client.send_goal_async(blocked))
        assert blocked_handle.accepted
        blocked_result = wait(blocked_handle.get_result_async())
        assert blocked_result.status == GoalStatus.STATUS_ABORTED
        assert 'no safe ordering' in blocked_result.result.message
        time.sleep(.1)
        assert views[-1]['state'] == 'failed'
        assert 'waiting-to-' in views[-1]['reason']
        assert len(views[-1]['paths']) == 2
        assert not starts, f'Unsafe mission moved a robot: {starts}'

        # Regression: first robot has finished and its parked tracked pose is
        # off the original checked polyline. The second task must still finish.
        nav1.duration, nav2.duration = .4, 1.5
        nav1.finish_pose = (3., 0.)
        starts.clear()
        arrival = wait(client.send_goal_async(request))
        assert arrival.accepted
        deadline = time.monotonic()+5.
        while len(starts) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(starts) == 2
        server._approach_paths = {'pinky1': [(0., 0.), (1., 0.)],
                                  'pinky2': [(0., 1.), (2., 1.)]}
        arrival_result = wait(arrival.get_result_async())
        assert arrival_result.status == GoalStatus.STATUS_SUCCEEDED, arrival_result.result.message
        assert arrival_result.result.success
        assert nav2.completed_at-nav1.completed_at > .8
        assert len(starts) == 2, 'Completed robot or second goal was needlessly reissued'

        # Confirm persistent deviation causes a bounded planner refresh and
        # retry, not terminal cancellation of either remaining task.
        nav1.duration = nav2.duration = 2.
        nav1.finish_pose = None
        nav1.pose.pose.pose.position.x = 0.
        nav1.publish_pose()
        starts.clear()
        refresh = wait(client.send_goal_async(request))
        assert refresh.accepted
        deadline = time.monotonic()+5.
        while len(starts) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(starts) == 2
        server._approach_paths = {'pinky1': [(0., 0.), (1., 0.)],
                                  'pinky2': [(0., 1.4), (2., 1.4)]}
        refresh_result = wait(refresh.get_result_async())
        assert refresh_result.status == GoalStatus.STATUS_SUCCEEDED, refresh_result.result.message
        assert refresh_result.result.success
        assert [name for name, _ in starts].count('pinky1') == 2, starts
        assert [name for name, _ in starts].count('pinky2') == 2, starts
        assert server._approach_paths['pinky2'] == [(0., 1.), (2., 1.)]
        assert any(view['yielding'] == 'pinky2' for view in views)
        assert any(view['state'] == 'replanning' for view in views)

        # Actual ROS goal chain, fake motion only: two collinear opposing
        # routes have blocked starts in BOTH orderings. Move pinky2 to a
        # registered off-route haven, then execute both original goals.
        nav1.duration = nav2.duration = .8
        nav1.move_to_goal = nav2.move_to_goal = True
        nav1.pose.pose.pose = park('pinky1',0.,0.).parking_goal.pose
        nav2.pose.pose.pose = park('pinky2',1.,0.).parking_goal.pose
        nav1.publish_pose(); nav2.publish_pose()
        points = PoseArray(); points.header.frame_id='map'
        points.poses=[park('pinky2',1.,1.).parking_goal.pose]
        point_publisher=client_node.create_publisher(PoseArray,'/central_control/yield_points',
            QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        point_publisher.publish(points)
        deadline=time.monotonic()+3.
        while not server._yield_points and time.monotonic()<deadline:
            time.sleep(.01)
        assert server._yield_points
        time.sleep(.15)
        starts.clear()
        escape=ExecuteMultiRobotMission.Goal()
        escape.tasks=[park('pinky1',2.,0.),park('pinky2',-1.,0.)]
        escape_handle=wait(client.send_goal_async(escape))
        assert escape_handle.accepted
        escape_result=wait(escape_handle.get_result_async(),timeout=15.)
        assert escape_result.status==GoalStatus.STATUS_SUCCEEDED,escape_result.result.message
        assert escape_result.result.success
        assert starts[0][0]=='pinky2',starts
        assert [name for name,_ in starts].count('pinky2')==2,starts
        assert [name for name,_ in starts].count('pinky1')==1,starts
        assert nav2.pose.pose.pose.position.x==-1.,'Original goal not restored'
        assert any(view['state']=='detouring' for view in views)
        print(f'PASS: isolated ROS Domain {domain}; both goals began {skew:.3f}s apart; '
              f'crossing goal cancelled/resumed after {delay:.3f}s; unsafe staging rejected; '
              'parked robot off old route did not cancel remaining task; '
              'persistent deviation refreshed routes; registered-haven evacuation '
              'confirmed then both original goals completed.')
    finally:
        executor.shutdown()
        for node in (client_node, nav2, nav1, server):
            node.destroy_node()
        rclpy.shutdown()
        worker.join(timeout=2.0)


if __name__ == '__main__':
    main()
