"""Send one clicked goal to two Pinky Nav2 stacks with simple yielding.

The node never publishes cmd_vel.  It only sends/cancels NavigateToPose actions,
so each robot keeps its own Nav2 collision checking and emergency behavior.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CoordinationState(Enum):
    IDLE = 'idle'
    MOVING = 'moving'
    YIELDING = 'yielding'


@dataclass
class Robot:
    """ROS interfaces and the latest pose for one robot."""

    name: str
    namespace: str
    goal_frame: str
    client: ActionClient
    pose: Optional[PoseWithCovarianceStamped] = None
    goal_handle: object = None
    completed: bool = False


class TwoPinkyCoordinator(Node):
    """Give both robots one point goal and give one robot priority at conflicts."""

    def __init__(self):
        super().__init__('two_pinky_coordinator')
        self.declare_parameter('robot_1_namespace', 'pinky1')
        self.declare_parameter('robot_2_namespace', 'pinky2')
        self.declare_parameter('robot_1_goal_frame', 'pinky1/map')
        self.declare_parameter('robot_2_goal_frame', 'pinky2/map')
        self.declare_parameter('priority_robot', 'pinky1')
        self.declare_parameter('target_topic', '/clicked_point')
        self.declare_parameter('conflict_distance', 0.55)
        self.declare_parameter('resume_distance', 0.80)
        self.declare_parameter('minimum_yield_seconds', 2.0)

        robot_1_ns = self._namespace('robot_1_namespace')
        robot_2_ns = self._namespace('robot_2_namespace')
        self.robots = {
            robot_1_ns: Robot(
                robot_1_ns,
                robot_1_ns,
                self.get_parameter('robot_1_goal_frame').value,
                ActionClient(self, NavigateToPose,
                             self._topic(robot_1_ns, 'navigate_to_pose'))),
            robot_2_ns: Robot(
                robot_2_ns,
                robot_2_ns,
                self.get_parameter('robot_2_goal_frame').value,
                ActionClient(self, NavigateToPose,
                             self._topic(robot_2_ns, 'navigate_to_pose'))),
        }
        priority = self._strip(self.get_parameter('priority_robot').value)
        if priority not in self.robots:
            raise ValueError('priority_robot must be robot_1_namespace or '
                             'robot_2_namespace')
        self.priority_name = priority
        self.yield_name = next(name for name in self.robots if name != priority)

        for robot in self.robots.values():
            self.create_subscription(
                PoseWithCovarianceStamped,
                self._topic(robot.namespace, 'amcl_pose'),
                lambda msg, name=robot.name: self._pose_callback(name, msg),
                10)

        target_topic = self.get_parameter('target_topic').value
        self.create_subscription(PointStamped, target_topic,
                                 self._target_callback, 10)
        self.timer = self.create_timer(0.2, self._coordinate)
        self.state = CoordinationState.IDLE
        self.target: Optional[PointStamped] = None
        self.yield_started_ns: Optional[int] = None
        self.get_logger().info(
            f'Ready. Publish Point to {target_topic}; priority robot: {self.priority_name}')

    def _strip(self, name: str) -> str:
        return str(name).strip('/')

    def _namespace(self, parameter: str) -> str:
        value = self._strip(self.get_parameter(parameter).value)
        if not value:
            raise ValueError(parameter + ' must not be empty')
        return value

    def _topic(self, namespace: str, suffix: str) -> str:
        return '/' + namespace + '/' + suffix

    def _pose_callback(self, name: str, message: PoseWithCovarianceStamped):
        self.robots[name].pose = message

    def _target_callback(self, message: PointStamped):
        if not math.isfinite(message.point.x) or not math.isfinite(message.point.y):
            self.get_logger().error('Ignored target with non-finite coordinates.')
            return
        self.target = message
        self.state = CoordinationState.MOVING
        self.yield_started_ns = None
        for robot in self.robots.values():
            robot.completed = False
            self._send_goal(robot)
        self.get_logger().info(
            f'New shared target: x={message.point.x:.3f} y={message.point.y:.3f}')

    def _send_goal(self, robot: Robot):
        if self.target is None:
            return
        if not robot.client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn(f'No Nav2 action server: {robot.name}')
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.header.frame_id = robot.goal_frame
        goal.pose.pose.position.x = self.target.point.x
        goal.pose.pose.position.y = self.target.point.y
        goal.pose.pose.orientation.w = 1.0
        future = robot.client.send_goal_async(goal)
        future.add_done_callback(
            lambda result, name=robot.name: self._goal_response(name, result))

    def _goal_response(self, name, future):
        robot = self.robots[name]
        try:
            robot.goal_handle = future.result()
        except Exception as error:  # Action transport failure.
            self.get_logger().error(f'Goal send failed for {name}: {error}')
            return
        if not robot.goal_handle.accepted:
            self.get_logger().error(f'Goal rejected by {name} Nav2.')
            return
        result_future = robot.goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, robot_name=name: self._goal_result(robot_name, result))

    def _goal_result(self, name, future):
        robot = self.robots[name]
        try:
            status = future.result().status
        except Exception as error:
            self.get_logger().error(f'Goal result failed for {name}: {error}')
            return
        # action_msgs/GoalStatus.STATUS_SUCCEEDED is 4.
        if status == 4:
            robot.completed = True
            self.get_logger().info(f'{name} reached its goal.')
        else:
            self.get_logger().warn(f'{name} goal ended with status {status}.')

    def _distance(self) -> Optional[float]:
        first = self.robots[self.priority_name].pose
        second = self.robots[self.yield_name].pose
        if first is None or second is None:
            return None
        a = first.pose.pose.position
        b = second.pose.pose.position
        return math.hypot(a.x - b.x, a.y - b.y)

    def _coordinate(self):
        if self.target is None or self.state == CoordinationState.IDLE:
            return
        distance = self._distance()
        if distance is None:
            return
        conflict = self.get_parameter('conflict_distance').value
        resume = self.get_parameter('resume_distance').value
        now_ns = self.get_clock().now().nanoseconds
        priority = self.robots[self.priority_name]
        yielding = self.robots[self.yield_name]

        if self.state == CoordinationState.MOVING and distance <= conflict:
            self.state = CoordinationState.YIELDING
            self.yield_started_ns = now_ns
            if yielding.goal_handle is not None:
                yielding.goal_handle.cancel_goal_async()
            self.get_logger().warn(
                f'Conflict {distance:.2f}m: {yielding.name} yields; {priority.name} keeps priority.')
            return

        if self.state != CoordinationState.YIELDING:
            return
        elapsed = (now_ns - self.yield_started_ns) / 1e9
        can_resume = priority.completed or distance >= resume
        minimum = self.get_parameter('minimum_yield_seconds').value
        if can_resume and elapsed >= minimum:
            self.state = CoordinationState.MOVING
            yielding.completed = False
            self._send_goal(yielding)
            self.get_logger().info(f'{yielding.name} resumes its shared goal.')


def main(args=None):
    rclpy.init(args=args)
    node = TwoPinkyCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
