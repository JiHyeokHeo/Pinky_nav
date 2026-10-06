"""Domain 50 mission server with Nav2 dispatch and simple two-robot yielding."""
import asyncio
import math
from typing import Dict

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask


class MissionActionServer(Node):
    """Run PARK/PATROL tasks concurrently and pause the lower-priority robot."""

    def __init__(self):
        super().__init__('mission_action_server')
        self.declare_parameter('action_name', '/execute_multi_robot_mission')
        self.declare_parameter('known_robots', ['pinky1', 'pinky2'])
        self.declare_parameter('priority_robot', 'pinky1')
        self.declare_parameter('conflict_distance', 0.55)
        self.declare_parameter('resume_distance', 0.80)
        self.declare_parameter('minimum_yield_seconds', 2.0)

        self._clients: Dict[str, ActionClient] = {}
        self._poses: Dict[str, PoseWithCovarianceStamped] = {}
        self._active_nav_goals: Dict[str, object] = {}
        self._completed_robots = set()
        self._mission_active = False
        self._yielding_robot = None
        self._yield_started_ns = None

        for robot_id in self.get_parameter('known_robots').value:
            robot_id = robot_id.strip('/')
            self.create_subscription(
                PoseWithCovarianceStamped, f'/{robot_id}/amcl_pose',
                lambda message, name=robot_id: self._pose_callback(name, message), 10)
        self.create_timer(0.1, self._collision_check)
        self._server = ActionServer(
            self, ExecuteMultiRobotMission,
            self.get_parameter('action_name').value,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info('Ready for /execute_multi_robot_mission on Domain 50.')

    def _pose_callback(self, robot_id, message):
        self._poses[robot_id] = message

    def goal_callback(self, request):
        known = {name.strip('/') for name in self.get_parameter('known_robots').value}
        ids = [task.robot_id.strip('/') for task in request.tasks]
        if not ids or len(ids) != len(set(ids)) or any(name not in known for name in ids):
            self.get_logger().error('Each mission needs one unique known robot_id per task.')
            return GoalResponse.REJECT
        for task in request.tasks:
            if task.task_type == RobotTask.PARK and not task.parking_goal.header.frame_id:
                self.get_logger().error('PARK requires parking_goal.header.frame_id.')
                return GoalResponse.REJECT
            if task.task_type == RobotTask.PATROL and not task.patrol_waypoints:
                self.get_logger().error('PATROL requires at least one waypoint.')
                return GoalResponse.REJECT
            if task.task_type not in (RobotTask.PARK, RobotTask.PATROL, RobotTask.STOP):
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        for nav_goal in list(self._active_nav_goals.values()):
            nav_goal.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _client(self, robot_id):
        if robot_id not in self._clients:
            self._clients[robot_id] = ActionClient(
                self, NavigateToPose, f'/{robot_id}/navigate_to_pose')
        return self._clients[robot_id]

    def _distance(self):
        priority = self.get_parameter('priority_robot').value.strip('/')
        others = [name.strip('/') for name in self.get_parameter('known_robots').value
                  if name.strip('/') != priority]
        if len(others) != 1 or priority not in self._poses or others[0] not in self._poses:
            return None, priority, others[0] if others else None
        a = self._poses[priority].pose.pose.position
        b = self._poses[others[0]].pose.pose.position
        return math.hypot(a.x - b.x, a.y - b.y), priority, others[0]

    def _collision_check(self):
        if not self._mission_active:
            return
        distance, priority, yielding = self._distance()
        if distance is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._yielding_robot is None and distance <= self.get_parameter('conflict_distance').value:
            self._yielding_robot = yielding
            self._yield_started_ns = now_ns
            nav_goal = self._active_nav_goals.get(yielding)
            if nav_goal is not None:
                nav_goal.cancel_goal_async()
            self.get_logger().warn(
                f'Conflict {distance:.2f}m: {yielding} yields; {priority} keeps priority.')
            return
        if self._yielding_robot is None:
            return
        elapsed = (now_ns - self._yield_started_ns) / 1e9
        can_resume = priority in self._completed_robots or distance >= self.get_parameter('resume_distance').value
        if can_resume and elapsed >= self.get_parameter('minimum_yield_seconds').value:
            resumed = self._yielding_robot
            self._yielding_robot = None
            self._yield_started_ns = None
            self.get_logger().info(f'{resumed} resumes its interrupted Nav2 goal.')

    def _feedback(self, goal, task, state, completed=0, total=0):
        feedback = ExecuteMultiRobotMission.Feedback()
        feedback.robot_id = task.robot_id
        feedback.task_type = task.task_type
        feedback.state = state
        feedback.completed_waypoints = completed
        feedback.total_waypoints = total
        goal.publish_feedback(feedback)

    async def _navigate(self, robot_id, pose, mission_goal):
        """Retry the same Nav2 pose after cancellation caused by yielding."""
        client = self._client(robot_id)
        while not mission_goal.is_cancel_requested:
            while self._yielding_robot == robot_id and not mission_goal.is_cancel_requested:
                await asyncio.sleep(0.1)
            if mission_goal.is_cancel_requested:
                return False
            while not client.wait_for_server(timeout_sec=0.2):
                if mission_goal.is_cancel_requested:
                    return False
                await asyncio.sleep(0.05)
            request = NavigateToPose.Goal()
            request.pose = pose
            nav_goal = await client.send_goal_async(request)
            if not nav_goal.accepted:
                self.get_logger().error(f'{robot_id}: Nav2 rejected goal.')
                return False
            self._active_nav_goals[robot_id] = nav_goal
            try:
                result = await nav_goal.get_result_async()
            finally:
                if self._active_nav_goals.get(robot_id) is nav_goal:
                    del self._active_nav_goals[robot_id]
            if self._yielding_robot == robot_id:
                # The collision timer cancelled this goal. Wait, then resend it.
                continue
            return result.status == GoalStatus.STATUS_SUCCEEDED
        return False

    async def _run_task(self, goal, task):
        robot_id = task.robot_id.strip('/')
        if task.task_type == RobotTask.STOP:
            active = self._active_nav_goals.get(robot_id)
            if active is not None:
                active.cancel_goal_async()
            self._feedback(goal, task, 'stopped')
            return True
        if task.task_type == RobotTask.PARK:
            self._feedback(goal, task, 'parking')
            ok = await self._navigate(robot_id, task.parking_goal, goal)
            if ok:
                self._completed_robots.add(robot_id)
            self._feedback(goal, task, 'parked' if ok else 'park_failed', int(ok), 1)
            return ok

        lap = completed = 0
        total = len(task.patrol_waypoints)
        while task.patrol_laps == 0 or lap < task.patrol_laps:
            lap += 1
            for pose in task.patrol_waypoints:
                if goal.is_cancel_requested:
                    return False
                state = 'yielding' if self._yielding_robot == robot_id else f'patrolling lap {lap}'
                self._feedback(goal, task, state, completed, total)
                if not await self._navigate(robot_id, pose, goal):
                    return False
                completed += 1
                if task.pause_seconds > 0:
                    await asyncio.sleep(task.pause_seconds)
        self._completed_robots.add(robot_id)
        self._feedback(goal, task, 'patrol_complete', completed, total)
        return True

    async def execute_callback(self, goal_handle):
        self._active_nav_goals.clear()
        self._completed_robots.clear()
        self._yielding_robot = None
        self._yield_started_ns = None
        self._mission_active = True
        try:
            outcomes = await asyncio.gather(
                *(self._run_task(goal_handle, task) for task in goal_handle.request.tasks))
        finally:
            self._mission_active = False
            self._yielding_robot = None
            self._yield_started_ns = None
        result = ExecuteMultiRobotMission.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success, result.message = False, 'Mission cancelled.'
        elif all(outcomes):
            goal_handle.succeed()
            result.success, result.message = True, 'All robot tasks completed.'
        else:
            goal_handle.abort()
            result.success, result.message = False, 'One or more robot tasks failed.'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MissionActionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
