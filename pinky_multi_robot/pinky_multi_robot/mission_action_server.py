"""Central mission server with Nav2 path preflight and reactive yielding."""
import math
import time
from typing import Dict

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from pinky_multi_robot.path_conflicts import path_distance, point_path_distance, remaining_path, path_prefix


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
        self.declare_parameter('path_clearance', 0.35)
        self.declare_parameter('yield_lookahead', 0.8)
        self.declare_parameter('route_deviation_limit', 0.25)
        self.declare_parameter('planning_timeout', 15.0)
        self.declare_parameter('pose_max_age', 10.0)
        self.declare_parameter('allow_namespaced_map_frames', False)

        self._nav_clients: Dict[str, ActionClient] = {}
        self._planner_clients: Dict[str, ActionClient] = {}
        self._poses: Dict[str, PoseWithCovarianceStamped] = {}
        self._active_nav_goals: Dict[str, object] = {}
        self._completed_robots = set()
        self._mission_active = False
        self._mission_reserved = False
        self._mission_robots = set()
        self._pose_receipt = {}
        self._yielding_robot = None
        self._yield_started_ns = None
        self._reserved_robot = None
        self._approach_paths = {}
        self._yield_retry_robots = set()
        self._priority_task_done = False
        self._priority_task_ok = False
        self._safety_fault = None
        self._wait_group = ReentrantCallbackGroup()

        for robot_id in self.get_parameter('known_robots').value:
            robot_id = robot_id.strip('/')
            self.create_subscription(
                PoseWithCovarianceStamped, f'/{robot_id}/tracked_pose',
                lambda message, name=robot_id: self._pose_callback(name, message), 10)
        self.create_timer(0.1, self._collision_check)
        self._server = ActionServer(
            self, ExecuteMultiRobotMission,
            self.get_parameter('action_name').value,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info('Ready for /execute_multi_robot_mission.')

    async def _sleep(self, seconds):
        """Non-blocking delay using a rclpy Future, not an asyncio Future."""
        if seconds <= 0:
            return
        future = self.executor.create_future()
        timer = self.create_timer(
            seconds,
            lambda: future.set_result(None) if not future.done() else None,
            callback_group=self._wait_group)
        try:
            await future
        finally:
            self.destroy_timer(timer)

    def _pose_callback(self, robot_id, message):
        self._poses[robot_id] = message
        self._pose_receipt[robot_id] = time.monotonic()

    def goal_callback(self, request):
        if self._mission_reserved:
            self.get_logger().warn('A mission is already active; cancel or finish it first.')
            return GoalResponse.REJECT

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
        self._mission_reserved = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        for nav_goal in list(self._active_nav_goals.values()):
            nav_goal.cancel_goal_async()
        return CancelResponse.ACCEPT

    def _client(self, robot_id):
        if robot_id not in self._nav_clients:
            self._nav_clients[robot_id] = ActionClient(
                self, NavigateToPose, f'/{robot_id}/navigate_to_pose')
        return self._nav_clients[robot_id]

    def _planner(self, robot_id):
        if robot_id not in self._planner_clients:
            self._planner_clients[robot_id] = ActionClient(
                self, ComputePathToPose, f'/{robot_id}/compute_path_to_pose')
        return self._planner_clients[robot_id]

    def _map_frame(self, robot_id, frame):
        if (self.get_parameter('allow_namespaced_map_frames').value and
                frame == f'{robot_id}/map'):
            return 'map'
        return frame

    async def _await_planning(self, future, deadline):
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError('Nav2 path planning timed out')
            await self._sleep(0.05)
        return future.result()

    async def _compute_path(self, robot_id, start, goal):
        client = self._planner(robot_id)
        deadline = time.monotonic() + float(self.get_parameter('planning_timeout').value)
        while not client.server_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{robot_id}: compute_path_to_pose unavailable')
            await self._sleep(0.1)
        request = ComputePathToPose.Goal()
        request.start = start
        request.goal = goal
        request.use_start = True
        handle = await self._await_planning(client.send_goal_async(request), deadline)
        if not handle.accepted:
            raise RuntimeError(f'{robot_id}: planner rejected path request')
        try:
            response = await self._await_planning(handle.get_result_async(), deadline)
        except TimeoutError:
            handle.cancel_goal_async()
            raise
        if response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.path.poses:
            raise RuntimeError(
                f'{robot_id}: no usable Nav2 path ({response.result.error_msg})')
        path = response.result.path
        frame = path.header.frame_id or path.poses[0].header.frame_id
        if not frame or self._map_frame(robot_id, frame) != self._map_frame(
                robot_id, goal.header.frame_id):
            raise RuntimeError(f'{robot_id}: planner path frame does not match goal')
        return [(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]

    async def _preflight(self, tasks):
        """Plan each task before moving either robot; reserve overlapping routes."""
        moving = [task for task in tasks if task.task_type != RobotTask.STOP]
        if len(moving) != 2:
            return None
        planned = {}
        starts = {}
        max_age = float(self.get_parameter('pose_max_age').value)
        for task in moving:
            robot_id = task.robot_id.strip('/')
            pose = self._poses.get(robot_id)
            if pose is None or time.monotonic() - self._pose_receipt.get(robot_id, 0) > max_age:
                raise RuntimeError(f'{robot_id}: tracked pose missing or stale; check localization READY, scan and map-to-base TF')
            start = PoseStamped()
            start.header = pose.header
            start.pose = pose.pose.pose
            starts[robot_id] = (start.pose.position.x, start.pose.position.y)
            goals = ([task.parking_goal] if task.task_type == RobotTask.PARK
                     else list(task.patrol_waypoints))
            if task.task_type == RobotTask.PATROL and task.patrol_laps != 1:
                goals.append(task.patrol_waypoints[0])
            points = []
            for goal in goals:
                if self._map_frame(robot_id, start.header.frame_id) != self._map_frame(
                        robot_id, goal.header.frame_id):
                    raise RuntimeError(f'{robot_id}: AMCL and goal map frames differ')
                leg = await self._compute_path(robot_id, start, goal)
                points.extend(leg)
                start = goal
            planned[robot_id] = points
        for robot_id in planned:
            if time.monotonic() - self._pose_receipt.get(robot_id, 0) > max_age:
                raise RuntimeError(f'{robot_id}: tracked pose became stale during planning')
        first, second = [task.robot_id.strip('/') for task in moving]
        if self._map_frame(first, self._poses[first].header.frame_id) != self._map_frame(
                second, self._poses[second].header.frame_id):
            raise RuntimeError('Robot paths do not use the same map frame')
        clearance = float(self.get_parameter('path_clearance').value)
        separation = path_distance(planned[first], planned[second])
        if separation >= clearance:
            self.get_logger().info(
                f'Preflight: independent routes, minimum separation {separation:.2f}m.')
            return None
        priority = self.get_parameter('priority_robot').value.strip('/')
        if priority not in planned:
            raise RuntimeError('Configured priority robot is not in this mission')
        priority_task = next(task for task in moving
                             if task.robot_id.strip('/') == priority)
        if (priority_task.task_type == RobotTask.PATROL and
                priority_task.patrol_laps == 0):
            raise RuntimeError(
                'Overlapping routes cannot be reserved behind an unlimited priority patrol')
        yielding = second if priority == first else first
        if (point_path_distance(starts[yielding], planned[priority]) < clearance or
                point_path_distance(planned[priority][-1], planned[yielding]) < clearance):
            raise RuntimeError(
                'Routes conflict at a waiting/parking position; no safe ordering. '
                'Choose a different staging or parking point.')
        if all(task.task_type == RobotTask.PARK for task in moving):
            self._approach_paths = planned
            self.get_logger().info(
                'Preflight: overlapping parking routes; both approach, yield before conflict zone.')
            return None
        self.get_logger().warn(
            f'Preflight: routes within {separation:.2f}m; {yielding} waits for {priority}.')
        return yielding

    def _distance(self):
        priority = self.get_parameter('priority_robot').value.strip('/')
        others = [name.strip('/') for name in self.get_parameter('known_robots').value
                  if name.strip('/') != priority]
        if len(others) != 1 or priority not in self._poses or others[0] not in self._poses:
            return None, priority, others[0] if others else None
        now = time.monotonic()
        if any(now - self._pose_receipt.get(robot, 0.0) > 2.5
               for robot in (priority, others[0])):
            return None, priority, others[0]
        a = self._poses[priority].pose.pose.position
        b = self._poses[others[0]].pose.pose.position
        return math.hypot(a.x - b.x, a.y - b.y), priority, others[0]

    def _collision_check(self):
        if not self._mission_active or self._safety_fault is not None:
            return
        if not self._approach_paths and self._yielding_robot is None and (
                len(self._mission_robots) != 2 or
                not self._mission_robots.issubset(self._active_nav_goals)):
            return
        distance, priority, yielding = self._distance()
        if distance is None:
            if self._active_nav_goals:
                self._safety_fault = 'Lost fresh tracked poses during dual-robot navigation'
                self.get_logger().error(self._safety_fault + '; cancelling both Nav2 goals.')
                for nav_goal in list(self._active_nav_goals.values()):
                    nav_goal.cancel_goal_async()
            return
        route_conflict = False
        if self._approach_paths:
            remaining = {}
            for robot, path in self._approach_paths.items():
                pos = self._poses[robot].pose.pose.position
                point = (pos.x, pos.y)
                if point_path_distance(point, path) > self.get_parameter('route_deviation_limit').value:
                    self._safety_fault = f'{robot}: left checked route; replan mission before proceeding'
                    for nav_goal in list(self._active_nav_goals.values()):
                        nav_goal.cancel_goal_async()
                    self.get_logger().error(self._safety_fault)
                    return
                remaining[robot] = remaining_path(point, path)
            horizon = float(self.get_parameter('yield_lookahead').value)
            # Include a stopping/communication margin, not only body clearance.
            clearance = max(float(self.get_parameter('path_clearance').value),
                            float(self.get_parameter('conflict_distance').value))
            route_conflict = path_distance(
                path_prefix(remaining[yielding], horizon), remaining[priority]) < clearance
            if route_conflict:
                pos = self._poses[yielding].pose.pose.position
                if point_path_distance((pos.x, pos.y), remaining[priority]) < clearance:
                    self._safety_fault = 'No safe waiting clearance; cancelling both goals'
                    for nav_goal in list(self._active_nav_goals.values()):
                        nav_goal.cancel_goal_async()
                    self.get_logger().error(self._safety_fault)
                    return
        now_ns = self.get_clock().now().nanoseconds
        if self._yielding_robot is None and (route_conflict or distance <= self.get_parameter('conflict_distance').value):
            self._yielding_robot = yielding
            self._yield_started_ns = now_ns
            nav_goal = self._active_nav_goals.get(yielding)
            if nav_goal is not None:
                self._yield_retry_robots.add(yielding)
                nav_goal.cancel_goal_async()
            self.get_logger().warn(
                f'Conflict {distance:.2f}m: {yielding} yields; {priority} keeps priority.')
            return
        if self._yielding_robot is None:
            return
        elapsed = (now_ns - self._yield_started_ns) / 1e9
        can_resume = (not route_conflict and
                      (priority in self._completed_robots or distance >= self.get_parameter('resume_distance').value))
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
        while not mission_goal.is_cancel_requested and self._safety_fault is None:
            while self._reserved_robot == robot_id and not mission_goal.is_cancel_requested:
                if self._priority_task_done:
                    if not self._priority_task_ok:
                        self.get_logger().error(
                            f'{robot_id}: priority robot failed; reserved route stays closed.')
                        return False
                    self._reserved_robot = None
                    break
                await self._sleep(0.1)
            while (self._yielding_robot == robot_id and
                   not mission_goal.is_cancel_requested and self._safety_fault is None):
                await self._sleep(0.1)
            if mission_goal.is_cancel_requested or self._safety_fault is not None:
                return False
            while not client.wait_for_server(timeout_sec=0.2):
                if mission_goal.is_cancel_requested:
                    return False
                await self._sleep(0.05)
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
            if robot_id in self._yield_retry_robots or self._yielding_robot == robot_id:
                self._yield_retry_robots.discard(robot_id)
                # The collision timer cancelled this goal. Wait, then resend it.
                continue
            if result.status != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().error(
                    f'{robot_id}: Nav2 finished with status {result.status}; '
                    f'{result.result.error_msg}')
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
                    await self._sleep(task.pause_seconds)
        self._completed_robots.add(robot_id)
        self._feedback(goal, task, 'patrol_complete', completed, total)
        return True

    async def _run_task_tracked(self, goal, task):
        robot_id = task.robot_id.strip('/')
        try:
            ok = await self._run_task(goal, task)
            return ok
        finally:
            if robot_id == self.get_parameter('priority_robot').value.strip('/'):
                self._priority_task_done = True
                self._priority_task_ok = robot_id in self._completed_robots

    async def execute_callback(self, goal_handle):
        self._active_nav_goals.clear()
        self._approach_paths = {}
        self._yield_retry_robots.clear()
        self._completed_robots.clear()
        self._yielding_robot = None
        self._yield_started_ns = None
        self._reserved_robot = None
        self._priority_task_done = False
        self._priority_task_ok = False
        self._safety_fault = None
        self._mission_robots = {task.robot_id.strip('/') for task in goal_handle.request.tasks}
        self._mission_active = True
        outcomes = []
        error = None
        try:
            self._reserved_robot = await self._preflight(goal_handle.request.tasks)
            if not goal_handle.is_cancel_requested:
                tasks = [self.executor.create_task(self._run_task_tracked(goal_handle, task))
                         for task in goal_handle.request.tasks]
                outcomes = [await task for task in tasks]
        except Exception as exception:
            error = str(exception)
            self.get_logger().error(f'Mission preflight/execution failed: {error}')
            for nav_goal in list(self._active_nav_goals.values()):
                nav_goal.cancel_goal_async()
        finally:
            self._mission_active = False
            self._yielding_robot = None
            self._yield_started_ns = None
            self._reserved_robot = None
            self._mission_robots.clear()
            self._mission_reserved = False
        result = ExecuteMultiRobotMission.Result()
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success, result.message = False, 'Mission cancelled.'
        elif error is not None:
            goal_handle.abort()
            result.success, result.message = False, error
        elif self._safety_fault is not None:
            goal_handle.abort()
            result.success, result.message = False, self._safety_fault
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
