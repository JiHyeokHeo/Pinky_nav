"""Central mission server with Nav2 path preflight and reactive yielding."""
import math
import json
import time
from copy import deepcopy
from typing import Dict

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String

from pinky_interfaces.action import ExecuteMultiRobotMission
from pinky_interfaces.msg import RobotTask
from pinky_multi_robot.path_conflicts import path_distance, point_path_distance, remaining_path, path_prefix
from pinky_multi_robot.mission_view import VIEW_TOPIC, mission_snapshot
from pinky_multi_robot.yield_detour import YieldDetourMixin


class MissionActionServer(YieldDetourMixin, Node):
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
        self.declare_parameter('route_deviation_confirm_seconds', 1.0)
        self.declare_parameter('yield_detour_timeout', 60.0)
        self.declare_parameter('yield_arrival_tolerance', 0.15)
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
        self._route_deviation_since = {}
        self._route_refresh_pending = False
        self._route_refresh_task = None
        self._route_revision = 0
        self._nav_targets = {}
        self._mission_generation = 0
        self._effective_priority_robot = None
        self._yield_points = []
        self._detour_attempted = set()
        self._detour_robot = None
        self._detour_point = None
        self._detour_work_active = False
        self._pending_detour_send = None
        self._yield_retry_robots = set()
        self._priority_task_done = False
        self._priority_task_ok = False
        self._safety_fault = None
        self._wait_group = ReentrantCallbackGroup()
        self._display_paths = {}
        self._display_frames = {}
        self._task_states = {}
        self._diagnostic_state = 'idle'
        self._mission_view_reason = 'No checked combined mission yet.'
        self._view_publisher = self.create_publisher(
            String, VIEW_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(.5, self._publish_mission_view)
        self.create_subscription(PoseArray, '/central_control/yield_points',
            self._yield_points_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

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

    def _publish_mission_view(self):
        self._view_publisher.publish(String(data=json.dumps(mission_snapshot(self), allow_nan=False)))

    def _yield_points_callback(self, message):
        if self._mission_active:
            self.get_logger().warn('Yield-point update ignored during active mission.')
            return
        if message.header.frame_id != 'map' or len(message.poses) > 32:
            self.get_logger().warn('Yield points require common map frame and at most 32 poses.')
            return
        points = []
        for pose in message.poses:
            values = [pose.position.x, pose.position.y, pose.orientation.x,
                      pose.orientation.y, pose.orientation.z, pose.orientation.w]
            if not all(math.isfinite(v) for v in values) or abs(sum(v*v for v in values[2:])-1.) > .01:
                self.get_logger().warn('Invalid yield-point pose ignored.')
                return
            goal = PoseStamped()
            goal.header, goal.pose = deepcopy(message.header), deepcopy(pose)
            points.append(goal)
        self._yield_points = points
        self._mission_view_reason = f'Registered {len(points)} yield points; Nav2 checks reachability when needed.'
        self._publish_mission_view()

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
        if (self._mission_reserved or self._active_nav_goals or self._detour_work_active or
                getattr(self, '_pending_detour_send', None) is not None):
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
            self._display_paths[robot_id] = points
            self._display_frames[robot_id] = self._map_frame(robot_id, goals[0].header.frame_id)
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
            self._mission_view_reason = f'Independent Nav2 routes: separation {separation:.2f}m.'
            self.get_logger().info(
                f'Preflight: independent routes, minimum separation {separation:.2f}m.')
            return None
        priority = self._effective_priority_robot or self.get_parameter('priority_robot').value.strip('/')
        if priority not in planned:
            raise RuntimeError('Configured priority robot is not in this mission')
        priority_task = next(task for task in moving
                             if task.robot_id.strip('/') == priority)
        if (priority_task.task_type == RobotTask.PATROL and
                priority_task.patrol_laps == 0):
            raise RuntimeError(
                'Overlapping routes cannot be reserved behind an unlimited priority patrol')
        yielding = second if priority == first else first
        waiting_distance = point_path_distance(starts[yielding], planned[priority])
        parking_distance = point_path_distance(planned[priority][-1], planned[yielding])
        if waiting_distance < clearance or parking_distance < clearance:
            reverse_wait = point_path_distance(starts[priority], planned[yielding])
            reverse_park = point_path_distance(planned[yielding][-1], planned[priority])
            if reverse_wait >= clearance and reverse_park >= clearance:
                priority, yielding = yielding, priority
                self._effective_priority_robot = priority
                self._mission_view_reason = f'Safer passage order selected: {priority} first.'
            elif self._yield_points:
                # Try only an ordering whose final parking point does not
                # permanently block the other route. Both robots hold first.
                options = [(priority, yielding, parking_distance),
                           (yielding, priority, reverse_park)]
                escape = next(((a,b) for a,b,d in options
                               if d >= clearance and b not in self._detour_attempted), None)
                if escape is not None:
                    priority, yielding = escape
                    if yielding in self._completed_robots:
                        raise RuntimeError('Cannot relocate an already completed robot automatically')
                    self._effective_priority_robot = priority
                    task = next(t for t in moving if t.robot_id.strip('/') == yielding)
                    original = task.parking_goal if task.task_type == RobotTask.PARK else task.patrol_waypoints[0]
                    self._route_refresh_pending = True
                    try:
                        await self._evacuate(priority, yielding, planned[priority], original)
                    finally:
                        self._route_refresh_pending = False
                    if self._mission_goal.is_cancel_requested:
                        raise RuntimeError('Mission cancelled during evacuation')
                    return await self._preflight(tasks)  # Fresh pose + original goals.
                raise RuntimeError('No safe ordering even with registered yield points: final parking blocks remaining route.')
            else:
                raise RuntimeError(
                    'Routes conflict at a waiting/parking position; no safe ordering. '
                    f'{yielding} waiting-to-{priority}-route={waiting_distance:.2f}m, '
                    f'{priority} parking-to-{yielding}-route={parking_distance:.2f}m '
                    f'(required {clearance:.2f}m). Register a yield point or choose different parking.')
        self._effective_priority_robot = priority
        priority_task = next(task for task in moving if task.robot_id.strip('/') == priority)
        if priority_task.task_type == RobotTask.PATROL and priority_task.patrol_laps == 0:
            raise RuntimeError('Overlapping routes require a finite priority task')
        if all(task.task_type == RobotTask.PARK for task in moving):
            self._approach_paths = planned
            self._mission_view_reason = 'Overlapping Nav2 routes: approach together, yield before conflict.'
            self.get_logger().info(
                'Preflight: overlapping parking routes; both approach, yield before conflict zone.')
            return None
        self.get_logger().warn(
            f'Preflight: routes within {separation:.2f}m; {yielding} waits for {priority}.')
        self._mission_view_reason = f'{yielding} waits until {priority} finishes the reserved route.'
        return yielding

    def _distance(self):
        priority = self._effective_priority_robot or self.get_parameter('priority_robot').value.strip('/')
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
        # Refresh owns a paused pair; never compare against its old routes.
        if self._route_refresh_pending:
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
        # A parked lower-priority robot cannot yield. The moving robot waits
        # instead; completed robots remain physical obstacles, not old routes.
        if yielding in self._completed_robots and priority not in self._completed_robots:
            priority, yielding = yielding, priority
        route_conflict = False
        if self._approach_paths:
            remaining = {}
            for robot, path in self._approach_paths.items():
                pos = self._poses[robot].pose.pose.position
                point = (pos.x, pos.y)
                if robot in self._completed_robots:
                    remaining[robot] = [point]
                    self._route_deviation_since.pop(robot, None)
                    continue
                deviation = point_path_distance(point, path)
                if deviation > self.get_parameter('route_deviation_limit').value:
                    now = time.monotonic()
                    since = self._route_deviation_since.setdefault(robot, now)
                    if now-since >= self.get_parameter('route_deviation_confirm_seconds').value:
                        self._request_route_refresh(robot, deviation)
                        return
                else:
                    self._route_deviation_since.pop(robot, None)
                # Current pose -> checked remaining route is included while
                # deviation is being confirmed, rather than ignoring the gap.
                remaining[robot] = [point] + remaining_path(point, path)
            horizon = float(self.get_parameter('yield_lookahead').value)
            # Include a stopping/communication margin, not only body clearance.
            clearance = max(float(self.get_parameter('path_clearance').value),
                            float(self.get_parameter('conflict_distance').value))
            route_conflict = path_distance(
                path_prefix(remaining[yielding], horizon), remaining[priority]) < clearance
            if route_conflict:
                pos = self._poses[yielding].pose.pose.position
                if point_path_distance((pos.x, pos.y), remaining[priority]) < clearance:
                    if (getattr(self, '_yield_points', []) and
                            yielding not in self._completed_robots and
                            yielding not in self._detour_attempted):
                        self._request_detour(priority, yielding, remaining[priority])
                        return
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
            self._mission_view_reason = f'{yielding} yields to {priority}; robot separation {distance:.2f}m.'
            return
        if self._yielding_robot is None:
            return
        elapsed = (now_ns - self._yield_started_ns) / 1e9
        can_resume = (not route_conflict and
                      distance > self.get_parameter('conflict_distance').value and
                      (priority in self._completed_robots or distance >= self.get_parameter('resume_distance').value))
        if can_resume and elapsed >= self.get_parameter('minimum_yield_seconds').value:
            resumed = self._yielding_robot
            self._yielding_robot = None
            self._yield_started_ns = None
            self.get_logger().info(f'{resumed} resumes its interrupted Nav2 goal.')
            self._mission_view_reason = f'{resumed} resumes: checked conflict cleared.'

    def _request_route_refresh(self, robot, deviation):
        """Pause active goals, then check new Nav2 routes without aborting mission."""
        if self._route_refresh_pending:
            return
        self._route_refresh_pending = True
        self._diagnostic_state = 'replanning'
        self._mission_view_reason = f'{robot}: sustained deviation {deviation:.2f}m; refreshing Nav2 routes.'
        self._route_revision += 1
        goals = deepcopy(self._nav_targets)
        generation = self._mission_generation
        for name, goal in list(self._active_nav_goals.items()):
            self._yield_retry_robots.add(name)
            goal.cancel_goal_async()
        self.get_logger().warn(
            f'{robot}: sustained route deviation {deviation:.2f}m; '
            'pausing active goals to refresh checked Nav2 routes.')
        self._route_refresh_task = self.executor.create_task(
            self._refresh_routes(generation, goals))

    async def _refresh_routes(self, generation, goals):
        """Bounded refresh; stale poses/planner failure keep robots stopped."""
        def current():
            return (generation == self._mission_generation and self._mission_active
                    and not self._mission_goal.is_cancel_requested)
        try:
            deadline = time.monotonic()+float(self.get_parameter('planning_timeout').value)
            while self._active_nav_goals and current():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Nav2 cancellation timed out during route refresh')
                await self._sleep(.05)
            if not current() or self._safety_fault is not None:
                return
            refreshed = {}
            for robot in self._approach_paths:
                if robot in self._completed_robots:
                    continue
                pose = self._poses.get(robot)
                if pose is None or time.monotonic()-self._pose_receipt.get(robot, 0.) > 2.5:
                    raise RuntimeError(f'{robot}: fresh tracked pose required for route refresh')
                if robot not in goals:
                    raise RuntimeError(f'{robot}: active goal unavailable for route refresh')
                start = PoseStamped()
                start.header = deepcopy(pose.header)
                start.pose = deepcopy(pose.pose.pose)
                refreshed[robot] = await self._compute_path(robot, start, goals[robot])
                if not current() or self._safety_fault is not None:
                    return
            # Atomic installation: never use a new route for just one robot.
            self._approach_paths.update(refreshed)
            if hasattr(self, '_display_paths'):
                self._display_paths.update(refreshed)
            self._route_deviation_since.clear()
            self._route_refresh_pending = False
            self._diagnostic_state = 'navigating'
            self._yielding_robot = None
            self._yield_started_ns = None
            self._collision_check()  # Parked position and new conflicts BEFORE resending.
            if self._safety_fault is None:
                if self._yielding_robot is None:
                    self._mission_view_reason = 'Nav2 routes refreshed; remaining tasks resume.'
                self.get_logger().info('Checked Nav2 routes refreshed; remaining tasks may resume.')
        except Exception as error:
            if current():
                self._safety_fault = f'Route refresh failed: {error}'
                self.get_logger().error(self._safety_fault)
                for goal in list(self._active_nav_goals.values()):
                    goal.cancel_goal_async()
        finally:
            if generation == self._mission_generation:
                self._route_refresh_pending = False

    def _feedback(self, goal, task, state, completed=0, total=0):
        self._task_states[task.robot_id.strip('/')] = state
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
        self._nav_targets[robot_id] = deepcopy(pose)
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
            while ((self._yielding_robot == robot_id or self._route_refresh_pending) and
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
            route_revision = self._route_revision
            nav_goal = await client.send_goal_async(request)
            if not nav_goal.accepted:
                self.get_logger().error(f'{robot_id}: Nav2 rejected goal.')
                return False
            self._active_nav_goals[robot_id] = nav_goal
            # A goal response may arrive after the timer started a refresh.
            route_changed = route_revision != self._route_revision
            must_yield = self._yielding_robot == robot_id
            if self._route_refresh_pending or route_changed or must_yield or self._safety_fault is not None or mission_goal.is_cancel_requested:
                if self._route_refresh_pending or route_changed or must_yield:
                    self._yield_retry_robots.add(robot_id)
                nav_goal.cancel_goal_async()
            try:
                result = await nav_goal.get_result_async()
            finally:
                if self._active_nav_goals.get(robot_id) is nav_goal:
                    del self._active_nav_goals[robot_id]
            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self._yield_retry_robots.discard(robot_id)
                self._nav_targets.pop(robot_id, None)
                return True  # Arrival wins a simultaneous refresh/cancel race.
            if (result.status == GoalStatus.STATUS_CANCELED and
                    (robot_id in self._yield_retry_robots or self._yielding_robot == robot_id)):
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
            if robot_id == (self._effective_priority_robot or self.get_parameter('priority_robot').value.strip('/')):
                self._priority_task_done = True
                self._priority_task_ok = robot_id in self._completed_robots

    async def execute_callback(self, goal_handle):
        self._mission_generation += 1
        self._mission_goal = goal_handle
        self._effective_priority_robot = None
        self._detour_attempted.clear()
        self._detour_robot = self._detour_point = None
        self._display_paths.clear()
        self._display_frames.clear()
        self._task_states = {task.robot_id.strip('/'): 'planning' for task in goal_handle.request.tasks}
        self._diagnostic_state = 'planning'
        self._mission_view_reason = 'Computing Nav2 paths and checking shared route clearance.'
        self._nav_targets.clear()
        self._route_deviation_since.clear()
        self._route_refresh_pending = False
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
            self._diagnostic_state = 'navigating'
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
        self._diagnostic_state = ('canceled' if goal_handle.is_cancel_requested else
                                  'succeeded' if result.success else 'failed')
        self._mission_view_reason = result.message
        for robot in self._task_states:
            self._task_states[robot] = ('completed' if robot in self._completed_robots else self._diagnostic_state)
        self._publish_mission_view()  # Retain failed plans/reason for UI inspection.
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
