"""Registered-haven evacuation. No raw velocity/reverse commands."""
import math
import time
from copy import deepcopy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from .path_conflicts import point_path_distance


class YieldDetourMixin:
    def _fresh_start(self, robot):
        pose = self._poses.get(robot)
        if pose is None or time.monotonic()-self._pose_receipt.get(robot, 0.) > 2.5:
            raise RuntimeError(f'{robot}: fresh tracked pose required for detour')
        start = PoseStamped()
        start.header, start.pose = deepcopy(pose.header), deepcopy(pose.pose.pose)
        return start

    async def _select_yield_point(self, priority, robot, priority_route, original_goal):
        start = self._fresh_start(robot)
        other = self._fresh_start(priority).pose.position
        origin = (start.pose.position.x, start.pose.position.y)
        clearance = float(self.get_parameter('path_clearance').value)
        margin = max(clearance, float(self.get_parameter('conflict_distance').value))
        candidates = sorted(self._yield_points, key=lambda p: math.dist(
            origin, (p.pose.position.x, p.pose.position.y)))
        deadline = time.monotonic()+float(self.get_parameter('planning_timeout').value)
        for saved in candidates[:8]:
            if self._mission_goal.is_cancel_requested or time.monotonic() >= deadline:
                break
            xy = (saved.pose.position.x, saved.pose.position.y)
            if math.dist(origin, xy) < .15 or point_path_distance(xy, priority_route) < margin:
                continue
            goal = deepcopy(saved)
            if self._map_frame(robot, start.header.frame_id) != goal.header.frame_id:
                continue
            goal.header.frame_id = start.header.frame_id
            try:
                route = await self._compute_path(robot, start, goal)
                # Other robot holds still: avoid its current body, not its
                # future path. Endpoint must clear its future route as well.
                if point_path_distance((other.x, other.y), route) < clearance:
                    continue
                back = await self._compute_path(robot, goal, original_goal)
                if point_path_distance(priority_route[-1], back) < clearance:
                    continue
            except (RuntimeError, TimeoutError) as error:
                self.get_logger().warn(f'Yield point rejected: {error}')
                continue
            self._fresh_start(priority)
            self._fresh_start(robot)
            return goal, route
        raise RuntimeError(f'{robot}: no reachable registered yield point with checked clearance')

    async def _evacuate(self, priority, robot, priority_route, original_goal):
        generation = self._mission_generation
        mission = self._mission_goal
        def current():
            return (generation == self._mission_generation and self._mission_active
                    and not mission.is_cancel_requested)
        if robot in self._detour_attempted:
            raise RuntimeError(f'{robot}: evacuation already attempted; operator review required')
        self._detour_attempted.add(robot)
        goal, route = await self._select_yield_point(priority, robot, priority_route, original_goal)
        if not current():
            raise RuntimeError('Mission cancelled before evacuation')
        self._detour_robot = robot
        self._detour_point = [goal.pose.position.x, goal.pose.position.y]
        self._diagnostic_state = 'detouring'
        self._mission_view_reason = f'{robot} goes to yield point; {priority} holds.'
        self._display_paths[robot] = route
        self._display_frames[robot] = self._map_frame(robot, goal.header.frame_id)
        deadline = time.monotonic()+float(self.get_parameter('yield_detour_timeout').value)
        handle, terminal = None, False
        try:
            request = NavigateToPose.Goal()
            request.pose = goal
            client = self._client(robot)
            if not client.server_is_ready():
                raise RuntimeError(f'{robot}: navigation proxy unavailable for evacuation')
            send = client.send_goal_async(request)
            try:
                handle = await self._await_planning(send, deadline)
            except TimeoutError:
                # A late acceptance must not become an untracked moving goal.
                self._pending_detour_send = send
                def cancel_late(done):
                    try:
                        late = done.result()
                        if late.accepted:
                            self._active_nav_goals[robot] = late
                            late.cancel_goal_async()
                            def finished(_result):
                                if self._active_nav_goals.get(robot) is late:
                                    self._active_nav_goals.pop(robot)
                            late.get_result_async().add_done_callback(finished)
                    finally:
                        self._pending_detour_send = None
                send.add_done_callback(cancel_late)
                raise
            if not handle.accepted:
                raise RuntimeError(f'{robot}: evacuation goal rejected')
            self._active_nav_goals[robot] = handle
            if not current():
                handle.cancel_goal_async()
            future = handle.get_result_async()
            while not future.done():
                if not current() or time.monotonic() >= deadline:
                    raise RuntimeError('Evacuation cancelled or timed out')
                robot_position = self._fresh_start(robot).pose.position
                other = self._fresh_start(priority).pose.position
                if math.hypot(robot_position.x-other.x, robot_position.y-other.y) < self.get_parameter('path_clearance').value:
                    raise RuntimeError('Evacuation lost stationary-robot clearance')
                await self._sleep(.05)
            terminal = True
            result = future.result()
            if not current() or result.status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f'{robot}: evacuation did not succeed')
            # Result and tracked pose travel separately. Allow a bounded
            # stopped synchronization interval, never more blind movement.
            arrival_deadline = min(deadline,time.monotonic()+2.)
            while True:
                arrived = self._fresh_start(robot).pose.position
                if math.dist((arrived.x, arrived.y), self._detour_point) <= self.get_parameter('yield_arrival_tolerance').value:
                    break
                if not current() or time.monotonic() >= arrival_deadline:
                    raise RuntimeError(f'{robot}: tracked pose has not confirmed yield-point arrival')
                await self._sleep(.05)
            self._mission_view_reason = f'{robot} reached yield point; checking original routes again.'
        except Exception:
            if handle is not None and handle.accepted and not terminal:
                handle.cancel_goal_async()
                await self._await_planning(handle.get_result_async(), time.monotonic()+5.)
                terminal = True
            raise
        finally:
            # An unconfirmed cancellation keeps the handle registered. New
            # missions are refused rather than hiding a possibly moving robot.
            if terminal and self._active_nav_goals.get(robot) is handle:
                self._active_nav_goals.pop(robot)
            self._detour_robot = None

    def _request_detour(self, priority, robot, route):
        if self._route_refresh_pending:
            return
        self._route_refresh_pending = True
        self._detour_work_active = True
        self._route_revision += 1
        goals = deepcopy(self._nav_targets)
        for name, handle in list(self._active_nav_goals.items()):
            self._yield_retry_robots.add(name)
            handle.cancel_goal_async()
        self.executor.create_task(self._detour_and_resume(
            self._mission_generation, priority, robot, route, goals))

    async def _detour_and_resume(self, generation, priority, robot, route, goals):
        try:
            deadline = time.monotonic()+float(self.get_parameter('planning_timeout').value)
            while self._active_nav_goals:
                if (generation != self._mission_generation or not self._mission_active or
                        self._mission_goal.is_cancel_requested):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cancellation timed out before evacuation')
                await self._sleep(.05)
            if robot not in goals:
                raise RuntimeError('Original goal missing for evacuation')
            if (generation != self._mission_generation or not self._mission_active or
                    self._mission_goal.is_cancel_requested):
                return
            await self._evacuate(priority, robot, route, goals[robot])
            await self._refresh_routes(generation, goals)
        except Exception as error:
            if generation == self._mission_generation:
                self._safety_fault = f'Yield detour failed: {error}'
                self.get_logger().error(self._safety_fault)
        finally:
            self._detour_work_active = False
            if generation == self._mission_generation:
                self._route_refresh_pending = False
