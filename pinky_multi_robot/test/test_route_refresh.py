"""No DDS/motors: test bounded route refresh and goal-result races."""
import asyncio
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from pinky_multi_robot.mission_action_server import MissionActionServer


def server():
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = 'map'
    goal = PoseStamped()
    goal.header.frame_id = 'map'
    params = {'planning_timeout': .5}
    return NS(_mission_generation=1, _mission_active=True,
              _mission_goal=NS(is_cancel_requested=False),
              _active_nav_goals={}, _completed_robots=set(),
              _poses={'pinky1': pose, 'pinky2': pose},
              _pose_receipt={'pinky1': time.monotonic(), 'pinky2': time.monotonic()},
              _approach_paths={'pinky1': [(0, 0), (1, 0)], 'pinky2': [(0, 2), (1, 2)]},
              _route_refresh_pending=True, _route_deviation_since={'pinky1': 0},
              _route_revision=0,
              _safety_fault=None, _yielding_robot='pinky2', _yield_started_ns=1,
              _nav_targets={'pinky1': goal, 'pinky2': goal}, _yield_retry_robots=set(),
              _compute_path=AsyncMock(return_value=[(0, 0), (2, 0)]),
              _collision_check=Mock(), _sleep=AsyncMock(),
              get_parameter=lambda name: NS(value=params[name]), get_logger=lambda: Mock())


def test_refresh_updates_both_routes_then_checks_collision_before_resume():
    s = server()
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert s._compute_path.await_count == 2
    assert s._approach_paths['pinky1'] == [(0, 0), (2, 0)]
    assert s._approach_paths['pinky2'] == [(0, 0), (2, 0)]
    s._collision_check.assert_called_once()
    assert not s._route_refresh_pending
    assert s._safety_fault is None


def test_refresh_never_replans_completed_robot():
    s = server()
    s._completed_robots.add('pinky1')
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert s._compute_path.await_count == 1
    assert s._compute_path.await_args.args[0] == 'pinky2'


def test_planner_failure_leaves_mission_faulted_not_resumed():
    s = server()
    s._compute_path.side_effect = RuntimeError('planner unavailable')
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert 'planner unavailable' in s._safety_fault
    s._collision_check.assert_not_called()
    assert s._approach_paths['pinky1'] == [(0, 0), (1, 0)]


def test_stale_pose_cannot_start_route_refresh():
    s = server()
    s._pose_receipt['pinky1'] -= 3.
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert 'fresh tracked pose required' in s._safety_fault
    s._compute_path.assert_not_called()


def test_old_refresh_cannot_modify_new_mission():
    s = server()
    s._mission_generation = 2
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert s._route_refresh_pending  # old finally cannot release new pause
    s._compute_path.assert_not_called()


def test_cancelled_mission_cannot_resume():
    s = server()
    s._mission_goal.is_cancel_requested = True
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    s._compute_path.assert_not_called()
    s._collision_check.assert_not_called()


def test_route_publication_is_atomic_on_second_planner_failure():
    s = server()
    s._compute_path.side_effect = [[(0, 0), (3, 0)], RuntimeError('second planner failed')]
    asyncio.run(MissionActionServer._refresh_routes(s, 1, s._nav_targets))
    assert s._approach_paths['pinky1'] == [(0, 0), (1, 0)]
    assert s._safety_fault


def test_request_marks_cancellations_retryable_without_fault():
    s = server()
    s._route_refresh_pending = False
    s._active_nav_goals = {'pinky1': Mock(), 'pinky2': Mock()}
    def schedule(coro):
        coro.close()  # unit fixture, not a running executor
        return 'scheduled'
    s.executor = NS(create_task=schedule)
    s._refresh_routes = lambda *args: MissionActionServer._refresh_routes(s, *args)
    MissionActionServer._request_route_refresh(s, 'pinky1', .4)
    assert s._yield_retry_robots == {'pinky1', 'pinky2'}
    assert s._route_refresh_pending and s._safety_fault is None
    for handle in s._active_nav_goals.values():
        handle.cancel_goal_async.assert_called_once()


def navigate_fixture(status):
    s = server()
    s._route_refresh_pending = False
    s._reserved_robot = None
    s._yielding_robot = None
    s._yield_retry_robots = {'pinky1'}
    handle = NS(accepted=True, get_result_async=AsyncMock(
        return_value=NS(status=status, result=NS(error_msg='failure'))), cancel_goal_async=Mock())
    client = NS(wait_for_server=lambda **kwargs: True, send_goal_async=AsyncMock(return_value=handle))
    s._client = lambda robot: client
    return s, client


def test_success_wins_cancel_race_and_is_not_reissued():
    s, client = navigate_fixture(GoalStatus.STATUS_SUCCEEDED)
    result = asyncio.run(MissionActionServer._navigate(s, 'pinky1', PoseStamped(), s._mission_goal))
    assert result is True
    assert client.send_goal_async.await_count == 1
    assert 'pinky1' not in s._yield_retry_robots


def test_real_abort_is_not_mistaken_for_yield_retry():
    s, client = navigate_fixture(GoalStatus.STATUS_ABORTED)
    result = asyncio.run(MissionActionServer._navigate(s, 'pinky1', PoseStamped(), s._mission_goal))
    assert result is False
    assert client.send_goal_async.await_count == 1


def test_late_goal_acceptance_after_refresh_is_cancelled_then_retried():
    s, client = navigate_fixture(GoalStatus.STATUS_CANCELED)
    first = client.send_goal_async.return_value
    second = NS(accepted=True, get_result_async=AsyncMock(return_value=NS(
        status=GoalStatus.STATUS_SUCCEEDED)), cancel_goal_async=Mock())
    count = 0
    async def send(_request):
        nonlocal count
        count += 1
        if count == 1:
            s._route_revision += 1
            return first
        return second
    client.send_goal_async.side_effect = send
    assert asyncio.run(MissionActionServer._navigate(s, 'pinky1', PoseStamped(), s._mission_goal))
    first.cancel_goal_async.assert_called_once()
    assert client.send_goal_async.await_count == 2


def test_yield_starting_during_goal_acceptance_cancels_late_handle():
    s, client = navigate_fixture(GoalStatus.STATUS_CANCELED)
    first = client.send_goal_async.return_value
    second = NS(accepted=True, get_result_async=AsyncMock(return_value=NS(
        status=GoalStatus.STATUS_SUCCEEDED)), cancel_goal_async=Mock())
    count = 0
    async def send(_request):
        nonlocal count
        count += 1
        if count == 1:
            s._yielding_robot = 'pinky1'
            return first
        return second
    async def release(_seconds):
        s._yielding_robot = None
    s._sleep = release
    client.send_goal_async.side_effect = send
    assert asyncio.run(MissionActionServer._navigate(s, 'pinky1', PoseStamped(), s._mission_goal))
    first.cancel_goal_async.assert_called_once()
    assert client.send_goal_async.await_count == 2
