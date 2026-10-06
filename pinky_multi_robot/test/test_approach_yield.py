"""No ROS nodes or movement: exercise collision timer with fake state."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from unittest.mock import patch
from geometry_msgs.msg import PoseWithCovarianceStamped
from pinky_multi_robot.mission_action_server import MissionActionServer
from pinky_multi_robot.path_conflicts import remaining_path, path_prefix


def server(x=-2., priority_y=-2.):
    params = dict(route_deviation_limit=.25, yield_lookahead=.8,
                  route_deviation_confirm_seconds=1.,
                  path_clearance=.35, conflict_distance=.55,
                  resume_distance=.8, minimum_yield_seconds=2.)
    poses = {}
    for robot, xy in [('pinky1', (0., priority_y)), ('pinky2', (x, 0.))]:
        p = PoseWithCovarianceStamped()
        p.pose.pose.position.x, p.pose.pose.position.y = xy
        poses[robot] = p
    s = NS(_mission_active=True, _safety_fault=None, _yielding_robot=None,
           _route_refresh_pending=False, _route_deviation_since={},
           _request_route_refresh=Mock(),
           _yield_retry_robots=set(), _mission_robots={'pinky1', 'pinky2'},
           _active_nav_goals={'pinky1': Mock(), 'pinky2': Mock()},
           _approach_paths={'pinky1': [(0., -3.), (0., 3.)],
                            'pinky2': [(-3., 0.), (3., 0.)]},
           _poses=poses, _completed_robots=set(),
           get_parameter=lambda name: NS(value=params[name]),
           get_logger=lambda: Mock(),
           get_clock=lambda: NS(now=lambda: NS(nanoseconds=10000000000)),
           _distance=lambda: (2., 'pinky1', 'pinky2'))
    return s


def test_far_from_crossing_both_continue():
    s = server()
    MissionActionServer._collision_check(s)
    assert s._yielding_robot is None
    s._active_nav_goals['pinky2'].cancel_goal_async.assert_not_called()


def test_approaching_crossing_yields_before_entry():
    s = server(x=-1.)
    MissionActionServer._collision_check(s)
    assert s._yielding_robot == 'pinky2'
    s._active_nav_goals['pinky2'].cancel_goal_async.assert_called_once()
    s._active_nav_goals['pinky1'].cancel_goal_async.assert_not_called()


def test_cleared_crossing_resumes_before_priority_finishes():
    s = server(x=-1., priority_y=1.5)
    s._yielding_robot = 'pinky2'
    s._yield_started_ns = 0
    MissionActionServer._collision_check(s)
    assert s._yielding_robot is None


def test_unsafe_waiting_position_stops_both():
    s = server(x=-.2)
    MissionActionServer._collision_check(s)
    assert s._safety_fault
    for goal in s._active_nav_goals.values():
        goal.cancel_goal_async.assert_called_once()


def test_pose_loss_while_one_robot_waits_stops_other():
    s = server()
    s._yielding_robot = 'pinky2'
    del s._active_nav_goals['pinky2']
    s._distance = lambda: (None, 'pinky1', 'pinky2')
    MissionActionServer._collision_check(s)
    assert s._safety_fault
    s._active_nav_goals['pinky1'].cancel_goal_async.assert_called_once()


def test_transient_route_deviation_does_not_abort_mission():
    s = server()
    s._poses['pinky2'].pose.pose.position.y = 1.
    MissionActionServer._collision_check(s)
    assert s._safety_fault is None
    s._request_route_refresh.assert_not_called()


def test_sustained_deviation_requests_replan_not_mission_failure():
    s = server()
    s._poses['pinky2'].pose.pose.position.y = 1.
    with patch('pinky_multi_robot.mission_action_server.time.monotonic', return_value=10.):
        MissionActionServer._collision_check(s)
    with patch('pinky_multi_robot.mission_action_server.time.monotonic', return_value=11.1):
        MissionActionServer._collision_check(s)
    s._request_route_refresh.assert_called_once_with('pinky2', 1.)
    assert s._safety_fault is None


def test_return_to_route_resets_deviation_confirmation():
    s = server()
    s._poses['pinky2'].pose.pose.position.y = 1.
    MissionActionServer._collision_check(s)
    s._poses['pinky2'].pose.pose.position.y = 0.
    MissionActionServer._collision_check(s)
    assert 'pinky2' not in s._route_deviation_since


def test_completed_priority_off_old_path_does_not_stop_other():
    s = server()
    s._poses['pinky1'].pose.pose.position.x = 4.
    s._completed_robots.add('pinky1')
    del s._active_nav_goals['pinky1']
    MissionActionServer._collision_check(s)
    assert s._safety_fault is None
    s._active_nav_goals['pinky2'].cancel_goal_async.assert_not_called()
    s._request_route_refresh.assert_not_called()


def test_parked_priority_on_other_route_is_still_an_obstacle():
    s = server(x=-1., priority_y=0.)
    s._completed_robots.add('pinky1')
    del s._active_nav_goals['pinky1']
    MissionActionServer._collision_check(s)
    assert s._yielding_robot == 'pinky2'
    s._active_nav_goals['pinky2'].cancel_goal_async.assert_called_once()


def test_completed_lower_priority_off_route_does_not_stop_priority():
    s = server()
    s._poses['pinky2'].pose.pose.position.y = 4.
    s._completed_robots.add('pinky2')
    del s._active_nav_goals['pinky2']
    MissionActionServer._collision_check(s)
    assert s._safety_fault is None
    s._active_nav_goals['pinky1'].cancel_goal_async.assert_not_called()


def test_completed_lower_priority_on_route_makes_moving_priority_yield():
    s = server(x=0., priority_y=-1.)
    s._completed_robots.add('pinky2')
    del s._active_nav_goals['pinky2']
    MissionActionServer._collision_check(s)
    assert s._yielding_robot == 'pinky1'
    s._active_nav_goals['pinky1'].cancel_goal_async.assert_called_once()


def test_finished_robot_does_not_clear_a_still_blocked_route():
    s = server(x=-1., priority_y=0.)
    s._completed_robots.add('pinky1')
    s._yielding_robot = 'pinky2'
    s._yield_started_ns = 0
    MissionActionServer._collision_check(s)
    assert s._yielding_robot == 'pinky2'


def test_refresh_pending_skips_old_route_checks():
    s = server()
    s._route_refresh_pending = True
    s._poses['pinky2'].pose.pose.position.y = 1.
    MissionActionServer._collision_check(s)
    assert s._safety_fault is None
    s._request_route_refresh.assert_not_called()


def test_projection_and_lookahead():
    assert remaining_path((1., 0.), [(0., 0.), (3., 0.)]) == [(1., 0.), (3., 0.)]
    assert path_prefix([(1., 0.), (3., 0.)], .5) == [(1., 0.), (1.5, 0.)]
