"""A new/ambiguous boundary cannot interrupt a committed stationary turn."""
import numpy as np
import pytest
from types import SimpleNamespace
from pinky_move.lane_corner import staged_command
from test_staged_corner import prepared
from test_lane_corner import ordinary
from test_lane_controller import controller


@pytest.mark.parametrize('sign', [1, -1])
def test_empty_single_wrong_side_and_mismatch_do_not_cancel_turn(sign):
    p, obs, target = prepared(sign)
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.5)
    pivot = p.staged['pivot'].copy()
    for now, measured in ((2., None), (2.2, dict(obs, side='left' if sign == 1 else 'right')),
                          (2.4, dict(obs, curve=obs['curve']+.2)), (2.6, None)):
        target = p.update(measured, None, now, pose, .22, .25, no_boundaries=measured is None)
        v, w, _ = staged_command(target, pose, now, .25, .025)
        assert v == 0 and 0 < sign*w <= .15
        np.testing.assert_array_equal(p.staged['pivot'], pivot)
        assert p.debug['mode'] == 'PIVOT_TURN'
        assert target['boundary_count'] == 0 and target['visible_side'] is None
    # After reaching the yaw, zero command while waiting for a real exit path.
    pose[2] = p.staged['exit_yaw']
    target = p.update(None, None, 3., pose, .22, .25)
    assert p.debug['mode'] == 'EXIT_REACQUIRE'
    assert staged_command(target, pose, 3., .25, .025)[:2] == (0., 0.)
    target = p.update(None, None, 30., pose, .22, .25)
    assert p.debug['mode'] == 'EXIT_REACQUIRE'
    assert staged_command(target, pose, 30., .25, .025)[:2] == (0., 0.)
    for now in (30.2, 30.4, 30.6):
        target = p.update(obs, ordinary(), now, pose, .22, .25)
    assert p.staged is None and not target.get('corner_staged')


def test_wrong_boundary_before_arrival_still_stops():
    p, obs, target = prepared()
    assert p.update(dict(obs, side='left'), None, 1.6, np.zeros(3), .22, .25) is None


def test_exact_reacquiring_left_error_preserves_turn(controller):
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.5)
    controller.corner_policy = p
    controller.robot_calibration = {'test': True}
    controller.parameters['corner_enabled'] = True
    controller.safety_clock.seconds = 2.
    controller._corner_pose_for_frame = lambda _: pose
    def reject(*args, **kwargs):
        raise ValueError('reacquiring left boundary: 1/3')
    controller.metric_tracker = SimpleNamespace(update=reject, last_observation=None)
    assert controller._update_lane_command([np.zeros((20,20))], 20, controller.safety_clock.now()) == 0
    assert controller.metric_target['corner_stationary']
    assert staged_command(controller.metric_target, pose, 2., .25, .025)[1] > 0


def test_exit_confirmation_gap_does_not_accumulate():
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.5)
    pose[2] = p.staged['exit_yaw']
    for now in (2., 2.2, 4.):
        target = p.update(obs, ordinary(), now, pose, .22, .25)
    assert p.staged is not None and p.staged['exit_count'] == 1


def test_exit_tracking_reset_is_one_shot_and_retains_measured_width(controller):
    p, obs, target = prepared()
    p.staged.update(brake_at=1.5, heading_reached=True)
    controller.corner_policy = p
    controller.robot_calibration = {'test': True}
    controller.parameters['corner_enabled'] = True
    pose = np.r_[target['corner_pivot_world'], p.staged['exit_yaw']]
    controller._corner_pose_for_frame = lambda _: pose
    controller.safety_clock.seconds = 2.
    width = controller.metric_tracker.width_estimator
    controller._update_lane_command([], 640, controller.safety_clock.now())
    tracker = controller.metric_tracker
    assert tracker.width_estimator is width and p.staged['exit_tracking_reset']
    controller.safety_clock.seconds = 2.2
    controller._update_lane_command([], 640, controller.safety_clock.now())
    assert controller.metric_tracker is tracker
