"""Arrival is remembered during inference failure, without authorizing motion."""
import numpy as np
import pytest
from test_staged_corner import prepared
from test_lane_controller import controller
from pinky_move.lane_autonomy import DriveState


def arrival_case():
    policy, _, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]-[.0187, .0022, 0.]
    return policy, pose


def test_log_case_arrival_after_point_eight_seconds_then_empty_frame():
    p, pose = arrival_case()
    pivot = p.staged['pivot'].copy()
    p.observe_arrival(pose, 2.6)  # last real match 1.4, no blind budget armed
    assert p.staged['brake_at'] == 2.6
    target = p.update(None, None, 2.7, pose, .22, .25, no_boundaries=True)
    assert target['corner_stationary']
    p.observe_arrival(pose, 2.8)
    assert p.staged['brake_at'] == 2.6  # no timer renewal
    np.testing.assert_array_equal(p.staged['pivot'], pivot)


@pytest.mark.parametrize('case', ['old', 'far', 'lateral', 'heading', 'nan', 'mismatch', 'fault'])
def test_unqualified_arrival_is_not_latched(case):
    p, pose = arrival_case()
    now = 2.6
    if case == 'old': now = 3.5
    if case == 'far': pose[0] -= .1
    if case == 'lateral': pose[1] += .05
    if case == 'heading': pose[2] = .4
    if case == 'nan': pose[0] = np.nan
    if case == 'mismatch': p.staged['blind_eligible'] = False
    if case == 'fault': p.staged['fault'] = 'odom discontinuity'
    p.observe_arrival(pose, now)
    assert p.staged['brake_at'] is None


@pytest.mark.parametrize('fresh', [True, False])
def test_control_records_arrival_but_inference_failure_still_publishes_zero(controller, fresh):
    p, pose = arrival_case()
    controller.corner_policy = p
    controller.safety_clock.seconds = 2.6
    controller.corner_odom = (2.6 if fresh else 2., pose)
    controller.metric_target = None
    def consume(now):
        # The local arrival must exist before an older remote result is handled.
        assert (p.staged['brake_at'] is not None) == fresh
    controller._consume_inference = consume
    controller._start_inference = lambda now: None
    controller._safety_stop_reason = lambda now: 'inference failed'
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].linear.x == 0.
    assert controller.commands[-1].angular.z == 0.
    assert (p.staged['brake_at'] is not None) == fresh
