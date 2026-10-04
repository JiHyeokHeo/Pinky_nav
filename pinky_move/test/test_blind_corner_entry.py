"""Near-pivot camera gaps may use bounded measured odometry, not blind search."""
import numpy as np
import pytest
from pinky_move.lane_corner import staged_command
from test_staged_corner import prepared, local_observation
from test_lane_controller import controller
from types import SimpleNamespace


def near():
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]-np.array([.03, 0., 0.])
    p.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    return p, pose


def test_five_mm_bridge_then_brake_and_turn():
    p, pose = near()
    target = p.update(None, None, 1.7, pose, .22, .25, no_boundaries=True)
    v, w, _ = staged_command(target, pose, 1.7, .25, .025)
    assert 0 < v <= .015 and abs(w) <= .15
    arrived = pose+np.array([.006, 0., 0.])
    p.observe_arrival(arrived, 1.8)
    assert p.staged['brake_at'] == 1.8
    assert staged_command(target, arrived, 1.8, .25, .025)[:2] == (0., 0.)
    target = p.update(None, None, 1.9, arrived, .22, .25, no_boundaries=True)
    assert target['corner_stationary']
    assert staged_command(target, arrived, 2.2, .25, .025)[0] == 0.
    assert staged_command(target, arrived, 2.2, .25, .025)[1] > 0.


def test_empty_perception_keeps_bounded_target_in_node(controller):
    p, pose = near()
    controller.corner_policy = p
    controller.parameters['corner_enabled'] = True
    controller.robot_calibration = {'test': True}
    controller.safety_clock.seconds = 1.7
    controller._corner_pose_for_frame = lambda now: pose
    def missing(*args, **kwargs):
        raise ValueError('no boundaries detected')
    controller.metric_tracker = SimpleNamespace(update=missing, last_observation=None)
    assert controller._update_lane_command([], 640, controller.safety_clock.now()) == 0
    assert controller.metric_target['corner_blind_entry'] is p.staged['blind_entry']
    assert controller.metric_target['corner_speed_cap'] <= .015


def test_deadline_does_not_renew_on_empty_frames():
    p, pose = near()
    for now in (1.7, 2., 2.5, 3.):
        target = p.update(None, None, now, pose, .22, .25, no_boundaries=True)
        assert target['corner_blind_deadline'] == 3.6
    assert staged_command(target, pose, 3.6, .25, .025)[:2] == (0., 0.)


def test_distance_budget_counts_back_and_forth():
    p, pose = near()
    target = p.update(None, None, 1.7, pose, .22, .25, no_boundaries=True)
    staged_command(target, pose-np.array([.02, 0., 0.]), 1.8, .25, .025)
    assert staged_command(target, pose, 1.9, .25, .025)[:2] == (0., 0.)
    p.observe_arrival(pose+np.array([.006, 0., 0.]), 2.)
    assert p.staged['brake_at'] is None


@pytest.mark.parametrize('case', ['far', 'stale', 'lateral', 'heading', 'invalid'])
def test_no_unqualified_blind_entry(case):
    p, pose = near()
    now = 1.7
    if case == 'far': pose = pose-np.array([.1, 0., 0.])
    if case == 'stale': now = 2.5
    if case == 'lateral': pose = pose+np.array([0., .04, 0.])
    if case == 'heading': pose = pose+np.array([0., 0., .5])
    assert p.update(None, None, now, pose, .22, .25, no_boundaries=case != 'invalid') is None
