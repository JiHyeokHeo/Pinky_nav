"""Bounded no-mask turns must not turn stored goals into blind entry motion."""
import numpy as np
import pytest
from pinky_move.lane_corner import staged_command, match_corner_fragment
from test_staged_corner import prepared


@pytest.mark.parametrize('sign', [1, -1])
def test_arrival_latched_before_empty_frame(sign):
    p, obs, target = prepared(sign)
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.5)
    assert p.staged['brake_at'] == 1.5
    target = p.update(None, None, 1.9, pose, .22, .25, no_boundaries=True)
    assert target['corner_stationary'] and target['boundary_count'] == 0
    v, w, _ = staged_command(target, pose, 1.9, .25, .025)
    assert v == 0 and 0 < sign*w <= .15
    assert sign*staged_command(target, pose, 3.4, .25, .025)[1] > 0.
    deadline = target['corner_spin_deadline']
    target = p.update(None, None, 3.5, pose, .22, .25, no_boundaries=True)
    assert target['corner_spin_deadline'] == deadline
    assert staged_command(target, pose, deadline, .25, .025)[:2] == (0., 0.)
    assert p.update(None, None, deadline, pose, .22, .25, no_boundaries=True) is None
    assert p.debug['mode'] == 'TURN_TIMEOUT'
    assert p.staged['matched_at'] == 1.4


def test_no_blind_entry_or_invalid_geometry_bypass():
    p, obs, target = prepared()
    assert p.update(None, None, 1.5, np.zeros(3), .22, .25, no_boundaries=True) is None
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.6)
    # Once the far blind-entry attempt invalidates evidence, teleporting the
    # odometry pose to the pivot must not re-authorize a turn.
    assert p.staged['brake_at'] is None
    assert p.update(None, None, 1.7, pose, .22, .25) is None
    # An independently valid arrival still tolerates changing exit detections.
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.6)
    assert p.update(None, None, 1.7, pose, .22, .25)['corner_stationary']
    assert p.update(dict(obs, side='left'), None, 1.8, pose, .22, .25)['corner_stationary']
    assert p.update(None, None, 1.9, None, .22, .25, no_boundaries=True) is None


def test_stale_match_cannot_latch_arrival():
    p, obs, target = prepared()
    # Arrival now has a two-second evidence window, not the .8s blind-entry
    # motion gate. At 3.5 the last real match at 1.4 is genuinely expired.
    p.observe_arrival(np.r_[target['corner_pivot_world'], 0.], 3.5)
    assert p.staged['brake_at'] is None


def test_independent_turn_stops_at_heading_and_drift_without_rearming():
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    p.observe_arrival(pose, 1.5)
    target = p.update(None, None, 1.9, pose, .22, .25, no_boundaries=True)
    deadline = target['corner_spin_deadline']
    for now in (3.5, 8., 15.):
        target = p.update(None, None, now, pose, .22, .25, no_boundaries=True)
        assert target['corner_spin_deadline'] == deadline
        assert 'corner_blind_deadline' not in target
        assert staged_command(target, pose, now, .25, .025)[0] == 0.
    aligned = pose.copy()
    aligned[2] = target['corner_spin_yaw']
    assert staged_command(target, aligned, 15., .25, .025)[:2] == (0., 0.)
    drifted = pose+np.array([.05, 0., 0.])
    assert staged_command(target, drifted, 15., .25, .025)[:2] == (0., 0.)


def test_partial_precise_contiguous_segment():
    # About half of the observed chain follows the reference; the rest turns
    # away. Accept its long exact segment, not an endpoint-only coincidence.
    reference = np.column_stack((np.linspace(0, .3, 120), np.zeros(120)))
    current = np.vstack((np.column_stack((np.linspace(.02, .14, 60), np.zeros(60))),
                         np.column_stack((np.full(59, .14), np.linspace(.002, .14, 59)))))
    result = match_corner_fragment(current, reference)
    assert .4 <= result['fragment_fraction'] < .7
    assert result['fragment_ok']
    assert not match_corner_fragment(current+np.array([0., .02]), reference)['fragment_ok']
