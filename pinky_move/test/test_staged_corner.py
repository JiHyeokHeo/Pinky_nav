"""Motor-free ENTRY -> BRAKE -> PIVOT_TURN tests in odometry coordinates."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerConfig, CornerPolicy, staged_command
from test_lane_corner import observation, ordinary
from test_lane_controller import controller
from pinky_move.lane_autonomy import DriveState


def local_observation(obs, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    inverse = np.array([[c, s], [-s, c]])
    return dict(obs, curve=(obs['curve']-pose[:2])@inverse.T)


def prepared(sign=1):
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation(sign)
    result = None
    for now in (1., 1.2, 1.4):
        result = policy.update(obs, ordinary(), now, np.zeros(3), .22, .25)
    assert result['corner_staged'] and not result.get('corner_stationary')
    return policy, obs, result


@pytest.mark.parametrize('sign', [1, -1])
def test_entry_then_stationary_turn_then_near_exit(sign):
    policy, obs, target = prepared(sign)
    pivot = np.array(target['corner_pivot_world'])
    assert pivot[0] == pytest.approx(.3, abs=.002)
    assert abs(pivot[1]) < .002
    v, w, _ = staged_command(target, np.zeros(3), 1.4, .25, .025)
    assert 0 < v <= .03 and abs(w) < .01
    pose = np.r_[pivot, 0.]
    # At 20 Hz, entry stops even before another inference arrives.
    assert staged_command(target, pose, 1.5, .25, .025)[:2] == (0., 0.)
    target = policy.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    assert target['corner_stationary']
    assert staged_command(target, pose, 1.7, .25, .025)[:2] == (0., 0.)
    v, w, _ = staged_command(target, pose, 2., .25, .025)
    assert v == 0 and sign*w > 0 and abs(w) <= .25
    pose[2] = sign*np.pi/2
    assert staged_command(target, pose, 2.1, .25, .025)[:2] == (0., 0.)
    # A distant path must NOT release the turn.
    far = dict(ordinary(), x_m=.5)
    target = policy.update(local_observation(obs, pose), far, 2.2, pose, .22, .25)
    assert target['corner_stationary']
    for now in (2.4, 2.6, 2.8):
        target = policy.update(local_observation(obs, pose), ordinary(), now, pose, .22, .25)
    assert not target.get('corner_staged') and policy.staged is None


def test_staged_loss_wrong_identity_and_timeout_do_not_drive():
    p, obs, target = prepared()
    assert p.update(None, None, 1.6, np.zeros(3), .22, .25) is None
    assert p.update(dict(obs, side='left'), None, 1.8, np.zeros(3), .22, .25) is None
    assert p.update(obs, None, 2., None, .22, .25) is None
    assert p.update(obs, None, 22., np.zeros(3), .22, .25) is None
    assert staged_command(target, np.zeros(3), 22., .25, .025)[:2] == (0., 0.)


def test_spin_deadline_and_drift_stop_at_control_rate():
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    target = p.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    assert staged_command(target, pose, target['corner_spin_deadline'], .25, .025)[:2] == (0., 0.)
    displaced = pose+np.array([.05, 0., 0.])
    assert staged_command(target, displaced, 2., .25, .025)[:2] == (0., 0.)


def test_short_observed_entry_allows_supported_pivot_but_not_extrapolation():
    from pinky_move.lane_corner import classify_boundary
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = dict(observation(), width=.154)
    f = classify_boundary(obs['curve'], p.config)
    f = dict(f, entry=f['corner']-.089*f['tin'])
    p._start_staged(obs, f, np.zeros(3), 1.)
    assert p.staged is not None
    # Half-width offset would put the pivot before the observed entry.
    f = dict(f, entry=f['corner']-.06*f['tin'])
    with pytest.raises(ValueError, match='outside observed'):
        p._start_staged(obs, f, np.zeros(3), 1.)


def test_controller_staged_spin_obeys_stale_frame_and_odom_stop(controller):
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    target = p.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    controller.metric_target = target
    controller.corner_odom = (10., pose)
    controller.last_result_input_time = controller.safety_clock.now()
    controller.last_lane_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.commands[-1].linear.x == 0.
    assert controller.commands[-1].angular.z > 0.
    controller.safety_clock.seconds += .4
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].angular.z == 0.
    controller.corner_odom = (10.4, pose)
    controller.parameters['result_timeout'] = .2
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].linear.x == controller.commands[-1].angular.z == 0.


@pytest.mark.parametrize('values', [dict(entry_speed_mps=.1), dict(pivot_tolerance_m=.2),
                                    dict(brake_seconds=0.), dict(spin_timeout_s=21.)])
def test_invalid_staged_limits(values):
    with pytest.raises(ValueError):
        CornerConfig(**values).validate()
