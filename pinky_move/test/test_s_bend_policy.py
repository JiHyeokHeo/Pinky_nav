"""S curves stay spatial curves; right-angle staging has explicit boundaries."""
import numpy as np
import pytest
import pinky_move.lane_corner as lane
from test_lane_corner import observation


def s_observation(sign=1):
    x = np.linspace(.10, .85, 180)
    return dict(curve=np.column_stack((x, sign*(.04*np.sin(2*np.pi*(x-.1)/.75)-.12))),
                side='right' if sign > 0 else 'left', width=.18, width_source='measured')


@pytest.mark.parametrize('sign', [1, -1])
def test_s_path_created_without_ordinary_target(sign):
    p = lane.CornerPolicy(lane.CornerConfig(staged_turn=True))
    target = p.update(s_observation(sign), None, 1., np.zeros(3), .22, .25)
    assert p.state == 'S_BEND'
    assert target['s_bend_path'] and target['x_m'] > .05
    assert p.staged is None and not target.get('corner_stationary')
    assert lane.first_self_intersection(np.array(target['center_path'])) is None


def test_s_evidence_is_excursion_not_sum_of_jitter():
    s = np.linspace(0, .8, 100)
    assert not lane.sustained_s_bend(np.deg2rad(8*np.sin(np.arange(100))), s, .05)
    theta = np.deg2rad(np.r_[np.linspace(0,100,50), np.linspace(100,-10,50)])
    assert lane.sustained_s_bend(theta, s, .05)
    assert not lane.sustained_s_bend(np.linspace(0, np.pi/2, 100), s, .05)


def test_s_history_expires_and_never_accepts_unrelated_boundary():
    p = lane.CornerPolicy()
    obs = s_observation()
    feature = lane.classify_boundary(obs['curve'], p.config)
    assert feature['kind'] == 'S_BEND'
    p._s_bend_feature(obs, feature, 1., np.zeros(3))
    partial = dict(obs, curve=obs['curve'][20:95])
    normal = dict(kind='NORMAL', angle_deg=0., reason='cropped')
    assert p._s_bend_feature(partial, normal, 1.5, np.zeros(3))['kind'] == 'S_BEND'
    assert p.s_bend_anchor['time'] == 1.
    assert p._s_bend_feature(partial, normal, 3.1, np.zeros(3))['kind'] == 'NORMAL'
    p._s_bend_feature(obs, feature, 4., np.zeros(3))
    shifted = dict(partial, curve=partial['curve']+[0., .2])
    assert p._s_bend_feature(shifted, normal, 4.2, np.zeros(3))['kind'] == 'NORMAL'


def test_s_history_uses_current_odometry_and_rejects_wrong_side():
    p = lane.CornerPolicy()
    obs = s_observation()
    feature = lane.classify_boundary(obs['curve'], p.config)
    p._s_bend_feature(obs, feature, 1., np.zeros(3))
    pose = np.array([.02, .01, .15])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    partial = dict(obs, curve=(obs['curve'][20:95]-pose[:2])@np.array([[c,-s],[s,c]]))
    normal = dict(kind='NORMAL', angle_deg=0., reason='cropped')
    assert p._s_bend_feature(partial, normal, 1.2, pose)['kind'] == 'S_BEND'
    assert p._s_bend_feature(dict(partial, side='left'), normal, 1.4, pose)['kind'] == 'NORMAL'


def test_s_keeps_valid_ordinary_target_and_rejects_invalid_width():
    p = lane.CornerPolicy()
    ordinary = dict(x_m=.21, y_m=.02, boundary_count=1)
    # Centre generation preserves a validated ordinary target. The subsequent
    # odom route tracker requires its full path, and selects its own arc target.
    target = p._s_bend_target(s_observation(), ordinary, .22)
    assert target['x_m'] == .21 and target['y_m'] == .02
    with pytest.raises(ValueError, match='invalid S bend'):
        p._s_bend_target(dict(s_observation(), width=float('nan')), None, .22)


@pytest.mark.parametrize('angle', [60, 74.9, -74.9, 115.1])
def test_corner_history_cannot_bypass_staged_angle_gate(angle):
    p = lane.CornerPolicy(lane.CornerConfig(staged_turn=True))
    obs = observation()
    feature = dict(lane.classify_boundary(obs['curve'], p.config), angle_deg=angle)
    for now in (1., 1.2, 1.4, 1.6):
        _, votes = p._history_feature(obs, feature, now, np.zeros(3))
        assert votes == 0


@pytest.mark.parametrize('angle,expected', [(60,False),(74.9,False),(75,True),(90,True),
                                           (115,True),(115.1,False),(-75,True),(-74.9,False),(-115,True)])
def test_staged_angle_gate(monkeypatch, angle, expected):
    p = lane.CornerPolicy(lane.CornerConfig(staged_turn=True))
    obs = observation()
    feature = dict(lane.classify_boundary(obs['curve'], p.config), angle_deg=angle)
    monkeypatch.setattr(lane, 'classify_boundary', lambda *args: feature)
    calls = []
    def staged(*args):
        calls.append(True)
        raise ValueError('test reached staged planner')
    def curved(*args):
        raise ValueError('test reached curved planner')
    monkeypatch.setattr(p, '_start_staged', staged)
    monkeypatch.setattr(lane, 'plan_corner', curved)
    for now in (1., 1.2, 1.4):
        p.update(obs, None, now, np.zeros(3), .22, .25)
    assert bool(calls) == expected
