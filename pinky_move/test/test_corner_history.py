"""Corner votes use recent measured geometry, never motor output."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerPolicy, CornerConfig, classify_boundary
from test_lane_corner import observation, ordinary
from test_staged_corner import local_observation


def seeded(sign=1):
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation(sign)
    feature = classify_boundary(obs['curve'], p.config)
    normal = dict(kind='NORMAL', angle_deg=0., reason='partial')
    for now, f in ((1.,feature),(1.2,normal),(1.4,feature),(1.6,normal),(1.8,feature)):
        result, votes = p._history_feature(obs, f, now, np.zeros(3))
    assert votes == 3
    return p, obs, feature, normal


@pytest.mark.parametrize('sign', [1, -1])
def test_nonconsecutive_votes_recover_in_current_robot_frame(sign):
    p, obs, feature, normal = seeded(sign)
    pose = np.array([.04, .01, .1])
    result, votes = p._history_feature(local_observation(obs, pose), normal, 2., pose)
    assert votes == 3 and result['kind'] == 'CORNER'
    expected = local_observation(dict(obs, curve=np.array([feature['corner']])), pose)['curve'][0]
    np.testing.assert_allclose(result['corner'], expected)


def test_history_never_overrides_missing_pose_s_bend_or_mismatching_boundary():
    for mode in ('pose', 'empty', 's_bend', 'geometry', 'width'):
        p, obs, _, normal = seeded()
        pose = np.zeros(3)
        if mode == 'pose': pose = None
        if mode == 'empty': obs = None
        if mode == 's_bend': normal = dict(normal, kind='S_BEND')
        if mode == 'geometry': obs = dict(obs, curve=obs['curve']+[0., .2])
        if mode == 'width': obs = dict(obs, width=.8)
        _, votes = p._history_feature(obs, normal, 2., pose)
        assert votes == 0


def test_history_expires_and_reset_clears_votes():
    p, obs, _, normal = seeded()
    assert p._history_feature(obs, normal, 4., np.zeros(3))[1] == 0
    p.reset()
    assert not p.corner_history


def test_straight_extension_is_not_a_partial_view_of_old_corner():
    p, obs, _, normal = seeded()
    straight = dict(obs, curve=np.column_stack((np.linspace(.1,.7,100),np.full(100,-.2))))
    assert p._history_feature(straight, normal, 2., np.zeros(3))[1] == 0


def test_full_policy_accumulates_interleaved_corner_frames(monkeypatch):
    import pinky_move.lane_corner as module
    obs = observation()
    real = classify_boundary(obs['curve'], CornerConfig())
    normal = dict(kind='NORMAL', angle_deg=0., reason='partial')
    features = iter([real, normal, real, normal, real])
    monkeypatch.setattr(module, 'classify_boundary', lambda *a: next(features))
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    for now in (1., 1.2, 1.4, 1.6, 1.8):
        target = p.update(obs, ordinary(), now, np.zeros(3), .22, .25)
    assert target['corner_staged']
    assert p.staged is not None


def test_low_ratio_and_opposite_direction_do_not_confirm():
    p, obs, feature, normal = seeded()
    for now in (1.9, 2., 2.1):
        _, votes = p._history_feature(obs, normal, now, np.zeros(3))
    assert votes == 0
    opposite = dict(feature, angle_deg=-feature['angle_deg'])
    assert p._history_feature(obs, opposite, 2.2, np.zeros(3))[1] == 0


@pytest.mark.parametrize('sign', [1, -1])
def test_real_corner_then_partial_entry_preserves_direction(sign):
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation(sign)
    feature = classify_boundary(obs['curve'], p.config)
    p._history_feature(obs, feature, 1., np.zeros(3))
    partial = dict(obs, curve=obs['curve'][:50])
    normal = classify_boundary(partial['curve'], p.config)
    result, count = p._history_feature(partial, normal, 1.2, np.zeros(3))
    assert result['kind'] == 'PARTIAL_CORNER' and count == 0
    pose = np.array([.02,0.,.05])
    result, count = p._history_feature(local_observation(partial,pose), normal, 1.4, pose)
    assert result['kind'] == 'CORNER' and count == 3
    assert sign*result['angle_deg'] > 80
    # Partial observations never extend the lifetime of the actual corner.
    result, count = p._history_feature(partial, normal, 3.01, np.zeros(3))
    assert count == 0 and result['kind'] == 'NORMAL'


def test_partial_straight_without_real_corner_never_creates_a_turn():
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation()
    obs = dict(obs, curve=obs['curve'][:50])
    feature = classify_boundary(obs['curve'], p.config)
    for now in (1.,1.2,1.4,1.6):
        result, count = p._history_feature(obs,feature,now,np.zeros(3))
        assert result['kind'] == 'NORMAL' and count == 0


def test_policy_enters_after_real_corner_and_two_fresh_partial_matches():
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation()
    p.update(obs, ordinary(), 1., np.zeros(3), .22, .25)
    partial = dict(obs, curve=obs['curve'][:50])
    p.update(partial, ordinary(), 1.2, np.zeros(3), .22, .25)
    target = p.update(partial, ordinary(), 1.4, np.zeros(3), .22, .25)
    assert target['corner_staged']
    # Even after history-assisted entry, absence of current boundary stops.
    assert p.update(None, None, 1.6, np.zeros(3), .22, .25) is None
