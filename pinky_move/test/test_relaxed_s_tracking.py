"""Fresh S observations can resume without stationary/5cm lockout; no motors."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerPolicy, CornerConfig
from test_s_route_progress import path, target, observation
from test_local_lane_map import scene


def policy():
    p = CornerPolicy(CornerConfig(staged_turn=True, relaxed_tracking=True))
    p._track_s_route(target(path(), np.zeros(3)), observation(), np.zeros(3), 1., .22)
    return p


def test_expired_route_resumes_after_more_than_five_cm_without_three_frame_wait():
    p = policy()
    pose = np.array([.0525, 0., np.deg2rad(18.3)])
    result = p._track_s_route(target(path(), pose), observation(), pose, 80., .22)
    assert result['s_route_tracking']
    assert p.s_route['time'] == 80.


def test_three_cm_observation_error_no_longer_blocks_connected_s():
    p = policy()
    result = p._track_s_route(target(path()+[0., .03], np.zeros(3)),
                             observation(), np.zeros(3), 1.2, .22)
    assert result['s_route_tracking']
    assert p.debug['s_match_distance_limit_m'] == .04


@pytest.mark.parametrize('reason', [
    'S bend: S route expired away from verified pose; reset required',
    'S bend: S route new path skips or contradicts pending bend',
    'S bend: S route hidden bend too sharp',
])
def test_current_map_replaces_rejected_stored_route(monkeypatch, reason):
    p = policy()
    obs, ordinary = scene('right')
    def blocked(*args):
        p.debug = {'reason': reason}
        return None
    monkeypatch.setattr(p, '_update', blocked)
    assert p.update(obs, ordinary, 1.1, np.zeros(3), .22, .5) is None
    result = p.update(obs, ordinary, 1.3, np.zeros(3), .22, .5)
    assert result['local_map_assisted']
    assert p.s_route is None
    assert p.debug['s_route_replaced_by_current_map']


def test_no_observation_does_not_authorize_map_motion(monkeypatch):
    p = policy()
    monkeypatch.setattr(p, '_update', lambda *a: None)
    p.debug = {'reason': 'S bend: S route expired away from verified pose; reset required'}
    assert p.update(None, None, 5., np.zeros(3), .22, .5) is None


@pytest.mark.parametrize('reason', ['insufficient observed legs',
                                  'sharp change without reliable entry/exit'])
def test_unknown_classification_keeps_current_valid_centre(monkeypatch, reason):
    p = CornerPolicy(CornerConfig(relaxed_tracking=True))
    obs, ordinary = scene('right')
    monkeypatch.setattr('pinky_move.lane_corner.classify_boundary',
                        lambda *a: dict(kind='UNKNOWN', angle_deg=0., reason=reason))
    result = p.update(obs, ordinary, 1., np.zeros(3), .22, .5)
    assert result['unknown_following']
    assert not p.block_recovery
    assert p.debug['classification_reason'] == reason
    assert result['center_path'] == ordinary['center_path']


@pytest.mark.parametrize('change', ['held', 'empty_path', 'nan', 'backward'])
def test_unknown_does_not_replay_invalid_target(monkeypatch, change):
    p = CornerPolicy(CornerConfig(relaxed_tracking=True))
    obs, ordinary = scene('right')
    if change == 'held': ordinary['held'] = True
    if change == 'empty_path': ordinary['center_path'] = []
    if change == 'nan': ordinary['x_m'] = float('nan')
    if change == 'backward': ordinary['x_m'] = -.1
    monkeypatch.setattr('pinky_move.lane_corner.classify_boundary',
                        lambda *a: dict(kind='UNKNOWN', angle_deg=0., reason='insufficient observed legs'))
    assert p.update(obs, ordinary, 1., np.zeros(3), .22, .5) is None
