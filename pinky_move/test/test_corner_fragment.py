"""Partial/reversed corner visibility tests without motor output."""
import numpy as np
import pytest
from pinky_move.lane_corner import match_corner_fragment
from test_lane_corner import observation
from test_staged_corner import prepared, local_observation


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('reverse', [False, True])
def test_exit_only_and_reversed_sampling_match(sign, reverse):
    curve = observation(sign)['curve']
    fragment = curve[95:145]
    if reverse: fragment = fragment[::-1]
    assert match_corner_fragment(fragment, curve)['fragment_ok']


def test_no_forward_clipping_when_robot_has_rotated():
    curve = observation()['curve']
    pose = np.array([.5, 0., np.pi/2])
    reference = local_observation(dict(curve=curve), pose)['curve']
    assert match_corner_fragment(reference[90:145], reference)['fragment_ok']


@pytest.mark.parametrize('case', ['parallel', 'endpoint', 'crossing', 'short', 'fold'])
def test_rejects_unrelated_or_unsupported_fragments(case):
    curve = observation()['curve']
    if case == 'parallel': fragment = curve[95:145]+[.06, 0.]
    if case == 'endpoint': fragment = np.column_stack((np.full(40,.5),np.linspace(.22,.4,40)))
    if case == 'crossing': fragment = np.column_stack((np.linspace(.3,.7,60),np.zeros(60)))
    if case == 'short': fragment = curve[100:103]
    if case == 'fold': fragment = np.vstack((curve[95:140],curve[95:140][::-1]))
    assert not match_corner_fragment(fragment, curve)['fragment_ok']


def test_staged_entry_accepts_fresh_exit_fragment_but_missing_observation_stops():
    policy, obs, _ = prepared()
    partial = dict(obs, curve=obs['curve'][95:145][::-1])
    target = policy.update(partial, None, 1.6, np.zeros(3), .22, .25)
    assert target['corner_staged'] and not target.get('corner_stationary')
    assert policy.debug['fragment_reversed']
    assert policy.update(None, None, 1.8, np.zeros(3), .22, .25) is None


def test_history_entry_keeps_original_measured_exit_for_later_visibility():
    from pinky_move.lane_corner import CornerPolicy, CornerConfig
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation()
    policy.update(obs, None, 1., np.zeros(3), .22, .25)
    entry = dict(obs, curve=obs['curve'][:50])
    policy.update(entry, None, 1.2, np.zeros(3), .22, .25)
    assert policy.update(entry, None, 1.4, np.zeros(3), .22, .25)['corner_staged']
    exit_only = dict(obs, curve=obs['curve'][95:145])
    assert policy.update(exit_only, None, 1.6, np.zeros(3), .22, .25)['corner_staged']


@pytest.mark.parametrize('length', [.031, .0393166])
def test_short_overlap_allowed_only_with_strong_correspondence(length):
    reference = np.column_stack((np.full(100,.5),np.linspace(-.2,.2,100)))
    fragment = np.column_stack((np.full(40,.5052),np.linspace(0.,length,40)))
    result = match_corner_fragment(fragment,reference)
    assert result['fragment_ok'] and result['fragment_strong']
    assert result['fragment_min_support_m'] == .03
    # Still within the general 35 mm distance limit, but not strong enough
    # to receive the shorter 30 mm support allowance.
    fragment[:,0] = .52
    result = match_corner_fragment(fragment,reference)
    assert not result['fragment_ok']
    assert result['fragment_min_support_m'] == .04
