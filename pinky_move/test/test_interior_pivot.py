"""Interior pivot selection is symmetric and uses measured entry evidence."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerConfig, CornerPolicy


def scene(sign):
    curve = np.vstack((np.column_stack((np.linspace(.04, .3, 70), np.full(70, -.107))),
                       np.column_stack((np.full(69, .3), np.linspace(-.107, .25, 70)[1:]))))
    curve[:, 1] *= sign
    obs = dict(curve=curve, side='right' if sign == 1 else 'left', width=.214)
    f = dict(entry=np.array([.201, -sign*.107]), exit=np.array([.3, sign*.12]),
             corner=np.array([.3, -sign*.107]), tin=np.array([1., 0.]),
             tout=np.array([0., float(sign)]))
    return obs, f


@pytest.mark.parametrize('sign', [1, -1])
def test_negative_eight_mm_window_can_use_real_entry_support(sign):
    obs, f = scene(sign)
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    p._start_staged(obs, f, np.zeros(3), 1.)
    assert p.debug['pivot_original_travel_m'] == pytest.approx(-.008)
    assert p.debug['pivot_rotation_margin_m'] >= .004
    assert abs(p.debug['pivot_shift_m']) <= .020001
    assert p.staged['pivot'][0] <= .193001  # Not deeper toward the paint.
    assert 0 < sign*p.staged['pivot'][1] <= .020001  # Small shift into the lane.
    pivot = p.staged['pivot'].copy()
    p.update(obs, None, 1.2, np.zeros(3), .22, .25)
    np.testing.assert_array_equal(p.staged['pivot'], pivot)


def test_cropped_entry_does_not_invent_support():
    obs, f = scene(1)
    # User-requested 10mm endpoint tolerance permits the old -8mm case.
    # A -12mm unsupported pivot must still be rejected.
    f['entry'][0] = .205
    obs['curve'] = obs['curve'][obs['curve'][:, 0] >= .205]
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    with pytest.raises(ValueError, match='outside observed entry support'):
        p._start_staged(obs, f, np.zeros(3), 1.)


@pytest.mark.parametrize('sign', [1, -1])
def test_eighty_five_degree_corner(sign):
    obs, f = scene(sign)
    f['tout'] = np.array([np.cos(np.deg2rad(85)), sign*np.sin(np.deg2rad(85))])
    f['exit'] = f['corner']+.23*f['tout']
    obs['curve'] = np.vstack((obs['curve'][:70],
                             f['corner']+np.linspace(.005, .36, 70)[:, None]*f['tout']))
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    p._start_staged(obs, f, np.zeros(3), 1.)
    assert p.debug['pivot_rotation_margin_m'] >= .004
    assert abs(p.debug['pivot_shift_m']) <= .020001
    assert abs(p.debug['pivot_lateral_shift_m']) <= .020001


def test_opposing_observed_boundary_can_reject_candidate():
    obs, f = scene(1)
    f['entry'][0] = .205  # Outside the explicit 10mm fallback tolerance.
    obs['other'] = obs['curve']+np.array([0., -.04])
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    with pytest.raises(ValueError, match='outside observed entry support'):
        p._start_staged(obs, f, np.zeros(3), 1.)
