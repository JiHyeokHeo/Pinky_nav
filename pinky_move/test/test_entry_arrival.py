import numpy as np
import pytest
from test_staged_corner import prepared, local_observation
from pinky_move.lane_corner import staged_command


@pytest.mark.parametrize('passed', [False, True])
def test_entry_plane_arrival_with_lateral_error_brakes_then_turns(passed):
    p, obs, initial = prepared()
    pivot = np.array(initial['corner_pivot_world'])
    # Radial error >25 mm, but both entry-axis errors within tolerance.
    pose = np.r_[pivot+np.array([.02 if passed else -.02, .02]), 0.]
    assert staged_command(initial,pose,1.5,.25,.025)[:2] == (0.,0.)
    target = p.update(local_observation(obs,pose),None,1.6,pose,.22,.25)
    assert target['corner_stationary']
    assert staged_command(target,pose,2.,.25,.025)[1] > 0
    np.testing.assert_allclose(p.staged['pivot'],pivot)


@pytest.mark.parametrize('offset', [[.04,0.],[0.,.04],[0.,.08]])
def test_bad_crossing_latches_stop_without_moving_goal(offset):
    p, obs, initial = prepared()
    pivot = np.array(initial['corner_pivot_world'])
    pose = np.r_[pivot+offset,0.]
    assert staged_command(initial,pose,1.5,.25,.025)[:2] == (0.,0.)
    assert p.update(local_observation(obs,pose),None,1.6,pose,.22,.25) is None
    assert p.staged['fault']
    assert p.update(obs,None,1.8,np.zeros(3),.22,.25) is None
    np.testing.assert_allclose(p.staged['pivot'],pivot)


def test_no_progress_stops_at_control_rate_and_latches():
    p, obs, target = prepared()
    assert staged_command(target,np.zeros(3),4.5,.25,.025)[:2] == (0.,0.)
    assert p.update(obs,None,4.5,np.zeros(3),.22,.25) is None
    assert 'progress' in p.staged['fault']


def test_progress_updates_deadline_not_goal():
    p, obs, target = prepared()
    pose = np.array([.02,0.,0.])
    updated = p.update(local_observation(obs,pose),None,3.,pose,.22,.25)
    assert updated['corner_progress_deadline'] == 6.
    np.testing.assert_allclose(updated['corner_pivot_world'],target['corner_pivot_world'])
