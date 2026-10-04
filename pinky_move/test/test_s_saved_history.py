"""Saved route continuation has fixed budgets and cannot renew observations."""
import numpy as np
import pytest
from test_s_route_progress import seeded, path, target, observation


def saved_case():
    p,first=seeded()
    p.s_route['boundary_world']=path()+[0.,-.11]
    return p,first


def test_history_corrects_pose_without_rewriting_saved_route():
    p,first=saved_case()
    old=p.s_route['path'].copy()
    pose=np.array([0.,0.,.1])
    result=p._saved_s_target(None,pose,1.2,.22,'empty')
    from pinky_move.lane_corner import world_point
    np.testing.assert_allclose(world_point([result['x_m'],result['y_m']],pose),
                               [first['x_m'],first['y_m']],atol=1e-8)
    assert result['s_route_history'] and result['corner_speed_cap'] <= .02
    assert p.s_route['time']==1. and p.s_route['covered_m']==0.
    np.testing.assert_array_equal(p.s_route['path'],old)


def test_repeated_missing_frames_taper_then_stop_without_renewing_budget():
    p,_=saved_case()
    for now in (1.2,1.4,1.8,2.,2.5,2.9):
        result=p.update(None,None,now,np.zeros(3),.22,.25,no_boundaries=True)
        assert result is not None
        assert result['corner_speed_cap']==pytest.approx(.02*min(1.,3.-now))
        assert p.s_route['time']==1.
    assert p.update(None,None,3.,np.zeros(3),.22,.25,no_boundaries=True) is None
    assert p.block_recovery


@pytest.mark.parametrize('pose,now', [(None,1.2),(np.zeros(3),1.),
    (np.zeros(3),3.),(np.array([.041,0.,0.]),1.2),
    (np.array([0.,0.,np.deg2rad(21)]),1.2)])
def test_budget_and_pose_guards(pose,now):
    p,_=saved_case()
    assert p._saved_s_target(None,pose,now,.22,'empty') is None


@pytest.mark.parametrize('case',['matching','shift','reverse','side','width'])
def test_partial_observation_must_not_contradict_history(case):
    p,_=saved_case()
    obs=dict(observation(),curve=p.s_route['boundary_world'][30:].copy())
    if case=='shift': obs['curve'] += [0.,.1]
    if case=='reverse': obs['curve']=obs['curve'][::-1]
    if case=='side': obs['side']='left'
    if case=='width': obs['width']=.4
    result=p._saved_s_target(obs,np.zeros(3),1.2,.22,'crop')
    assert (result is not None)==(case=='matching')


def test_ambiguous_nonempty_detection_cannot_use_empty_history_branch():
    p,_=saved_case()
    assert p.update(None,None,1.2,np.zeros(3),.22,.25,no_boundaries=False) is None


def test_crop_failure_uses_bounded_history_in_policy(monkeypatch):
    import pinky_move.lane_corner as lane
    p,_=saved_case()
    obs=dict(observation(),curve=p.s_route['boundary_world'][30:].copy())
    monkeypatch.setattr(lane,'classify_boundary',lambda *a:dict(kind='S_BEND',angle_deg=50.,reason='test'))
    monkeypatch.setattr(p,'_s_bend_feature',lambda obs,feature,*a:feature)
    monkeypatch.setattr(p,'_history_feature',lambda obs,feature,*a:(feature,0))
    ordinary=target(path()[30:],np.zeros(3))
    result=p.update(obs,ordinary,1.2,np.zeros(3),.22,.25)
    assert result['s_route_history']
    assert p.s_route['time']==1.
