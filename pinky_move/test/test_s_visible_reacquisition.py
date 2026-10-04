"""Visible overlap must not discard the near blind prefix or skip a bend."""
import numpy as np
import pytest
from test_s_route_progress import seeded, path, target, observation


def test_cropped_near_prefix_keeps_goal_and_progress():
    p, first = seeded()
    before = p.s_route['path'].copy()
    fresh = path()[5:]  # ~3.4cm cropped; first visible x~13.4cm.
    result=p._track_s_route(target(fresh,np.zeros(3)),observation(),np.zeros(3),1.2,.22)
    np.testing.assert_allclose([result['x_m'],result['y_m']],
                               [first['x_m'],first['y_m']],atol=1e-9)
    np.testing.assert_allclose(p.s_route['path'][:len(before)],before,atol=1e-9)
    assert p.s_route['covered_m']==0.
    assert .02 < p.debug['s_route_near_crop_m'] < .06


@pytest.mark.parametrize('case',['far','shift','reverse','hidden_corner'])
def test_crop_cannot_skip_unseen_geometry(case):
    p,_=seeded()
    fresh=path()[5:]
    if case=='far':fresh=path()[20:]
    if case=='shift':fresh=fresh+[0.,.06]
    if case=='reverse':fresh=fresh[::-1]
    if case=='hidden_corner':
        p.s_route['path'][1:4,1]+=.02
    before=p.s_route['path'].copy()
    with pytest.raises(ValueError):
        p._track_s_route(target(fresh,np.zeros(3)),observation(),np.zeros(3),1.2,.22)
    np.testing.assert_array_equal(p.s_route['path'],before)


def test_stationary_timeout_recovers_after_three_real_matches():
    p,_=seeded()
    old=p.s_route['path'].copy()
    for now,count in [(4.,1),(4.2,2)]:
        with pytest.raises(ValueError,match=f'reacquisition {count}/3'):
            p._track_s_route(target(path(),np.zeros(3)),observation(),np.zeros(3),now,.22)
        assert p.s_route['time']==1.
    result=p._track_s_route(target(path(),np.zeros(3)),observation(),np.zeros(3),4.4,.22)
    assert result['s_route_tracking'] and p.s_route['time']==4.4
    np.testing.assert_allclose(p.s_route['path'][:len(old)],old,atol=1e-9)


def test_bad_frame_breaks_reacquisition_and_moved_robot_cannot_rearm():
    p,_=seeded()
    with pytest.raises(ValueError,match='1/3'):
        p._track_s_route(target(path(),np.zeros(3)),observation(),np.zeros(3),4.,.22)
    with pytest.raises(ValueError):
        p._track_s_route(target(path()+[0.,.1],np.zeros(3)),observation(),np.zeros(3),4.2,.22)
    with pytest.raises(ValueError,match='1/3'):
        p._track_s_route(target(path(),np.zeros(3)),observation(),np.zeros(3),4.4,.22)
    pose=np.array([.06,0.,0.])
    with pytest.raises(ValueError,match='expired away'):
        p._track_s_route(target(path(),pose),observation(),pose,4.6,.22)


def test_settling_before_stop_recovers_only_after_stable_precise_frames():
    p,_=seeded()
    pose=np.array([.02,0.,np.deg2rad(10)])
    for now,count in [(4.,1),(4.2,2)]:
        with pytest.raises(ValueError,match=f'reacquisition {count}/3'):
            p._track_s_route(target(path(),pose),observation(),pose,now,.22)
    assert p._track_s_route(target(path(),pose),observation(),pose,4.4,.22)['s_route_tracking']


def test_reacquisition_requires_stability_from_first_not_previous_frame():
    p,_=seeded()
    for now,x,count in [(4.,.02,1),(4.2,.024,2),(4.4,.028,1)]:
        pose=np.array([x,0.,0.])
        with pytest.raises(ValueError,match=f'reacquisition {count}/3'):
            p._track_s_route(target(path(),pose),observation(),pose,now,.22)


def test_recovery_does_not_relax_current_geometry():
    p,_=seeded()
    with pytest.raises(ValueError,match='precise 8cm'):
        p._track_s_route(target(path()+[0.,.015],np.zeros(3)),observation(),np.zeros(3),4.,.22)


def test_optical_crop_uses_real_boundary_not_centre(monkeypatch):
    import json
    from pathlib import Path
    import pinky_move.robot_projection as projection
    p,_=seeded()
    p.floor_calibration=json.loads((Path(__file__).parents[1]/'config/robot_floor_calibration.json').read_text())
    hidden=np.column_stack((np.linspace(.145,.18,16),np.zeros(16)))
    route=dict(width=.2,boundary_world=hidden+[0.,-.1])
    monkeypatch.setattr(projection,'robot_floor_pixel',lambda *args: (100.,100.))
    assert not p._near_boundary_out_of_view(hidden,route,np.zeros(3))
    def invisible(*args):raise ValueError('Target outside image')
    monkeypatch.setattr(projection,'robot_floor_pixel',invisible)
    assert p._near_boundary_out_of_view(hidden,route,np.zeros(3))
    p.floor_calibration=None
    assert not p._near_boundary_out_of_view(hidden,route,np.zeros(3))


def rotation_case(monkeypatch):
    p, first = seeded()
    boundary = path()+[0.,-.11]
    p.s_route['boundary_world'] = boundary
    monkeypatch.setattr(p, '_near_boundary_out_of_view', lambda *a, **kw: True)
    pose = np.array([0.,0.,.1])
    obs = dict(observation(), curve=target(boundary[30:],pose)['center_path'])
    return p, first, pose, obs


def test_rotation_crop_keeps_committed_goal(monkeypatch):
    p, first, pose, obs = rotation_case(monkeypatch)
    result=p._track_s_route(target(path()[30:],pose),obs,pose,1.2,.22)
    from pinky_move.lane_corner import world_point
    np.testing.assert_allclose(world_point([result['x_m'],result['y_m']],pose),
                               [first['x_m'],first['y_m']],atol=1e-8)
    assert p.debug['s_route_rotation_crop']
    assert p.s_route['covered_m'] == 0.


@pytest.mark.parametrize('case',['visible','wrong_boundary','reverse_boundary','no_boundary','time','travel','yaw'])
def test_rotation_crop_rejects_unsupported_or_unbounded_replay(monkeypatch,case):
    p, _, pose, obs = rotation_case(monkeypatch)
    p._track_s_route(target(path()[30:],pose),obs,pose,1.2,.22)
    now=1.4
    if case=='visible':
        monkeypatch.setattr(p,'_near_boundary_out_of_view',lambda *a,**kw:False)
    if case=='wrong_boundary': obs['curve']=(np.asarray(obs['curve'])+[0.,.03]).tolist()
    if case=='reverse_boundary': obs['curve']=obs['curve'][::-1]
    if case=='no_boundary': obs.pop('curve')
    if case=='time': now=3.05
    if case=='travel': pose=pose+[.061,0.,0.]
    if case=='yaw': pose=pose+[0.,0.,.5]
    old=p.s_route['path'].copy()
    with pytest.raises(ValueError):
        p._track_s_route(target(path()[30:],pose),obs,pose,now,.22)
    np.testing.assert_array_equal(p.s_route['path'],old)


def test_strict_visibility_rejects_bottom_band(monkeypatch):
    import json
    from pathlib import Path
    import pinky_move.robot_projection as projection
    p,_=seeded()
    p.floor_calibration=json.loads((Path(__file__).parents[1]/'config/robot_floor_calibration.json').read_text())
    hidden=np.column_stack((np.linspace(.145,.18,16),np.zeros(16)))
    route=dict(width=.2,boundary_world=hidden+[0.,-.1])
    monkeypatch.setattr(projection,'robot_floor_pixel',lambda *args:(100.,479.))
    assert p._near_boundary_out_of_view(hidden,route,np.zeros(3))
    assert not p._near_boundary_out_of_view(hidden,route,np.zeros(3),strict=True)
