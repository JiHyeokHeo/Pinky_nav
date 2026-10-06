"""Local map retention is independent of permission to move blindly."""
import numpy as np
import pytest
from pinky_move.lane_history import LocalLaneMap
from pinky_move.lane_corner import CornerPolicy
from pinky_move.lane_corner import CornerConfig


def scene(side='left'):
    x=np.linspace(.14,.4,80)
    curve=np.column_stack((x,np.full_like(x,.08 if side=='left' else -.08)))
    path=np.column_stack((x,np.zeros_like(x)))
    return dict(side=side,curve=curve,width=.16), dict(
        center_path=path.tolist(),x_m=.22,y_m=0.,boundary_count=1,inferred=True)


def populated():
    m=LocalLaneMap()
    obs,target=scene()
    for t in (1.,1.2):
        assert m.prepare(t,np.zeros(3))
        m.remember(obs,t,np.zeros(3))
    return m,obs,target


def test_recent_map_confirms_only_near_current_path():
    m,obs,target=populated()
    result=m.current_target(obs,target,1.4,np.zeros(3),.7)
    assert result['local_map_assisted']
    assert result['corner_speed_cap']==.02
    assert .20<result['x_m']<.25
    assert len(target['center_path'])==80


def test_map_role_matches_current_transformed_pixels_not_image_side():
    m, obs, _ = populated()
    pose = np.array([.02, .12, 0.])
    current = obs['curve']-pose[:2]
    assert np.all(current[:,1]<0)  # 화면 중심 반대편으로 이동해도 같은 실관측 선.
    assert m.measured_side(current, 1.4, pose) == 'left'
    assert m.measured_side(current, 11.3, pose) is None
    assert m.measured_side(current+[0., .05], 1.4, pose) is None
    assert m.measured_side(current[::-1], 1.4, pose) is None


def test_map_role_requires_two_unique_times_and_unique_side():
    m = LocalLaneMap()
    obs, _ = scene()
    m.remember(obs, 1., np.zeros(3))
    m.remember(obs, 1., np.zeros(3))
    assert m.measured_side(obs['curve'], 1.4, np.zeros(3)) is None
    m.remember(obs, 1.2, np.zeros(3))
    assert m.measured_side(obs['curve'], 1.4, np.zeros(3)) == 'left'
    for t in (1.,1.2):
        m.remember(dict(obs, side='right'), t, np.zeros(3))
    assert m.measured_side(obs['curve'], 1.4, np.zeros(3)) is None


def test_odom_translation_keeps_measured_world_boundary():
    m,obs,target=populated()
    pose=np.array([.02,0.,0.])
    obs=dict(obs,curve=obs['curve']-pose[:2])
    target=dict(target,center_path=(np.asarray(target['center_path'])-pose[:2]).tolist())
    assert m.current_target(obs,target,1.4,pose,.7) is not None


def test_ten_seconds_retained_but_old_evidence_cannot_drive():
    m,obs,target=populated()
    m.prepare(10.9,np.zeros(3))
    assert len(m.frames)==2
    assert m.current_target(obs,target,10.9,np.zeros(3),.7) is None
    m.prepare(11.3,np.zeros(3))
    assert not m.frames


@pytest.mark.parametrize('change',['side','width','shift','empty','held'])
def test_invalid_or_conflicting_observations_do_not_bypass(change):
    m,obs,target=populated()
    if change=='side': obs=dict(obs,side='right')
    if change=='width': obs=dict(obs,width=.25)
    if change=='shift': obs=dict(obs,curve=obs['curve']+[0,.04])
    if change=='empty': obs=None
    if change=='held': target=dict(target,held=True)
    assert m.current_target(obs,target,1.4,np.zeros(3),.7) is None


def test_duplicate_missing_pose_and_jump():
    m,obs,target=populated()
    assert not m.prepare(1.2,np.zeros(3))
    assert not m.prepare(1.3,None)
    assert m.prepare(1.4,np.array([1.,0.,0.]))
    assert not m.frames


def test_corner_integration_uses_map_only_for_route_shape_conflict(monkeypatch):
    p=CornerPolicy()
    obs,target=scene()
    p.s_route={'side':'left'}
    def stopped(*args):
        p.debug={'reason':'S bend: S route new path skips or contradicts pending bend'}
        return None
    monkeypatch.setattr(p,'_update',stopped)
    assert p.update(obs,target,1.,np.zeros(3),.22,.5) is None
    assert p.update(obs,target,1.2,np.zeros(3),.22,.5) is None
    assert p.update(obs,target,1.4,np.zeros(3),.22,.5)['local_map_assisted']
    assert p.s_route=={'side':'left'}  # Never overwrite the original S exit.
    p.staged={}
    assert p.update(obs,target,1.6,np.zeros(3),.22,.5) is None
    p.staged=None
    monkeypatch.setattr(p,'_update',lambda *a: None)
    p.debug={'reason':'S route missing boundary not verified'}
    assert p.update(obs,target,1.8,np.zeros(3),.22,.5) is None


@pytest.mark.parametrize('preserve',[False,True])
def test_current_map_recovery_can_preserve_untraversed_s_route(monkeypatch,preserve):
    p=CornerPolicy(CornerConfig(relaxed_tracking=True))
    obs,target=scene()
    target=dict(target,preserve_pending_s=preserve)
    route={'side':'left','exit_world':np.array([.7,.4])}
    p.s_route=route
    def stopped(*args):
        p.debug={'reason':'S bend: S route rotation visibility budget exhausted'}
        return None
    monkeypatch.setattr(p,'_update',stopped)
    for now in (1.,1.2,1.4):
        result=p.update(obs,target,now,np.zeros(3),.22,.5)
        if result is not None:
            break
    assert result['local_map_assisted']
    if preserve:
        assert p.s_route is route
        assert p.debug['s_route_preserved_by_current_map']
        np.testing.assert_array_equal(p.s_route['exit_world'],[.7,.4])
    else:
        assert p.s_route is None
        assert p.debug['s_route_replaced_by_current_map']


def test_unknown_label_cannot_erase_pending_s_when_current_geometry_is_marked(monkeypatch):
    import pinky_move.lane_corner as module
    p=CornerPolicy(CornerConfig(relaxed_tracking=True))
    obs,current=scene()
    current=dict(current,preserve_pending_s=True)
    route={'side':'left','exit_world':np.array([.7,.4])}
    p.s_route=route
    feature=dict(kind='UNKNOWN',angle_deg=0.,reason='insufficient observed legs')
    monkeypatch.setattr(module,'classify_boundary',lambda *a:feature)
    monkeypatch.setattr(p,'_s_bend_feature',lambda obs,f,*a:f)
    monkeypatch.setattr(p,'_history_feature',lambda obs,f,*a:(f,0))
    monkeypatch.setattr(p,'_s_bend_target',lambda obs,target,*a:target)
    monkeypatch.setattr(p,'_track_s_route',lambda target,*a:dict(target,s_route_tracking=True))
    result=p.update(obs,current,1.,np.zeros(3),.22,.5)
    assert result['s_route_tracking']
    assert p.s_route is route
