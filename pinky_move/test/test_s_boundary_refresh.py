"""New observed tails are not contradictions; only verified overlap extends history."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerPolicy, CornerConfig


def line(start=.1,end=.4,y=-.1):
    return np.column_stack((np.linspace(start,end,80),np.full(80,y)))


def test_novel_tail_does_not_dilute_common_match_ratio():
    ok,detail,_=CornerPolicy._boundary_overlap(line(.3,.9),line())
    assert ok and detail['ratio']==1.
    assert detail['new_points']>60
    assert detail['overlap_m']>.09


@pytest.mark.parametrize('case',['shift','reverse','no_overlap','tiny_overlap','interior_jump'])
def test_common_support_rejects_wrong_or_insufficient_boundary(case):
    actual=line(.3,.9)
    if case=='shift': actual[:,1]+=.1
    if case=='reverse': actual=actual[::-1]
    if case=='no_overlap': actual=line(.5,.9)
    if case=='tiny_overlap': actual=line(.395,.9)
    if case=='interior_jump':
        actual=line(.15,.5);actual[15:35,1]+=.1
    ok,detail,_=CornerPolicy._boundary_overlap(actual,line())
    assert not ok and detail['reason']!='matched common support'


def test_successful_route_appends_real_boundary_without_rewriting_pending_points():
    p=CornerPolicy(CornerConfig())
    pose=np.zeros(3)
    def obs(end):return dict(side='right',width=.2,curve=line(.1,end))
    def goal(end):return dict(x_m=.22,y_m=0.,center_path=line(.1,end,0.).tolist(),boundary_count=1)
    p._track_s_route(goal(.4),obs(.4),pose,1.,.22)
    old_boundary=p.s_route['boundary_world'].copy()
    old_path=p.s_route['path'].copy()
    before=p._track_s_route(goal(.6),obs(.6),pose,1.2,.22)
    np.testing.assert_array_equal(p.s_route['boundary_world'][:len(old_boundary)],old_boundary)
    np.testing.assert_allclose(p.s_route['path'][:len(old_path)],old_path,atol=1e-10)
    assert p.s_route['boundary_world'][-1,0]>.59
    assert before['x_m']==pytest.approx(.22)
    assert p.debug['s_boundary_tail_appended']
    assert p.s_route['boundary_verified_at']==1.2


def test_bad_refresh_does_not_overwrite_boundary_or_mark_verified():
    p=CornerPolicy(CornerConfig())
    stored=line(); route=dict(boundary_world=stored.copy())
    p._refresh_s_boundary(route,dict(curve=line(.1,.6,y=.2)),np.zeros(3),1.)
    np.testing.assert_array_equal(route['boundary_world'],stored)
    assert 'boundary_verified_at' not in route


def test_failed_route_does_not_commit_boundary_refresh():
    p=CornerPolicy(CornerConfig())
    obs=dict(side='right',width=.2,curve=line())
    target=dict(x_m=.22,y_m=0.,center_path=line(y=0.).tolist(),boundary_count=1)
    p._track_s_route(target,obs,np.zeros(3),1.,.22)
    saved=p.s_route['boundary_world'].copy()
    bad=dict(target,center_path=line(.1,.7,y=.15).tolist())
    with pytest.raises(ValueError):
        p._track_s_route(bad,dict(obs,curve=line(.1,.7)),np.zeros(3),1.2,.22)
    np.testing.assert_array_equal(p.s_route['boundary_world'],saved)


def test_history_accepts_shared_support_without_refreshing_its_timer():
    p=CornerPolicy(CornerConfig())
    target=dict(x_m=.22,y_m=0.,center_path=line(y=0.).tolist(),boundary_count=1)
    obs=dict(side='right',width=.2,curve=line())
    p._track_s_route(target,obs,np.zeros(3),1.,.22)
    result=p._saved_s_target(dict(obs,curve=line(.3,.9)),np.zeros(3),1.2,.22,'crop')
    assert result['s_route_history'] and p.s_route['time']==1.
    assert p.debug['s_history_boundary_ratio']==1.
