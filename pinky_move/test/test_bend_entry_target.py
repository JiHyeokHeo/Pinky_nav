"""Use observed entrance points, never guessed extra straight travel."""
import numpy as np
import pytest
import pinky_move.lane_corner as lane
from test_lane_corner import observation


def centre(sign=1):
    a=np.column_stack((np.linspace(.08,.20,40),np.zeros(40)))
    t=np.linspace(0,np.pi/2,50)[1:]
    b=np.column_stack((.20+.12*np.sin(t),sign*.12*(1-np.cos(t))))
    return np.vstack((a,b))


@pytest.mark.parametrize('sign',[1,-1])
def test_far_target_is_limited_to_measured_entrance(sign):
    p=centre(sign)
    target=dict(x_m=float(p[-1,0]),y_m=float(p[-1,1]),center_path=p.tolist())
    result=lane.limit_bend_entry(target)
    assert result['bend_entry_limited']
    assert .15 < result['x_m'] < .25
    assert abs(result['y_m']) < .03
    assert result['center_path'] == target['center_path']
    np.testing.assert_allclose(result['approach_path'][-1], [result['x_m'],result['y_m']])
    np.testing.assert_allclose(result['approach_path'][-1],result['bend_remaining_path'][0])
    np.testing.assert_allclose(result['bend_remaining_path'][-1],p[-1])


def test_already_near_target_straight_path_and_no_target_are_unchanged():
    assert lane.limit_bend_entry(None) is None
    target=dict(x_m=.1,y_m=0.,center_path=centre().tolist())
    assert lane.limit_bend_entry(target) is target


def test_replaced_path_cannot_keep_old_approach_display():
    p=centre()
    target=lane.limit_bend_entry(dict(x_m=float(p[-1,0]),y_m=float(p[-1,1]),center_path=p.tolist()))
    target['center_path']=np.column_stack((np.linspace(.08,.4,100),np.zeros(100))).tolist()
    result=lane.limit_bend_entry(target)
    assert 'approach_path' not in result and 'bend_entry_limited' not in result


def test_entry_cap_releases_when_robot_reaches_entrance():
    p=centre()-[.20,0.]
    target=dict(x_m=float(p[-1,0]),y_m=float(p[-1,1]),center_path=p.tolist())
    assert lane.limit_bend_entry(target) is target
    target['center_path']=np.column_stack((np.linspace(.08,.4,100),np.zeros(100))).tolist()
    target['x_m']=.35
    assert lane.limit_bend_entry(target) is target


def test_gentle_corner_keeps_valid_centre_when_inside_approach_radius(monkeypatch):
    policy=lane.CornerPolicy(lane.CornerConfig(staged_turn=True))
    obs=observation()
    feature=dict(lane.classify_boundary(obs['curve'],policy.config),
                 angle_deg=60.,corner=np.array([.20,0.]))
    monkeypatch.setattr(lane,'classify_boundary',lambda *args:feature)
    monkeypatch.setattr(lane,'plan_corner',lambda *args:pytest.fail('discarded valid curve'))
    target=dict(x_m=.3,y_m=.1,center_path=centre().tolist(),boundary_count=1)
    for now in (1.,1.2,1.4):
        result=policy.update(obs,target,now,np.zeros(3),.22,.25)
        assert result is not None and result['bend_entry_limited']
        assert policy.staged is None
