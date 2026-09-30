"""Corner decisions and geometry tested without cameras, services or motors."""
import numpy as np
import pytest
from pinky_move.lane_corner import (CornerConfig, CornerPolicy, classify_boundary,
                                    plan_corner, world_point)
from pinky_move.metric_lane import first_self_intersection


def boundary(sign=1):
    entry = np.column_stack((np.linspace(.1,.5,80),np.full(80,-.2)))
    exit_leg = np.column_stack((np.full(79,.5),np.linspace(-.2,.2,80)[1:]))
    p = np.vstack((entry,exit_leg))
    p[:,1] *= sign
    return p


def observation(sign=1):
    return dict(curve=boundary(sign),side='right' if sign==1 else 'left',
                width=.4,width_source='configured')


def ordinary():
    return dict(x_m=.22,y_m=0.,inferred=True,boundary_count=1,normal_width_m=.4)


@pytest.mark.parametrize('sign', [1,-1])
def test_localized_corner_has_supported_direction(sign):
    f = classify_boundary(boundary(sign), CornerConfig())
    assert f['kind']=='CORNER'
    assert sign*f['angle_deg'] > 80
    np.testing.assert_allclose(f['corner'], [.5,-sign*.2], atol=.002)


def test_straight_s_bend_spike_and_incomplete_exit():
    x = np.linspace(.1,.7,100)
    cfg = CornerConfig()
    assert classify_boundary(np.column_stack((x,np.full(100,.1))),cfg)['kind']=='NORMAL'
    s = np.column_stack((x,.04*np.sin(np.linspace(0,2*np.pi,100))))
    assert classify_boundary(s,cfg)['kind']=='S_BEND'
    short = boundary()[:84]
    assert classify_boundary(short,cfg)['kind'] != 'CORNER'
    assert classify_boundary(np.array([[.2,.1],[.21,.1]]),cfg)['kind']=='UNKNOWN'
    noisy = np.column_stack((x,.1+.0005*np.sin(np.arange(100))))
    assert classify_boundary(noisy,cfg)['kind']!='CORNER'


@pytest.mark.parametrize('sign',[1,-1])
def test_swept_footprint_path_and_angular_cap(sign):
    obs=observation(sign);cfg=CornerConfig()
    feature=classify_boundary(obs['curve'],cfg)
    path,cap=plan_corner(obs,feature,cfg,.15)
    assert len(path)>=60 and np.all(path[:,0]>.05)
    assert first_self_intersection(path) is None
    assert 0 < cap <= cfg.speed_mps
    heading=np.unwrap(np.arctan2(*np.gradient(path,axis=0)[:,::-1].T))
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    assert cap*max(abs(np.gradient(heading,distance))) <= .150001


def test_reject_narrow_conflicting_pair_and_missing_footprint_support():
    obs=observation();cfg=CornerConfig();f=classify_boundary(obs['curve'],cfg)
    with pytest.raises(ValueError,match='width insufficient'):
        plan_corner(dict(obs,width=.154),f,cfg,.15)
    with pytest.raises(ValueError,match='no supported'):
        plan_corner(dict(obs,other=obs['curve']+np.array([0.,-.05])),f,cfg,.15)
    with pytest.raises(ValueError,match='too short'):
        plan_corner(obs,dict(f,entry=np.array([.47,-.2])),cfg,.15)
    with pytest.raises(ValueError,match='no supported'):
        plan_corner(obs,f,cfg,.0001)


def test_zero_painted_line_margin_retains_physical_robot_width():
    cfg = CornerConfig(clearance_m=0.)
    cfg.validate()
    obs = observation()
    feature = classify_boundary(obs['curve'], cfg)
    with pytest.raises(ValueError, match='width insufficient'):
        plan_corner(dict(obs, width=.149), feature, cfg, .15)
    with pytest.raises(ValueError, match='invalid corner configuration'):
        CornerConfig(clearance_m=-.001).validate()


def test_corner_search_can_follow_visible_boundary_closer_than_lanecentre():
    obs = observation()
    cfg = CornerConfig(clearance_m=0.)
    feature = classify_boundary(obs['curve'], cfg)
    path, _ = plan_corner(obs, feature, cfg, .15)
    entry = feature['entry']+(cfg.rear_m+.005)*feature['tin']
    inward = np.array([-feature['tin'][1], feature['tin'][0]])
    offset = abs(float((path[0]-entry)@inward))
    assert cfg.half_width_m <= offset < obs['width']*.5


def test_corner_support_uses_footprint_not_duplicate_fixed_padding():
    obs = observation()
    cfg = CornerConfig(clearance_m=0.)
    feature = classify_boundary(obs['curve'], cfg)
    # The observed legs clear the real 9 cm rear / 6 cm front, but not the
    # old extra 3 cm fixed gate. The full footprint sweep remains mandatory.
    feature = dict(feature,
                   entry=feature['corner']-.11*feature['tin'],
                   exit=feature['corner']+.08*feature['tout'])
    try:
        plan_corner(obs, feature, cfg, .15)
    except ValueError as exc:
        assert 'observed legs too short' not in str(exc)


def test_three_fresh_confirmations_not_three_duplicate_frames():
    p=CornerPolicy();obs=observation()
    assert p.update(obs,None,1.,np.zeros(3),.22,.15) is None
    assert p.update(obs,None,1.,np.zeros(3),.22,.15) is None
    assert p.count==1
    assert p.update(obs,None,1.2,np.zeros(3),.22,.15) is None
    result=p.update(obs,None,1.4,np.zeros(3),.22,.15)
    assert result['corner_path'] and result['corner_speed_cap']<=.01
    assert p.debug['confirmed'] and p.state=='APPROACH'
    assert p.block_recovery


def test_valid_centre_path_continues_through_far_corner_candidate_and_approach():
    p = CornerPolicy()
    obs = observation()
    for now in (1., 1.2, 1.4):
        result = p.update(obs, ordinary(), now, np.zeros(3), .22, .15)
        assert result is not None and not result.get('corner_path')
        assert result['corner_speed_cap'] == p.config.speed_mps
        assert not p.block_recovery
    assert p.state == 'APPROACH' and p.confirmed_exit is None
    # Once the same corner is within the lookahead + robot-front distance,
    # the special path is required and normal centre following stops.
    result = p.update(obs, ordinary(), 1.6, np.zeros(3), .50, .15)
    assert result is not None and result['corner_path']
    assert p.block_recovery


def test_odometry_compensates_landmark_and_rejects_unrelated_corner():
    policy=CornerPolicy();obs=observation()
    for i in range(3):
        translation=i*.015
        current=dict(obs,curve=obs['curve']-np.array([translation,0.]))
        result=policy.update(current,None,1.+i*.2,np.array([translation,0.,0.]),.22,.15)
    assert result and policy.count==3
    displaced=dict(obs,curve=obs['curve']+np.array([.2,0.]))
    assert policy.update(displaced,ordinary(),1.6,np.zeros(3),.22,.15) is None
    assert policy.count==1
    np.testing.assert_allclose(world_point(np.array([1.,0.]),[2.,3.,np.pi/2]),[2.,4.])


def test_missing_odom_gap_and_wrong_direction_do_not_confirm():
    p=CornerPolicy();obs=observation()
    for now in (1.,1.2,1.4):
        assert p.update(obs,ordinary(),now,None,.22,.15) is None
    assert p.count==0 and p.block_recovery
    p.update(obs,ordinary(),1.6,np.zeros(3),.22,.15)
    p.update(obs,ordinary(),2.5,np.zeros(3),.22,.15)
    assert p.count==1
    p.update(observation(-1),ordinary(),2.7,np.zeros(3),.22,.15)
    assert p.count==1


def test_confirmed_corner_cannot_disappear_into_normal_without_exit_check():
    p=CornerPolicy();obs=observation()
    for now in (1.,1.2,1.4):p.update(obs,None,now,np.zeros(3),.22,.15)
    assert p.confirmed_exit is not None
    assert p.update(None,None,1.6,np.zeros(3),.22,.15) is None
    assert p.block_recovery and p.confirmed_exit is not None
    straight=dict(obs,curve=np.column_stack((np.linspace(.1,.7,100),np.full(100,-.2))))
    assert p.update(straight,ordinary(),1.8,np.zeros(3),.22,.15) is None
    for i in range(3):
        result=p.update(straight,ordinary(),2.+i*.2,np.array([0.,0.,np.pi/2]),.22,.15)
        assert result
    assert p.state=='NORMAL' and not p.block_recovery


@pytest.mark.parametrize('failure', ['plan_corner', 'select_lookahead'])
def test_failed_corner_never_arms_exit_and_normal_path_can_resume(monkeypatch, failure):
    import pinky_move.lane_corner as module
    def reject(*args, **kwargs):
        raise ValueError('test rejected candidate')
    monkeypatch.setattr(module, failure, reject)
    p = CornerPolicy(); obs = observation()
    for now in (1., 1.2, 1.4):
        assert p.update(obs, None, now, np.zeros(3), .22, .15) is None
    assert p.confirmed_exit is None
    straight = dict(obs, curve=np.column_stack((np.linspace(.1, .7, 100), np.full(100, -.2))))
    assert p.update(straight, ordinary(), 1.6, np.zeros(3), .22, .15) == ordinary()
    assert not p.block_recovery


def test_failed_replan_retains_previous_valid_exit(monkeypatch):
    import pinky_move.lane_corner as module
    p = CornerPolicy(); obs = observation()
    for now in (1., 1.2, 1.4):
        p.update(obs, None, now, np.zeros(3), .22, .15)
    previous = p.confirmed_exit
    assert previous is not None
    def reject(*args, **kwargs):
        raise ValueError('test rejected replan')
    monkeypatch.setattr(module, 'plan_corner', reject)
    assert p.update(obs, None, 1.6, np.array([0., 0., .1]), .22, .15) is None
    assert p.confirmed_exit == previous


def test_short_entry_reaches_sweep_without_bypassing_coverage(monkeypatch):
    import pinky_move.lane_corner as module
    cfg = CornerConfig(clearance_m=0.)
    obs = observation()
    feature = classify_boundary(obs['curve'], cfg)
    feature = dict(feature, entry=feature['corner']-.089*feature['tin'])
    # Previously rejected immediately (8.9 cm leg - 9.5 cm trim = -6 mm).
    # It must now reach full footprint validation, not automatically pass it.
    queries = []
    original = module.nearest_on_chain
    def measured(query, bound):
        queries.append(len(query))
        return original(query, bound)
    monkeypatch.setattr(module, 'nearest_on_chain', measured)
    with pytest.raises(ValueError, match='no supported'):
        plan_corner(obs, feature, cfg, .25)
    assert queries and min(queries) == 70*9
    # Removing actual observed support must still prevent a turn.
    clipped = dict(obs, curve=obs['curve'][obs['curve'][:, 0] >= .45])
    with pytest.raises(ValueError, match='no supported'):
        plan_corner(clipped, feature, cfg, .25)


@pytest.mark.parametrize('values',[dict(angle_deg=0),dict(confirm_frames=1),
    dict(window_m=.01),dict(speed_mps=float('nan')),dict(min_speed_mps=.1)])
def test_invalid_configuration_rejected(values):
    with pytest.raises(ValueError):CornerPolicy(CornerConfig(**values))
