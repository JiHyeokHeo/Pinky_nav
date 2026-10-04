"""A small isolated distance mismatch needs stable evidence, not a blanket waiver."""
import numpy as np
import pytest
from test_s_route_progress import seeded, path, target, observation


@pytest.mark.parametrize('start',[1.2,4.])
def test_three_stable_frames_accept_227mm_without_replacing_goal(start):
    p,first=seeded(); old=p.s_route['path'].copy()
    for now,count in [(start,1),(start+.2,2)]:
        with pytest.raises(ValueError,match=f'confirmation {count}/3'):
            p._track_s_route(target(path()+[0.,.0227],np.zeros(3)),observation(),np.zeros(3),now,.22)
        assert p.s_route['time']==1.
    result=p._track_s_route(target(path()+[0.,.0227],np.zeros(3)),observation(),np.zeros(3),start+.4,.22)
    assert result['s_distance_reacquired'] and result['corner_speed_cap']<=.02
    np.testing.assert_allclose([result['x_m'],result['y_m']],[first['x_m'],first['y_m']],atol=1e-8)
    np.testing.assert_allclose(p.s_route['path'][:len(old)],old,atol=1e-8)
    # Subsequent fresh matching frames can continue without stop/start cycling.
    assert p._track_s_route(target(path()+[0.,.0227],np.zeros(3)),observation(),np.zeros(3),start+.6,.22)['s_distance_reacquired']


@pytest.mark.parametrize('case',['too_far','reversed','width','side','moving'])
def test_bad_evidence_cannot_accumulate_confirmation(case):
    p,_=seeded(); pose=np.zeros(3); obs=observation(); curve=path()+[0.,.0227]
    with pytest.raises(ValueError,match='confirmation 1/3'):
        p._track_s_route(target(curve,pose),obs,pose,1.2,.22)
    if case=='too_far':curve=path()+[0.,.03]
    if case=='reversed':curve=curve[::-1]
    if case=='width':obs=dict(obs,width=.4)
    if case=='side':obs=dict(obs,side='left')
    if case=='moving':pose=np.array([.006,0.,0.])
    with pytest.raises(ValueError):
        p._track_s_route(target(curve,pose),obs,pose,1.4,.22)
    assert p.s_route['time']==1.
    with pytest.raises(ValueError,match='confirmation 1/3'):
        p._track_s_route(target(path()+[0.,.0227],np.zeros(3)),observation(),np.zeros(3),1.6,.22)


def test_small_distance_requires_ten_cm_support_and_parallel_tangents():
    p,_=seeded()
    short=(path()+[0.,.0227])[:10]
    with pytest.raises(ValueError):
        p._track_s_route(target(short,np.zeros(3)),observation(),np.zeros(3),1.2,.22)
    assert p.s_route_reacquire is None


def test_gap_restarts_confirmations():
    p,_=seeded()
    for now in (1.2,2.2):
        with pytest.raises(ValueError,match='confirmation 1/3'):
            p._track_s_route(target(path()+[0.,.0227],np.zeros(3)),observation(),np.zeros(3),now,.22)
