"""One current trusted stripe may show only a single bend of a stored S."""
import numpy as np
import pytest
import pinky_move.lane_corner as lane
from test_s_bend_policy import s_observation
from test_lane_corner import boundary
from test_lane_controller import controller
from types import SimpleNamespace


def partial_feature(sign=1):
    return dict(lane.classify_boundary(boundary(sign), lane.CornerConfig()),
                angle_deg=sign*55., reason='single visible bend')


@pytest.mark.parametrize('sign', [1, -1])
def test_single_boundary_partial_corner_keeps_route_beyond_two_seconds(monkeypatch, sign):
    p = lane.CornerPolicy(lane.CornerConfig(staged_turn=True))
    obs = s_observation(sign)
    target = p.update(obs, None, 1., np.zeros(3), .22, .25)
    assert target['boundary_count'] == 1
    feature = partial_feature(sign)
    monkeypatch.setattr(lane, 'classify_boundary', lambda *args: feature)
    # The old S identity expires, but each fresh matching curve independently
    # validates route continuation. No loss timer for a continuously seen line.
    for now in np.arange(1.2, 5.1, .2):
        target = p.update(obs, None, float(now), np.zeros(3), .22, .25)
        assert target is not None
        assert target['s_route_tracking'] and target['boundary_count'] == 1
        assert p.staged is None and not target.get('corner_stationary')


@pytest.mark.parametrize('case', ['shift', 'side', 'gap'])
def test_partial_label_does_not_bypass_real_route_checks(monkeypatch, case):
    p = lane.CornerPolicy()
    obs = s_observation()
    assert p.update(obs, None, 1., np.zeros(3), .22, .25)
    before = p.s_route['path'].copy()
    feature = partial_feature()
    monkeypatch.setattr(lane, 'classify_boundary', lambda *args: feature)
    now = 1.2
    if case == 'shift': obs = dict(obs, curve=obs['curve']+[0., .1])
    if case == 'side': obs = dict(obs, side='left')
    if case == 'gap': now = 3.1
    assert p.update(obs, None, now, np.zeros(3), .22, .25) is None
    assert p.block_recovery and p.staged is None
    np.testing.assert_array_equal(p.s_route['path'], before)


@pytest.mark.parametrize('far_context',[False,True])
@pytest.mark.parametrize('error,allowed', [('inferred target discontinuity',True),
    ('visible boundary identity ambiguous',False), ('no forward centre path',False)])
def test_single_line_node_revalidates_geometry_not_ambiguous_identity(controller,error,allowed,far_context):
    node=controller
    obs=s_observation()
    node.corner_policy=lane.CornerPolicy()
    node.corner_policy.update(obs,None,1.,np.zeros(3),.22,.25)
    node.safety_clock.seconds=1.2
    node.robot_calibration={'test':True}
    node.parameters['corner_enabled']=True
    node._corner_pose_for_frame=lambda *args:np.zeros(3)
    def reject(*args,**kwargs):
        raise ValueError(error)
    node.metric_tracker=SimpleNamespace(update=reject,last_observation=obs,
                                       context_only_indices=[1] if far_context else [])
    masks=[np.zeros((20,20),np.uint8)]*(2 if far_context else 1)
    count=node._update_lane_command(masks,20,node.safety_clock.now())
    assert (count==1) == allowed
    assert (node.metric_target is not None) == allowed
    if allowed:
        assert node.metric_target['s_route_tracking']
