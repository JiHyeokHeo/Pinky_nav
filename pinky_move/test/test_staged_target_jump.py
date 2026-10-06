"""Only confirmed staged control may ignore an ordinary target jump."""
import numpy as np
import pytest
from types import SimpleNamespace
from test_lane_controller import controller
from test_staged_corner import prepared
from test_lane_corner import observation
from pinky_move.lane_corner import CornerConfig, CornerPolicy


def test_hybrid_s_geometry_keeps_connected_legs_instead_of_axis_shortcut(controller, monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    node = controller
    node.parameters['simulation_yolo_white'] = True
    node.robot_calibration = {}
    node.corner_policy.config.staged_turn = False  # 연속 S 추종 프로파일
    vertices = np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]])
    curve = chain(vertices)
    obs = dict(curve=curve[:20], side='right', width=.2)
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *args:[curve])
    measured, target = node._hybrid_connected_observation([np.ones((20,20))],obs,None)
    assert len(measured['curve']) > len(obs['curve'])
    assert target is not None and target['source_mask_index'] == 0
    assert abs(target['y_m']) < .01  # 아직 먼 굽힘을 보고 먼저 회전하지 않는다.
    np.testing.assert_allclose(np.asarray(target['center_path'])[-1],[.9,.5],atol=.01)
    # 중복/모호한 연결 경계나 미배정 선은 승격하지 않는다.
    assert node._hybrid_connected_observation([np.ones((20,20))]*2,obs,None) == (obs,None)
    assert node._hybrid_connected_observation([np.ones((20,20))],None,None) == (None,None)
    node.parameters['simulation_yolo_white'] = False
    assert node._hybrid_connected_observation([np.ones((20,20))],obs,None) == (obs,None)


def test_single_current_assigned_mask_survives_axis_interpolation_shortcut(controller, monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    node = controller
    node.parameters['simulation_yolo_white'] = True
    node.robot_calibration = {}
    node.corner_policy.config.staged_turn = False
    curve = chain(np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]]))
    # 축 샘플러가 같은 마스크의 여러 다리 사이를 잘못 보간한 현재 관측.
    shortcut = np.column_stack((np.linspace(.14,.48,20),np.linspace(-.1,.15,20)))
    obs = dict(curve=shortcut, side='right', width=.2)
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *args:[curve])
    measured, target = node._hybrid_connected_observation([np.ones((20,20))],obs,None)
    assert target is not None and abs(target['y_m']) < .01
    np.testing.assert_array_equal(measured['curve'],curve)


def test_hybrid_sharp_s_uses_first_supported_staged_corner(controller,monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    from pinky_move.lane_corner import classify_boundary
    controller.parameters['simulation_yolo_white']=True
    controller.robot_calibration={}
    controller.corner_policy.config.staged_turn=True
    curve=chain(np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]]))
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *a:[curve])
    obs=dict(curve=curve[:20],side='right',width=.2)
    ordinary=dict(x_m=.22,y_m=0.,center_path=[[.14,0.],[.4,0.]])
    measured,target=controller._hybrid_connected_observation([np.ones((20,20))],obs,ordinary)
    assert measured['sharp_s_prefix']
    assert classify_boundary(measured['curve'],controller.corner_policy.config)['kind']=='CORNER'
    assert target is ordinary  # 일반 진입 목표는 없애거나 미리 회전시키지 않는다.
    assert controller.metric_tracker.preferred_observation_side=='right'


@pytest.mark.parametrize('pixel_jitter',[0.,.006])
def test_already_measured_full_s_does_not_require_axis_rematching(controller,monkeypatch,pixel_jitter):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    from pinky_move.lane_corner import classify_boundary
    controller.parameters['simulation_yolo_white']=True
    controller.robot_calibration={}
    controller.corner_policy.config.staged_turn=True
    curve=chain(np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]]))
    curve[1,1]+=pixel_jitter
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *a:pytest.fail('re-matched trusted full S'))
    obs=dict(curve=curve,side='right',width=.2)
    measured,current=controller._hybrid_connected_observation([np.ones((20,20))],obs,None)
    assert measured['sharp_s_prefix']
    assert classify_boundary(measured['curve'],controller.corner_policy.config)['kind']=='CORNER'
    assert current is None


def test_pair_role_can_match_precise_near_leg_when_axis_far_tail_is_wrong(controller,monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    controller.parameters['simulation_yolo_white']=True
    controller.robot_calibration={}
    controller.corner_policy.config.staged_turn=True
    left=np.column_stack((np.linspace(.14,.48,30),np.full(30,.1)))
    right=chain(np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]]))
    # 먼 구간만 축 보간으로 잘못 이어진 기준. 가까운 실경계 역할은 유효하다.
    other=chain(np.array([[.14,-.1],[.3,-.1],[.9,-.5]]))
    masks=[np.ones((20,20)),np.ones((20,20))*2]
    monkeypatch.setattr(lane,'connected_floor_curve',lambda m,*a:[left if m is masks[0] else right])
    obs=dict(curve=left,other=other,side='left',width=.2,width_source='measured')
    measured,_=controller._hybrid_connected_observation(masks,obs,None)
    assert measured['side']=='right' and measured['sharp_s_prefix']
    # 가까운 역할도 불일치하면 먼 형상만으로 대체하지 않는다.
    measured,_=controller._hybrid_connected_observation(masks,dict(obs,other=other+[0.,.08]),None)
    assert measured is obs or measured['side']=='left'


def test_current_connected_corner_center_prevents_far_entry_commit(controller,monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    controller.parameters['simulation_yolo_white']=True
    controller.robot_calibration={}
    controller.corner_policy=CornerPolicy(CornerConfig(staged_turn=True,clearance_m=0.))
    curve=chain(np.array([[.14,-.1],[.5,-.1],[.5,.4]]))
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *a:[curve])
    observed=dict(curve=curve[:20],side='right',width=.2)
    measured,target=controller._hybrid_connected_observation([np.ones((20,20))],observed,None)
    assert target is not None and abs(target['y_m'])<.01
    for now in (1.,1.2,1.4):
        result=controller.corner_policy.update(measured,target,now,np.zeros(3),.22,.25)
    assert result is not None and not result.get('corner_staged')
    assert controller.corner_policy.staged is None
    assert controller.corner_policy.debug['reason']=='ordinary centre path approaches distant confirmed corner'


def test_valid_pair_can_use_s_geometry_from_the_other_assigned_boundary(controller, monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    node = controller
    node.parameters['simulation_yolo_white'] = True
    node.robot_calibration = {}
    node.corner_policy.config.staged_turn = False
    left = np.column_stack((np.linspace(.14,.48,30),np.full(30,.1)))
    right = chain(np.array([[.14,-.1],[.5,-.1],[.5,.4],[.9,.4]]))
    masks = [np.ones((20,20)),np.ones((20,20))*2]
    monkeypatch.setattr(lane,'connected_floor_curve',lambda m,*args:[left if m is masks[0] else right])
    obs = dict(curve=left, other=right[:20], side='left', width=.2, width_source='measured')
    measured,target = node._hybrid_connected_observation(masks,obs,None)
    assert measured['side']=='right' and target['source_mask_index']==1
    assert node.metric_tracker.preferred_observation_side == 'right'
    assert abs(target['y_m']) < .01
    # 미관측/미확인 폭으로 반대 경계의 역할을 새로 만들지 않는다.
    measured,target = node._hybrid_connected_observation(masks,dict(obs,width_source='configured'),None)
    assert target is None and measured['side']=='left'


def test_capture_pose_survives_own_geometry_delay_but_other_frame_cannot_reuse_it(controller):
    node = controller
    node.parameters['simulation_yolo_white'] = True
    node.corner_capture_stamp = 123
    capture_pose = np.array([.4,.1,.0])
    node.hybrid_frame_pose = (123,capture_pose)
    node._corner_pose_for_frame = lambda *args: None  # 현재 odom이 처리 지연으로 stale.
    assert node._geometry_frame_pose(node.safety_clock.now()) is capture_pose
    node.corner_capture_stamp = 124
    assert node._geometry_frame_pose(node.safety_clock.now()) is None
    node.corner_capture_stamp = 123
    node.parameters['simulation_yolo_white'] = False
    assert node._geometry_frame_pose(node.safety_clock.now()) is None


def test_cropped_s_bend_keeps_connected_centre_when_only_one_corner_is_visible(controller, monkeypatch):
    import pinky_move.metric_lane as lane
    from test_connected_path import chain
    node = controller
    node.parameters['simulation_yolo_white'] = True
    node.robot_calibration = {}
    node.corner_policy.s_route = {'test':True}
    curve = chain(np.array([[.14,-.1],[.5,-.1],[.5,.4]]))
    obs = dict(curve=curve[:20], side='right', width=.2)
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *args:[curve])
    _, target = node._hybrid_connected_observation([np.ones((20,20))],obs,None)
    assert target is not None and abs(target['y_m']) < .01
    np.testing.assert_allclose(np.asarray(target['center_path'])[-1],[.4,.4],atol=.01)


@pytest.mark.parametrize('case', ['valid','unconfirmed','wrong_side','wrong_curve','no_pose','no_obs','other_error','multi_mask'])
@pytest.mark.parametrize('error', ['inferred target discontinuity','no forward centre path'])
def test_target_jump_separation_preserves_all_corner_guards(controller, case, error):
    node = controller
    node.safety_clock.seconds = 1.6
    node.robot_calibration = {'test':True}
    node.parameters['corner_enabled'] = True
    policy, obs, _ = prepared()
    if case == 'unconfirmed':
        policy = CornerPolicy(CornerConfig(staged_turn=True,clearance_m=0.))
    if case == 'wrong_side': obs = dict(obs,side='left')
    if case == 'wrong_curve': obs = dict(obs,curve=obs['curve']+[0.,.1])
    if case == 'no_obs': obs = None
    node.corner_policy = policy
    node._corner_pose_for_frame = lambda *a: None if case=='no_pose' else np.zeros(3)
    def reject(*args, **kwargs):
        raise ValueError('visible boundary identity ambiguous' if case=='other_error'
                         else error)
    node.metric_tracker = SimpleNamespace(update=reject,last_observation=obs)
    masks = [np.zeros((20,20),np.uint8)]*(2 if case=='multi_mask' else 1)
    count = node._update_lane_command(masks,20,node.safety_clock.now())
    if case == 'valid':
        assert count == 1 and node.metric_target['corner_staged']
        assert node.corner_policy.debug['ordinary_target_ignored']
    else:
        assert count == 0 and node.metric_target is None


def test_hybrid_staged_skeleton_replaces_incorrect_axis_observation(controller):
    node = controller
    node.safety_clock.seconds = 1.6
    node.robot_calibration = {'test': True}
    node.parameters.update(corner_enabled=True, simulation_yolo_white=True)
    policy, measured, _ = prepared()
    node.corner_policy = policy
    node._corner_pose_for_frame = lambda *args: np.zeros(3)
    def reject(*args, **kwargs):
        raise ValueError('center_path curve folds back')
    wrong_axis = dict(measured, curve=measured['curve']+[0., .1])
    node.metric_tracker = SimpleNamespace(update=reject, last_observation=wrong_axis)
    policy.staged_observation = lambda *args: measured
    assert node._update_lane_command([np.zeros((20,20),np.uint8)], 20,
                                     node.safety_clock.now()) == 1
    assert node.metric_target['corner_staged']


def test_hybrid_forward_target_failure_keeps_current_corner_evidence(controller):
    node = controller
    node.robot_calibration = {'test': True}
    node.parameters.update(corner_enabled=True, simulation_yolo_white=True)
    node.corner_policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    node._corner_pose_for_frame = lambda *args: np.zeros(3)
    def reject(*args, **kwargs):
        raise ValueError('no forward centre path')
    node.metric_tracker = SimpleNamespace(update=reject, last_observation=observation())
    node.corner_policy.staged_observation = lambda *args: None
    masks = [np.zeros((20,20),np.uint8)]
    for now in (1., 1.2):
        node.safety_clock.seconds = now
        assert node._update_lane_command(masks, 20, node.safety_clock.now()) == 0
    node.safety_clock.seconds = 1.4
    assert node._update_lane_command(masks, 20, node.safety_clock.now()) == 1
    assert node.metric_target['corner_staged']
