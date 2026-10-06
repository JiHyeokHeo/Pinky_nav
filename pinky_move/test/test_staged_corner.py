"""Motor-free ENTRY -> BRAKE -> PIVOT_TURN tests in odometry coordinates."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerConfig, CornerPolicy, staged_command
from test_lane_corner import observation, ordinary
from test_lane_controller import controller
from pinky_move.lane_autonomy import DriveState


def local_observation(obs, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    inverse = np.array([[c, s], [-s, c]])
    return dict(obs, curve=(obs['curve']-pose[:2])@inverse.T)


def prepared(sign=1):
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation(sign)
    result = None
    for now in (1., 1.2, 1.4):
        result = policy.update(obs, ordinary(), now, np.zeros(3), .22, .25)
    assert result['corner_staged'] and not result.get('corner_stationary')
    return policy, obs, result


@pytest.mark.parametrize('sign',[1,-1])
def test_verified_first_sharp_s_corner_replaces_route_only_after_confirmation(sign):
    policy=CornerPolicy(CornerConfig(staged_turn=True,clearance_m=0.))
    obs=dict(observation(sign),sharp_s_prefix=True)
    route={'side':obs['side'],'exit_world':np.array([.9,.4])}
    policy.s_route=route
    # 오래된 S 분류 히스토리가 새로 확인된 직각을 다시 S로 덮지 않는다.
    policy.s_bend_anchor={'time':1.,'side':obs['side'],'width':.4,
                         'curve':obs['curve'].copy()}
    for now in (1.,1.2):
        policy.update(obs,ordinary(),now,np.zeros(3),.22,.25)
        assert policy.s_route is route and policy.staged is None
        assert policy.debug['mode']!='S_BEND'
    result=policy.update(obs,ordinary(),1.4,np.zeros(3),.22,.25)
    assert result['corner_staged'] and policy.staged['sharp_s_prefix']
    assert policy.s_route is None and policy.s_bend_anchor is None


@pytest.mark.parametrize('sign', [1, -1])
def test_entry_then_stationary_turn_then_near_exit(sign):
    policy, obs, target = prepared(sign)
    pivot = np.array(target['corner_pivot_world'])
    assert pivot[0] == pytest.approx(.3, abs=.002)
    assert abs(pivot[1]) < .002
    v, w, _ = staged_command(target, np.zeros(3), 1.4, .25, .025)
    assert 0 < v <= .03 and abs(w) < .01
    pose = np.r_[pivot, 0.]
    # At 20 Hz, entry stops even before another inference arrives.
    assert staged_command(target, pose, 1.5, .25, .025)[:2] == (0., 0.)
    target = policy.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    assert target['corner_stationary']
    assert staged_command(target, pose, 1.7, .25, .025)[:2] == (0., 0.)
    v, w, _ = staged_command(target, pose, 2., .25, .025)
    assert v == 0 and sign*w > 0 and abs(w) <= .25
    pose[2] = sign*np.pi/2
    assert staged_command(target, pose, 2.1, .25, .025)[:2] == (0., 0.)
    # A distant path must NOT release the turn.
    far = dict(ordinary(), x_m=.5)
    target = policy.update(local_observation(obs, pose), far, 2.2, pose, .22, .25)
    assert target['corner_stationary']
    for now in (2.4, 2.6, 2.8):
        target = policy.update(local_observation(obs, pose), ordinary(), now, pose, .22, .25)
    assert not target.get('corner_staged') and policy.staged is None


def test_staged_loss_wrong_identity_and_timeout_do_not_drive():
    p, obs, target = prepared()
    assert p.update(None, None, 1.6, np.zeros(3), .22, .25) is None
    assert p.update(dict(obs, side='left'), None, 1.8, np.zeros(3), .22, .25) is None
    assert p.update(obs, None, 2., None, .22, .25) is None
    assert p.update(obs, None, 22., np.zeros(3), .22, .25) is None
    assert staged_command(target, np.zeros(3), 22., .25, .025)[:2] == (0., 0.)


def test_spin_deadline_and_drift_stop_at_control_rate():
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    target = p.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    assert staged_command(target, pose, target['corner_spin_deadline'], .25, .025)[:2] == (0., 0.)
    displaced = pose+np.array([.05, 0., 0.])
    assert staged_command(target, displaced, 2., .25, .025)[:2] == (0., 0.)


def test_short_observed_entry_allows_supported_pivot_but_not_extrapolation():
    from pinky_move.lane_corner import classify_boundary
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = dict(observation(), width=.154)
    f = classify_boundary(obs['curve'], p.config)
    f = dict(f, entry=f['corner']-.089*f['tin'])
    p._start_staged(obs, f, np.zeros(3), 1.)
    assert p.staged is not None
    # Half-width offset would put the pivot before the observed entry.
    f = dict(f, entry=f['corner']-.06*f['tin'])
    with pytest.raises(ValueError, match='outside observed'):
        p._start_staged(obs, f, np.zeros(3), 1.)


def test_cropped_entry_uses_two_recent_measured_map_supports():
    from pinky_move.lane_corner import classify_boundary
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    full = dict(observation(), width=.20)
    feature = classify_boundary(full['curve'], p.config)
    cropped = dict(feature, entry=feature['corner']-.08*feature['tin'])
    p.local_lane_map.remember(full, .6, np.zeros(3))
    # One old line is not enough to invent a supported entry.
    assert p._mapped_entry_support(full, cropped, np.zeros(3), 1.) is cropped
    p.local_lane_map.remember(full, .8, np.zeros(3))
    recovered = p._mapped_entry_support(full, cropped, np.zeros(3), 1.)
    assert recovered['mapped_entry_support_m'] > .02
    np.testing.assert_array_equal(recovered['corner'], cropped['corner'])
    np.testing.assert_array_equal(recovered['tout'], cropped['tout'])
    p._start_staged(full, cropped, np.zeros(3), 1.)
    assert p.staged is not None
    assert p._mapped_entry_support(full, cropped, np.zeros(3), 3.) is not cropped
    assert p._mapped_entry_support(full, cropped, np.zeros(3), 11.3) is cropped


def test_staged_current_fragment_requires_unique_observed_match(monkeypatch):
    import pinky_move.metric_lane as lane
    p, obs, _ = prepared()
    monkeypatch.setattr(lane, 'connected_floor_curve', lambda *args: [obs['curve']])
    assert p.staged_observation([np.ones((20,20))], {}, np.zeros(3))['side'] == obs['side']
    assert p.staged_observation([np.ones((20,20)), np.ones((20,20))], {}, np.zeros(3)) is None
    monkeypatch.setattr(lane, 'connected_floor_curve', lambda *args: [obs['curve']+[0., .2]])
    assert p.staged_observation([np.ones((20,20))], {}, np.zeros(3)) is None


def test_newly_visible_long_exit_does_not_dilute_corner_correspondence():
    from pinky_move.lane_corner import trim_corner_endpoint_extensions, match_corner_fragment
    from test_lane_corner import boundary
    reference = boundary()
    current = np.vstack((reference, np.column_stack((np.full(300,.5), np.linspace(.205,2.,300)))))
    assert not match_corner_fragment(current, reference)['fragment_ok']
    clipped = trim_corner_endpoint_extensions(current, reference)
    assert match_corner_fragment(clipped, reference)['fragment_ok']
    assert np.max(clipped[:,1]) <= .205


def test_displaced_parallel_exit_is_not_an_endpoint_extension():
    from pinky_move.lane_corner import trim_corner_endpoint_extensions, match_corner_fragment
    from test_lane_corner import boundary
    reference = boundary()
    current = reference.copy()
    current[80:,0] += .1
    clipped = trim_corner_endpoint_extensions(current, reference)
    np.testing.assert_array_equal(clipped, current)
    assert not match_corner_fragment(clipped, reference)['fragment_ok']


def test_rejected_staging_keeps_valid_distant_ordinary_path(monkeypatch):
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    def reject(*args):
        raise ValueError('corner entry: pivot outside observed entry support')
    monkeypatch.setattr(p, '_start_staged', reject)
    for now in (1., 1.2, 1.4):
        target = p.update(observation(), ordinary(), now, np.zeros(3), .22, .25)
    assert target['x_m'] == ordinary()['x_m']
    assert not p.block_recovery and p.staged is None
    assert p.update(observation(), None, 1.6, np.zeros(3), .22, .25) is None
    assert p.block_recovery


def test_valid_centre_approaches_before_distant_corner_is_committed():
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    obs = observation()
    current = dict(ordinary(), center_path=np.column_stack((
        np.linspace(.14,.3,20), np.zeros(20))).tolist())
    for now in (1.,1.2,1.4):
        target = policy.update(obs, current, now, np.zeros(3), .22, .25)
        assert target is not None and not target.get('corner_staged')
    assert policy.staged is None and not policy.block_recovery
    # No valid centre remains: the same confirmed measured corner can commit.
    target = policy.update(obs, None, 1.6, np.zeros(3), .22, .25)
    assert target['corner_staged']


@pytest.mark.parametrize('sign', [-1, 1])
def test_inner_corner_miter_support_is_not_boundary_endpoint_extrapolation(sign):
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    entry = np.array([.14, sign*.1])
    corner = np.array([.3, sign*.1])
    exit_point = np.array([.3, sign*.45])
    curve = np.vstack((np.linspace(entry, corner, 40, endpoint=False),
                       np.linspace(corner, exit_point, 40)))
    obs = dict(curve=curve, side='left' if sign>0 else 'right', width=.2)
    feature = dict(entry=entry, corner=corner, exit=exit_point,
                   tin=np.array([1.,0.]), tout=np.array([0.,float(sign)]))
    p._start_staged(obs, feature, np.zeros(3), 1.)
    assert p.debug['entry_miter_extension_m'] == pytest.approx(.1)
    assert p.debug['entry_center_support_m'] == pytest.approx(.26)
    assert p.staged is not None
    # A displaced exit is not excused by the analytic lane-width extension.
    with pytest.raises(ValueError, match='outside observed'):
        p._start_staged(obs, dict(feature, exit=exit_point+[.04,0.]), np.zeros(3), 1.)
    with pytest.raises(ValueError, match='insufficient measured exit'):
        p._start_staged(obs, dict(feature, exit=corner+[0.,sign*.08]), np.zeros(3), 1.)


def test_inner_boundary_loss_hands_over_observation_without_moving_pivot(monkeypatch):
    import pinky_move.metric_lane as lane
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    entry, corner, end = np.array([.14,.1]), np.array([.3,.1]), np.array([.3,.45])
    curve = np.vstack((np.linspace(entry, corner, 40, endpoint=False), np.linspace(corner, end, 40)))
    obs = dict(curve=curve, side='left', width=.2)
    feature = dict(entry=entry, corner=corner, exit=end, tin=np.array([1.,0.]), tout=np.array([0.,1.]))
    p._start_staged(obs, feature, np.zeros(3), 1.)
    pivot = p.staged['pivot'].copy()
    other = p.staged['opposite_curve'].copy()
    assert other is not None
    monkeypatch.setattr(lane, 'connected_floor_curve', lambda *args: [other])
    current = p.staged_observation([np.ones((20,20))], {}, np.zeros(3))
    assert current['side'] == 'right' and current['staged_opposite_validated']
    target = p.update(current, None, 1.2, np.zeros(3), .22, .25)
    assert target['corner_staged'] and target['visible_side'] == 'right'
    np.testing.assert_array_equal(p.staged['pivot'], pivot)
    assert p.debug['entry_boundary_handover'] == 'inferred_width'
    monkeypatch.setattr(lane, 'connected_floor_curve', lambda *args: [other+[0., .05]])
    assert p.staged_observation([np.ones((20,20))], {}, np.zeros(3)) is None


def test_measured_opposite_entry_only_does_not_hide_later_real_exit(monkeypatch):
    import pinky_move.metric_lane as lane
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    entry, corner, end = np.array([.14,.1]), np.array([.3,.1]), np.array([.3,.45])
    curve = np.vstack((np.linspace(entry, corner, 40, endpoint=False), np.linspace(corner,end,40)))
    other = np.column_stack((np.linspace(.14,.48,40), np.full(40,-.1)))
    obs = dict(curve=curve, side='left', width=.2, other=other)
    feature = dict(entry=entry, corner=corner, exit=end, tin=np.array([1.,0.]), tout=np.array([0.,1.]))
    p._start_staged(obs, feature, np.zeros(3), 1.)
    assert p.staged['opposite_source'] == 'inferred_width'
    pivot = p.staged['pivot'].copy()
    # 새 출구의 실제 픽셀이 가리키는 횡방향 관측만 남은 상황.
    exit_curve = np.column_stack((np.full(40,.5), np.linspace(-.08,.1,40)))
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *args:[exit_curve])
    measured = p.staged_observation([np.ones((20,20))], {}, np.zeros(3))
    assert measured is not None and measured['staged_opposite_validated']
    assert p.update(measured, None, 1.2, np.zeros(3), .22, .25)['corner_staged']
    np.testing.assert_array_equal(p.staged['pivot'], pivot)


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('travel', [.001, .003, .0049])
def test_visible_entry_origin_is_not_robot_distance(sign, travel):
    p = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    # Analytic 90-degree measured outer boundary with a ~20 cm axle pivot.
    obs = dict(observation(sign), width=.214)
    feature = dict(entry=np.array([.307-.107-travel, -sign*.107]),
                   corner=np.array([.307, -sign*.107]),
                   exit=np.array([.307, sign*.15]),
                   tin=np.array([1., 0.]), tout=np.array([0., float(sign)]))
    p._start_staged(obs, feature, np.zeros(3), 1.)
    np.testing.assert_allclose(p.staged['pivot'], [.2, 0.], atol=1e-9)
    assert p.debug['entry_travel_m'] == pytest.approx(travel)
    # Still reject an intersection beyond the explicit 10mm endpoint grace.
    p.reset()
    feature['entry'][0] = .211
    with pytest.raises(ValueError, match='outside observed'):
        p._start_staged(obs, feature, np.zeros(3), 1.)


def test_controller_staged_spin_obeys_stale_frame_and_odom_stop(controller):
    p, obs, target = prepared()
    pose = np.r_[target['corner_pivot_world'], 0.]
    target = p.update(local_observation(obs, pose), None, 1.6, pose, .22, .25)
    controller.metric_target = target
    controller.corner_odom = (10., pose)
    controller.last_result_input_time = controller.safety_clock.now()
    controller.last_lane_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.commands[-1].linear.x == 0.
    assert controller.commands[-1].angular.z > 0.
    controller.safety_clock.seconds += .4
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].angular.z == 0.
    controller.corner_odom = (10.4, pose)
    controller.parameters['result_timeout'] = .2
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].linear.x == controller.commands[-1].angular.z == 0.


@pytest.mark.parametrize('values', [dict(entry_speed_mps=.1), dict(pivot_tolerance_m=.2),
                                    dict(brake_seconds=0.), dict(spin_timeout_s=21.)])
def test_invalid_staged_limits(values):
    with pytest.raises(ValueError):
        CornerConfig(**values).validate()
