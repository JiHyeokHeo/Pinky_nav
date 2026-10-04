"""End-to-end planning without ROS nodes, sockets, camera or motors."""
from copy import deepcopy
from types import SimpleNamespace
import json
import cv2
import numpy as np
import pytest
from pinky_move.lane_planning_remote import PCPlanner, validate_plan, plain
from pinky_move.lane_wire import ClassIds
from pinky_move import lane_wire
from pinky_move.lane_inference_worker import predict_reply
from std_msgs.msg import Header
from test_lane_controller import controller
from test_metric_lane import calibration, masks_for_lines
from test_staged_corner import prepared


def scene(node, seq=1):
    masks = masks_for_lines()
    polygons = [max(cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                     cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea).reshape(-1, 2) for m in masks]
    frame = np.zeros((*masks[0].shape, 3), np.uint8)
    result = SimpleNamespace(boxes=SimpleNamespace(cls=ClassIds([1]*len(masks))),
                             masks=SimpleNamespace(xy=polygons))
    params = dict(node.parameters, remote_geometry=True, corner_enabled=True)
    request = dict(token=f'session:{seq}', capture_sec=10, capture_nanosec=seq*100_000_000,
                   frame_id='camera', planning=dict(generation=1, enabled=True,
                   now_ns=10_000_000_000+seq*100_000_000, pose=[0.,0.,0.],
                   calibration=calibration(), parameters=params, staged=None))
    return request, result, frame


def test_pc_geometry_matches_legacy_target_and_robot_does_not_fit(controller):
    planner = PCPlanner()
    for seq in (1, 2, 3):
        request, result, frame = scene(controller, seq)
        plan = planner.process(request, result, frame)
        controller.parameters.update(request['planning']['parameters'], remote_geometry=False)
        controller.robot_calibration = request['planning']['calibration']
        controller._corner_pose_for_frame = lambda _: np.zeros(3)
        controller.safety_clock.seconds = 10.+seq*.1
        header = Header()
        header.stamp.sec, header.stamp.nanosec = request['capture_sec'], request['capture_nanosec']
        controller._process_result(frame, result, controller.safety_clock.now(), header)
        if plan['target'] is not None:
            for key in ('x_m', 'y_m', 'width_m'):
                assert plan['target'][key] == pytest.approx(controller.metric_target[key])
    validate_plan(plan, request['token'], 1, request['planning']['now_ns'])
    assert plan['target'] is not None
    assert not hasattr(planner.node, 'cmd_publisher')
    assert not hasattr(planner.node, '_control_loop')
    controller.remote_plan_context = request['planning']
    controller.corner_odom = (10.3, np.zeros(3))
    controller._process_planned_result(plan, controller.safety_clock.now())
    assert controller.metric_target['remote_world_target'] == pytest.approx(
        [plan['target']['x_m'], plan['target']['y_m']])
    assert controller.latest_linear > 0
    with pytest.raises(ValueError, match='out-of-order'):
        planner.process(request, result, frame)


def test_robot_process_bypasses_geometry_and_reuses_debug_observation(controller):
    request, result, frame = scene(controller)
    plan = PCPlanner().process(request, result, frame)
    controller.parameters.update(remote_geometry=True, publish_debug_image=False)
    controller.remote_inference = True
    controller.remote_plan_context = request['planning']
    controller._corner_pose_for_frame = lambda _: np.zeros(3)
    controller._update_lane_command = lambda *args: pytest.fail('geometry executed on robot')
    controller.image_identity.match = lambda *args: pytest.fail('image association executed on robot')
    result.remote_plan = plan
    header = Header()
    header.stamp.sec, header.stamp.nanosec = request['capture_sec'], request['capture_nanosec']
    controller._process_result(frame, result, controller.safety_clock.now(), header)
    assert controller.metric_target is not None
    assert controller.commands == []  # Only control timer may publish.


def test_wire_roundtrip_with_actual_worker_and_calibration(controller):
    request, result, frame = scene(controller)
    header = Header()
    header.stamp.sec, header.stamp.nanosec = request['capture_sec'], request['capture_nanosec']
    wire = lane_wire.unpack(lane_wire.encode_request(frame, request['token'], header, 'hash'))
    wire['planning'] = request['planning']
    model = SimpleNamespace(names={0: 'crossline', 1: 'lane'}, predict=lambda *a, **kw: [result])
    data = lane_wire.unpack(predict_reply(model, 'hash', wire, planner=PCPlanner()))
    assert not data.get('error'), data.get('error')
    plan = validate_plan(data['plan'], wire['token'], 1, request['planning']['now_ns'])
    assert plan['target'] is not None
    assert 'linear_velocity' not in data and 'angular_velocity' not in data


def test_remote_disconnect_and_stale_odom_stop(controller):
    controller.parameters.update(remote_geometry=True, use_lidar_guard=False)
    controller.metric_target = dict(x_m=.2, y_m=0., boundary_count=2, remote_world_target=[.2,0.])
    controller._calibration_stop_reason = lambda: None
    controller.inference_error = None
    controller.last_result_input_time = controller.safety_clock.now()
    controller.corner_odom = None
    assert controller._safety_stop_reason(controller.safety_clock.now()) == 'corner odometry timeout'
    controller.corner_odom = (10., np.zeros(3))
    controller.inference_error = 'remote inference timeout'
    assert controller._safety_stop_reason(controller.safety_clock.now()) == 'inference failed'


@pytest.mark.parametrize('case', ['token', 'generation', 'capture', 'nan', 'future'])
def test_rejected_plan_cannot_refresh_motion(controller, case):
    request, result, frame = scene(controller)
    plan = PCPlanner().process(request, result, frame)
    if case == 'token': plan['token'] = 'old:1'
    if case == 'generation': plan['generation'] = 0
    if case == 'capture': plan['capture_ns'] = 0
    if case == 'nan': plan['debug']['bad'] = float('nan')
    if case == 'future': plan['times']['last_lane_time'] = request['planning']['now_ns']+1
    controller.remote_plan_context = request['planning']
    with pytest.raises(ValueError):
        validate_plan(plan, request['token'], 1, request['planning']['now_ns'])
        controller._process_planned_result(plan, controller.safety_clock.now())
    assert controller.commands == []


def test_inflight_plan_cannot_undo_local_arrival(controller):
    policy, obs, target = prepared()
    controller.corner_policy = policy
    request, result, frame = scene(controller)
    request['planning']['staged'] = plain(policy.staged)
    plan = PCPlanner().process(request, result, frame)
    # Simulate a reply still carrying pre-arrival state, then local odom arrival.
    plan['staged'] = plain(policy.staged)
    plan['target'] = plain(target)
    controller.remote_plan_context = request['planning']
    policy.staged['brake_at'] = 1.6
    controller._process_planned_result(plan, controller.safety_clock.now())
    assert controller.corner_policy.staged['brake_at'] == 1.6
    changed = deepcopy(plan)
    changed['staged']['pivot'][0] += .05
    with pytest.raises(ValueError, match='move committed'):
        controller._process_planned_result(changed, controller.safety_clock.now())
