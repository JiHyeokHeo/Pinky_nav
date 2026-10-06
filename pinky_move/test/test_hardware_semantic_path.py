"""실차 PC planning → wire → 로봇 명령 검증; 모터/SSH 접속 없음."""
from copy import deepcopy
import json
import numpy as np
import pytest
from test_lane_controller import controller
from test_remote_planning import scene
from pinky_move.lane_planning_remote import PCPlanner, validate_plan
from pinky_move.lane_inference_worker import predict_reply
from pinky_move import lane_wire
from std_msgs.msg import Header
from types import SimpleNamespace


def semantic_scene(controller):
    controller.parameters.update(semantic_lane_following=True, remote_inference=True,
        remote_geometry=True, metric_path_min_m=.08, metric_path_max_m=.70,
        metric_lookahead_m=.16, maximum_angular_speed=.6, single_line_max_speed=.06)
    return scene(controller)


def test_hardware_pc_path_uses_same_algorithm_and_no_staged_policy(controller):
    request, result, frame = semantic_scene(controller)
    planner = PCPlanner()
    plan = planner.process(request, result, frame)
    assert plan['target'] is not None, plan['error']
    assert plan['target']['white_path'] and plan['staged'] is None
    assert plan['debug']['mode'] == 'SEMANTIC_PATH'
    assert not planner.node._bool_parameter('simulation_white_lane')
    assert not hasattr(planner.node, 'cmd_publisher')
    assert planner.perception_instances
    validate_plan(plan, request['token'], 1, request['planning']['now_ns'])
    controller.remote_plan_context = request['planning']
    controller._process_planned_result(plan, controller.safety_clock.now())
    assert controller.perception_source == 'YOLO_WHITE_CONNECTED_PC'
    assert controller.metric_target['white_path']
    assert controller.commands == []


def test_semantic_wire_preserves_crossline_and_selected_label_order(controller):
    request, result, frame = semantic_scene(controller)
    from test_lane_controller import rectangle
    result.boxes.cls.append(0)
    result.masks.xy.append(rectangle(20,60,180,65))
    header = Header()
    header.stamp.sec, header.stamp.nanosec = request['capture_sec'], request['capture_nanosec']
    wire = lane_wire.unpack(lane_wire.encode_request(frame,request['token'],header,'hash'))
    wire['planning'] = request['planning']
    model = SimpleNamespace(names={0:'crossline',1:'lane'},predict=lambda *a,**kw:[result])
    reply = json.loads(predict_reply(model,'hash',wire,retry_imgsz=0,planner=PCPlanner()))
    assert not reply.get('error'), reply.get('error')
    assert reply['instances'][-1]['class'] == 'crossline'
    count = sum(item['class']=='lane' for item in reply['instances'])
    assert len(reply['plan']['labels']) == count
    assert reply['plan']['target']['boundary_count'] == count


def test_zero_yolo_cold_start_does_not_follow_white_floor(controller):
    request, result, frame = semantic_scene(controller)
    result.boxes.cls.clear(); result.masks.xy.clear()
    frame[:] = 255
    plan = PCPlanner().process(request,result,frame)
    assert plan['target'] is None
    assert plan['debug']['supplement']['yolo'] == 0


def test_semantic_mode_does_not_spend_frame_ttl_on_640_retry(controller):
    request, result, frame = semantic_scene(controller)
    result.boxes.cls.clear(); result.masks.xy.clear()
    header = Header()
    header.stamp.sec, header.stamp.nanosec = request['capture_sec'], request['capture_nanosec']
    wire = lane_wire.unpack(lane_wire.encode_request(frame,request['token'],header,'hash'))
    wire['planning'] = request['planning']
    calls = []
    def predict(*a,**kw):
        calls.append(kw['imgsz'])
        return [result]
    model = SimpleNamespace(names={0:'crossline',1:'lane'}, predict=predict)
    reply = json.loads(predict_reply(model,'hash',wire,retry_imgsz=640,planner=PCPlanner()))
    assert not reply.get('error'), reply.get('error')
    assert calls == [320]
    assert reply['plan']['target'] is None


def test_legacy_robot_rejects_semantic_reply(controller):
    request, result, frame = semantic_scene(controller)
    plan = PCPlanner().process(request,result,frame)
    controller.parameters['semantic_lane_following'] = False
    controller.remote_plan_context = request['planning']
    with pytest.raises(ValueError,match='legacy profile'):
        controller._process_planned_result(plan,controller.safety_clock.now())


@pytest.mark.parametrize('bad', [None, [[0.,0.]], [[float('inf'),0.],[.2,0.]]])
def test_malformed_semantic_path_rejected_before_commands(controller,bad):
    request, result, frame = semantic_scene(controller)
    plan = PCPlanner().process(request,result,frame)
    plan['target']['center_path'] = bad
    with pytest.raises((ValueError,TypeError)):
        validate_plan(plan,request['token'],1,request['planning']['now_ns'])
    assert controller.commands == []


def test_real_semantic_profile_requires_remote_geometry(controller):
    controller.parameters['semantic_lane_following'] = True
    with pytest.raises(ValueError,match='PC inference and geometry'):
        controller._validate_yolo_white_profile()


def test_transverse_target_can_rotate_instead_of_remote_forward_rejection(controller):
    controller.parameters.update(semantic_lane_following=True, remote_geometry=True,
                                 maximum_angular_speed=.6)
    controller.metric_target = dict(x_m=0.,y_m=.15,white_path=True,
        remote_world_target=[0.,.15],boundary_count=1,inferred=True)
    controller.corner_odom = (10.,np.zeros(3))
    controller.last_lane_time = controller.safety_clock.now()
    controller.boundary_count = 1
    controller._safety_stop_reason = lambda _: None
    controller._control_loop()
    assert controller.commands[-1].linear.x == 0.
    assert controller.commands[-1].angular.z > 0.


def test_transverse_rotation_still_stops_on_stale_input(controller):
    from rclpy.time import Time
    from rclpy.clock import ClockType
    controller.parameters.update(semantic_lane_following=True, remote_geometry=True,
                                 result_timeout=.8)
    controller.metric_target = dict(x_m=0.,y_m=.15,white_path=True,
        remote_world_target=[0.,.15],boundary_count=1,inferred=True)
    controller.corner_odom = (10.,np.zeros(3))
    controller.last_lane_time = controller.safety_clock.now()
    controller.last_result_input_time = Time(nanoseconds=9_000_000_000,clock_type=ClockType.STEADY_TIME)
    controller._calibration_stop_reason = lambda: None
    controller._control_loop()
    assert controller.commands[-1].linear.x == controller.commands[-1].angular.z == 0.
    assert 'inference input expired' in controller.last_status_text
