"""YOLO 흰 픽셀 보충의 semantic gate/신선도/흰 바닥 거부 검사."""
from types import SimpleNamespace
import numpy as np
import pytest
from pinky_move.yolo_white import YoloWhiteSupplement
from test_lane_controller import controller
from test_lane_controller import detection


def scene():
    frame = np.full((100, 200, 3), 40, np.uint8)
    white = np.zeros((100, 200), np.uint8)
    white[55:99, 30:35] = 1
    frame[white > 0] = 255
    seed = white.copy()
    seed[:80] = 0
    return frame, white, seed


def test_white_cannot_start_without_yolo():
    frame, white, _ = scene()
    masks, info = YoloWhiteSupplement().update(frame, [], 1., {}, [white])
    assert masks == [] and info['white_tracked'] == 0


def test_current_yolo_seed_completes_same_white_component():
    frame, white, seed = scene()
    masks, info = YoloWhiteSupplement().update(frame, [seed], 1., {}, [white])
    assert len(masks) == 1 and info['white_current'] == 1
    np.testing.assert_array_equal(masks[0], white)


def test_short_dropout_uses_current_pixels_not_old_mask():
    frame, white, seed = scene()
    helper = YoloWhiteSupplement()
    helper.update(frame, [seed], 1., {}, [white])
    masks, info = helper.update(frame, [], 1.2, {}, [white])
    assert len(masks) == 1 and info['white_tracked'] == 1
    masks, _ = helper.update(frame, [], 1.4, {}, [])
    assert masks == []


@pytest.mark.parametrize('now', [.9, 1., 2.])
def test_old_duplicate_or_backward_frame_does_not_reuse_history(now):
    frame, white, seed = scene()
    helper = YoloWhiteSupplement()
    helper.update(frame, [seed], 1., {}, [white])
    masks, _ = helper.update(frame, [], now, {}, [white])
    assert masks == []


def test_semantic_confirmation_expires_even_if_white_stays_visible():
    frame, white, seed = scene()
    helper = YoloWhiteSupplement(semantic_timeout_s=.3)
    helper.update(frame, [seed], 1., {}, [white])
    assert helper.update(frame, [], 1.2, {}, [white])[0]
    assert helper.update(frame, [], 1.4, {}, [white])[0] == []


def test_white_floor_is_not_accepted_as_lane():
    frame = np.full((100, 200, 3), 255, np.uint8)
    seed = np.zeros((100, 200), np.uint8)
    seed[80:90, 10:15] = 1
    masks, info = YoloWhiteSupplement().update(frame, [seed], 1., {}, [np.ones((100,200),np.uint8)])
    assert info['white_current'] == 0
    np.testing.assert_array_equal(masks[0], seed)


def test_unrelated_white_stripe_is_not_supplemented():
    frame, white, seed = scene()
    unrelated = np.roll(white, 110, axis=1)
    masks, info = YoloWhiteSupplement().update(frame, [seed], 1., {}, [white, unrelated])
    assert len(masks) == 1 and info['white_current'] == 1
    np.testing.assert_array_equal(masks[0], white)


def test_distant_tracked_fragment_cannot_block_current_control_interval(monkeypatch):
    import pinky_move.robot_projection as projection
    frame, white, seed = scene()
    distant = np.roll(white, 110, axis=1)
    monkeypatch.setattr(projection, 'robot_floor_points', lambda pixels, *args:
        np.column_stack((np.where(pixels[:, 0] < 100, .22, .56), np.zeros(len(pixels)))))
    helper = YoloWhiteSupplement(support_interval=(.14, .48))
    helper.update(frame, [seed, distant], 1., {}, [white, distant])
    masks, info = helper.update(frame, [seed], 1.2, {}, [white, distant])
    assert len(masks) == 1 and info['unsupported_white'] == 1
    np.testing.assert_array_equal(masks[0], white)


def test_supported_long_component_keeps_far_exit_pixels(monkeypatch):
    import pinky_move.robot_projection as projection
    frame, white, seed = scene()
    monkeypatch.setattr(projection, 'robot_floor_points', lambda pixels, *args:
        np.column_stack((np.where(pixels[:, 1] > 80, .22, .80), np.zeros(len(pixels)))))
    masks, _ = YoloWhiteSupplement(support_interval=(.14, .48)).update(
        frame, [seed], 1., {}, [white])
    np.testing.assert_array_equal(masks[0], white)


def test_one_yolo_instance_cannot_create_multiple_white_boundaries():
    frame, white, _ = scene()
    other = np.roll(white, 110, axis=1)
    model = white.copy()
    model[55:68] |= other[55:68]
    masks, info = YoloWhiteSupplement().update(frame, [model], 1., {}, [white, other])
    assert len(masks) == 1 and info['white_current'] == 1 and info['retained_yolo'] == 0
    np.testing.assert_array_equal(masks[0], white)


def test_one_history_track_cannot_duplicate_into_two_current_fragments():
    frame, white, seed = scene()
    helper = YoloWhiteSupplement()
    helper.update(frame, [seed], 1., {}, [white])
    large, small = white.copy(), white.copy()
    large[:69] = 0
    small[69:] = 0
    masks, info = helper.update(frame, [], 1.2, {}, [large, small])
    assert len(masks) == 1 and info['white_tracked'] == 1
    np.testing.assert_array_equal(masks[0], large)


def test_stage_white_requires_committed_corner_and_verified_current_pixels(controller, monkeypatch):
    import pinky_move.white_lane as white_lane
    from test_staged_corner import prepared
    frame, white, _ = scene()
    controller.parameters['simulation_yolo_white'] = True
    controller.robot_calibration = {'method':'intrinsics_urdf_floor'}
    controller._corner_pose_for_frame = lambda *args: np.zeros(3)
    assert controller._stage_white_mask(frame, controller.safety_clock.now()) is None
    controller.corner_policy, _, _ = prepared()
    monkeypatch.setattr(white_lane, 'white_floor_masks', lambda *args: [white])
    controller.corner_policy.staged_observation = lambda *args: None
    assert controller._stage_white_mask(frame, controller.safety_clock.now()) is None
    controller.corner_policy.staged_observation = lambda *args: {'source_mask_index':0}
    np.testing.assert_array_equal(controller._stage_white_mask(frame, controller.safety_clock.now()), white)
    controller.parameters['simulation_yolo_white'] = False
    assert controller._stage_white_mask(frame, controller.safety_clock.now()) is None


def test_current_white_map_reassociation_requires_recent_model_and_unique_real_role(controller, monkeypatch):
    import pinky_move.white_lane as white_lane
    import pinky_move.lane_autonomy as autonomy
    frame, white, _ = scene()
    controller.parameters['simulation_yolo_white'] = True
    controller.robot_calibration = {'method':'intrinsics_urdf_floor'}
    controller._corner_pose_for_frame = lambda *args: np.zeros(3)
    controller.hybrid_last_semantic_time = 9.
    monkeypatch.setattr(white_lane,'white_floor_masks',lambda *args:[white])
    monkeypatch.setattr(autonomy,'connected_floor_curve',lambda *args:[np.column_stack((np.linspace(.14,.4,20),np.full(20,.1)))])
    controller.corner_policy.local_lane_map.measured_side = lambda *args: 'left'
    np.testing.assert_array_equal(controller._stage_white_mask(frame,controller.safety_clock.now()),white)
    assert controller.hybrid_white_map_side == 'left'
    controller.corner_policy.local_lane_map.measured_side = lambda *args: None
    assert controller._stage_white_mask(frame,controller.safety_clock.now()) is None
    controller.corner_policy.local_lane_map.measured_side = lambda *args: 'left'
    monkeypatch.setattr(white_lane,'white_floor_masks',lambda *args:[white,white])
    assert controller._stage_white_mask(frame,controller.safety_clock.now()) is None
    monkeypatch.setattr(white_lane,'white_floor_masks',lambda *args:[white])
    controller.hybrid_last_semantic_time = -1.
    assert controller._stage_white_mask(frame,controller.safety_clock.now()) is None


def test_known_white_map_never_overwrites_present_yolo_boundaries(controller):
    from std_msgs.msg import Header
    controller.parameters['simulation_yolo_white'] = True
    controller._corner_pose_for_frame = lambda *args: np.zeros(3)
    controller._update_lane_command = lambda masks,*args: len(masks)
    def forbidden(*args):
        raise AssertionError('map restoration overwrote present YOLO boundaries')
    controller._stage_white_mask = forbidden
    result = detection()
    result.supplement = {'yolo':2}
    controller._process_result(np.zeros((100,200,3),np.uint8),result,
                               controller.safety_clock.now(),Header())
    assert controller.lane_instances == 2


@pytest.mark.parametrize('mode', ['remote_inference', 'remote_geometry', 'simulation_white_lane'])
def test_remote_or_white_only_cannot_bypass_hybrid_guard(controller, mode):
    controller.parameters.update(simulation_yolo_white=True, image_topic='/lane_sim/camera/image_raw')
    controller.parameters[mode] = True
    controller.context = SimpleNamespace(get_domain_id=lambda: 172)
    controller.robot_calibration = {'status': 'simulation_only_ideal_camera'}
    with pytest.raises(ValueError, match='only local YOLO'):
        controller._validate_yolo_white_profile()


@pytest.mark.parametrize('domain', [20, 22, 52])
def test_hybrid_cannot_run_in_actual_robot_domain(controller, domain):
    controller.parameters['simulation_yolo_white'] = True
    controller.context = SimpleNamespace(get_domain_id=lambda: domain)
    controller.robot_calibration = {'status': 'simulation_only_ideal_camera'}
    controller.parameters['image_topic'] = '/lane_sim/camera/image_raw'
    with pytest.raises(ValueError, match='isolated simulation'):
        controller._load_model('unused')
