"""YOLO 흰 픽셀 보충의 semantic gate/신선도/흰 바닥 거부 검사."""
from types import SimpleNamespace
import numpy as np
import pytest
from pinky_move.yolo_white import YoloWhiteSupplement
from test_lane_controller import controller


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
