"""격리된 OpenCV 물리 시뮬 프로파일의 기하·도메인 회귀 검사."""
from types import SimpleNamespace
import numpy as np
import pytest
from pinky_move.white_lane import miter_center, white_center_path, white_target, white_floor_masks
from test_lane_controller import controller


def candidate(y):
    p = np.column_stack((np.linspace(.14, .65, 30), np.full(30, y)))
    return np.ones((10, 10), np.uint8), p, [p]


@pytest.mark.parametrize('sign', [-1, 1])
def test_normal_miter_preserves_right_angle_vertex(sign):
    p = np.array([[0., 0.], [.65, 0.], [.65, sign*.5]])
    centre = miter_center(p, sign*.1)
    assert np.allclose(centre[0], [0., sign*.1])
    assert np.allclose(centre[-1], [.55, sign*.5])
    assert np.max(centre[:, 0]) <= .55+1e-6


def test_three_stripes_choose_adjacent_pair_not_second_lane():
    path, selected, _ = white_center_path([candidate(-.1), candidate(.1), candidate(.3)], .2)
    assert set(selected) == {'left', 'right'}
    assert np.allclose(path[:, 1], 0., atol=1e-6)
    assert np.allclose(selected['left'][2][:, 1], .1)


@pytest.mark.parametrize('y', [-.1, .1])
def test_one_boundary_produces_virtual_centre(y):
    path, selected, side = white_center_path([candidate(y)], .2)
    target = white_target(path, .16, len(selected), side)
    assert target['inferred']
    assert abs(target['y_m']) < 1e-6
    assert target['x_m'] > .14


def test_lateral_exit_is_not_rejected_for_small_forward_x():
    target = white_target([[.01, .02], [.01, .1], [.01, .4]], .16, 1, 'right')
    assert target['x_m'] == .01
    assert target['y_m'] > .1


def test_path_exhaustion_does_not_extrapolate():
    with pytest.raises(ValueError, match='exhausted'):
        white_target([[.4, 0.], [.1, 0.]], .16, 1, 'left')


@pytest.mark.parametrize('domain', [20, 22, 52])
def test_white_mode_rejects_hardware_and_control_domains(controller, domain):
    controller.parameters['simulation_white_lane'] = True
    controller.parameters['image_topic'] = '/lane_sim/camera/image_raw'
    controller.robot_calibration = {'status': 'simulation_only_ideal_camera'}
    controller.context = SimpleNamespace(get_domain_id=lambda: domain)
    with pytest.raises(ValueError, match='isolated'):
        controller._load_model('unused')


def test_white_mode_skips_yolo_only_in_simulator(controller):
    controller.parameters.update(simulation_white_lane=True, image_topic='/lane_sim/camera/image_raw')
    controller.robot_calibration = {'status': 'simulation_only_ideal_camera'}
    controller.context = SimpleNamespace(get_domain_id=lambda: 172)
    controller._load_model('nonexistent-model')
    assert controller.model is None


def test_actual_default_remains_yolo(controller):
    assert not controller._bool_parameter('simulation_white_lane')


@pytest.mark.parametrize('colour', [(45, 45, 45), (0, 0, 255)])
def test_dark_floor_and_saturated_red_are_not_white_paint(colour):
    frame = np.full((480, 640, 3), colour, np.uint8)
    assert white_floor_masks(frame, {}) == []


@pytest.mark.parametrize('invalid', ['camera', 'calibration'])
def test_white_mode_rejects_actual_camera_or_mounting(controller, invalid):
    controller.parameters.update(simulation_white_lane=True, image_topic='/lane_sim/camera/image_raw')
    controller.robot_calibration = {'status': 'simulation_only_ideal_camera'}
    controller.context = SimpleNamespace(get_domain_id=lambda: 172)
    if invalid == 'camera':
        controller.parameters['image_topic'] = '/pinky/camera/image_raw/compressed'
    else:
        controller.robot_calibration['status'] = 'unverified_mounting'
    with pytest.raises(ValueError, match='isolated'):
        controller._load_model('unused')
