"""같은 경로 추종기로 YOLO semantic 입력과 OpenCV 입력을 비교한다."""
import numpy as np
import pytest
from pinky_move.white_lane import semantic_center_path
from test_lane_controller import controller


def test_empty_semantic_input_cannot_create_white_lane(monkeypatch):
    import pinky_move.metric_lane as metric
    monkeypatch.setattr(metric, 'connected_floor_curve', lambda *a, **k: pytest.fail('no input'))
    with pytest.raises(ValueError, match='no near white boundary'):
        semantic_center_path([], {}, .2)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_semantic_single_line_uses_normal_offset(monkeypatch, side):
    import pinky_move.metric_lane as metric
    sign = 1 if side == 'left' else -1
    curve = np.column_stack((np.linspace(.1, .8, 100), np.full(100, sign*.1)))
    mask = np.ones((10, 10), np.uint8)
    monkeypatch.setattr(metric, 'connected_floor_curve', lambda m, *a, **k: [curve])
    path, selected, observed_side = semantic_center_path([mask], {}, .2)
    assert observed_side == side and len(selected) == 1
    np.testing.assert_allclose(path[:, 1], 0., atol=1e-7)


def test_semantic_profile_cannot_run_without_hybrid(controller):
    controller.parameters['simulation_semantic_path'] = True
    with pytest.raises(ValueError, match='requires YOLO/white'):
        controller._validate_yolo_white_profile()


def test_lateral_semantic_stripe_uses_current_axis_fallback(monkeypatch):
    import pinky_move.metric_lane as metric
    curve = np.column_stack((np.linspace(.1, .8, 100), np.full(100, -.1)))
    monkeypatch.setattr(metric, 'connected_floor_curve', lambda *a, **k: [])
    monkeypatch.setattr(metric, 'floor_curves', lambda *a, **k: [curve])
    path, selected, side = semantic_center_path([np.ones((10,10))], {}, .2)
    assert side == 'right' and len(selected) == 1
    np.testing.assert_allclose(path[:,1], 0., atol=1e-7)


def test_semantic_target_does_not_enter_staged_policy(controller):
    controller.parameters['simulation_semantic_path'] = True
    controller.sim_white_path = np.array([[.1, 0.], [.5, 0.], [.5, .4]])
    controller.sim_white_side = 'right'
    controller.corner_policy.update = lambda *a, **k: pytest.fail('staged policy bypass')
    assert controller._update_lane_command([np.ones((10,10))], 10, controller.safety_clock.now()) == 1
    assert controller.metric_target['white_path']
