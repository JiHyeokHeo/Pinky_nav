"""동일 관측으로 생성한 연결 miter가 출구를 보존하는지 검사한다."""
import numpy as np
import pytest
from pinky_move.connected_path import connected_miter_center
from test_lane_controller import controller


def chain(vertices):
    return np.vstack([np.linspace(a, b, 25, endpoint=False)
                      for a, b in zip(vertices[:-1], vertices[1:])] + [vertices[-1:]])


@pytest.mark.parametrize('sign', [-1, 1])
def test_connected_right_angle_preserves_offset_legs(sign):
    vertices = np.array([[.14, 0], [.4, 0], [.4, sign*.4]])
    _, center = connected_miter_center(chain(vertices), sign*.1)
    assert np.allclose(center[0], [.14, sign*.1], atol=1e-6)
    assert np.allclose(center[-1], [.3, sign*.4], atol=1e-6)
    assert np.max(center[:, 0]) <= .3+1e-6


def test_straight_normal_offset_is_exact():
    p = np.column_stack((np.linspace(.14, .48, 30), np.full(30, .1)))
    _, center = connected_miter_center(p, -.1)
    assert np.allclose(center[:, 1], 0, atol=1e-6)


def test_tight_inside_offset_is_not_made_drivable():
    p = chain(np.array([[.14, 0], [.2, 0], [.2, .2]]))
    with pytest.raises(ValueError, match='reverses'):
        connected_miter_center(p, .1)


def test_intersecting_source_is_rejected():
    p = chain(np.array([[.14, 0], [.4, .2], [.14, .2], [.4, 0]]))
    with pytest.raises(ValueError, match='intersects'):
        connected_miter_center(p, .02)


def test_s_turn_keeps_observation_order_not_x_sorting():
    p = chain(np.array([[.14, 0], [.35, 0], [.35, .4], [.6, .4]]))
    actual, center = connected_miter_center(p, -.05)
    assert np.allclose(actual[-1], [.6, .4], atol=1e-6)
    assert np.allclose(center[-1], [.6, .35], atol=1e-6)


def test_miter_fallback_is_opt_in_and_only_after_offset_failure(monkeypatch):
    import pinky_move.metric_lane as lane
    p = chain(np.array([[.14, 0], [.4, 0], [.4, .4]]))
    def folded(*args):
        raise lane.OffsetCurveError('center_path curve folds back', np.empty((0, 2)))
    monkeypatch.setattr(lane, 'center_path_from_boundary', folded)
    with pytest.raises(lane.OffsetCurveError):
        lane.fit_drivable_boundary(p, .1, fitted=p)
    actual, center, warning = lane.fit_drivable_boundary(
        p, .1, fitted=p, connected_geometry=True)
    assert 'miter fallback' in warning
    assert len(actual) == len(center) == 100


def test_successful_normal_center_is_unchanged_when_enabled():
    from pinky_move.metric_lane import fit_drivable_boundary
    p = np.column_stack((np.linspace(.14, .48, 30), np.full(30, .1)))
    before = fit_drivable_boundary(p, -.08)
    after = fit_drivable_boundary(p, -.08, connected_geometry=True)
    for a, b in zip(before[:2], after[:2]):
        np.testing.assert_array_equal(a, b)
    assert before[2] == after[2]


def test_tracker_cache_does_not_reuse_previous_frame(monkeypatch):
    import pinky_move.metric_lane as lane
    tracker = lane.MetricLaneTracker(connected_geometry=True)
    mask = np.zeros((20, 20), np.uint8)
    calibration = {}
    calls = []
    monkeypatch.setattr(lane, 'floor_curves', lambda *args, **kwargs: calls.append(kwargs) or [])
    tracker._floor_curves([mask], calibration, 2., .14, .48, 3)
    tracker._floor_curves([mask], calibration, 2., .14, .48, 3)
    assert len(calls) == 1 and calls[0]['connected_geometry']
    with pytest.raises(ValueError):
        tracker.update([], calibration, 1.)
    tracker._floor_curves([mask], calibration, 2., .14, .48, 3)
    assert len(calls) == 2


def test_enabled_controller_keeps_history_and_state_machine(controller):
    controller.parameters['connected_geometry'] = True
    controller._reset_transient_state()
    assert controller.metric_tracker.connected_geometry
    assert controller.corner_policy is not None
    assert controller.lane_history is not None
    assert not controller.parameters['simulation_white_lane']


def test_successful_axis_does_not_run_new_skeleton(monkeypatch):
    import pinky_move.metric_lane as lane
    p = np.column_stack((np.linspace(.14, .4, 50), np.full(50, .08)))
    mask = np.ones((20, 20), np.uint8)
    monkeypatch.setattr(lane, '_axis_floor_curves', lambda masks, cal, maximum, column: [] if column else [p])
    monkeypatch.setattr(lane, 'connected_floor_curve', lambda *args: pytest.fail('successful axis replaced'))
    curves = lane.floor_curves([mask], {}, 2., .14, .48, connected_geometry=True)
    np.testing.assert_array_equal(curves[0], p)


def test_existing_long_s_success_target_is_identical_with_option():
    from pinky_move.metric_lane import MetricLaneTracker
    from test_connected_approach import scene
    mask, calibration = scene()
    targets = [MetricLaneTracker(connected_geometry=flag).update(
        [mask], calibration, 1., lookahead=.22, lane_width=.168,
        path_min_m=.14, path_max_m=.48) for flag in (False, True)]
    for key in ('x_m', 'y_m', 'center_path', 'actual_curve'):
        np.testing.assert_array_equal(targets[0][key], targets[1][key])
