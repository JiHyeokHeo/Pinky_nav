import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from pinky_move.metric_lane import (metric_target, pursuit, MetricLaneTracker, LaneWidthEstimator, initial_boundary_side,
                                   floor_curves, fit_boundary, normal_offset, select_lookahead,
                                   center_from_pair, loss_speed_scale, LaneImageIdentity, near_pixel_side)
from pinky_move.robot_projection import robot_floor_point, robot_floor_pixel, draw_metric_target, validate_calibration


def calibration():
    return json.loads((Path(__file__).parents[1]/'config/robot_floor_calibration.json').read_text())


@pytest.mark.parametrize('column,expected', [(15, 'left'), (85, 'right'), (50, None)])
def test_near_pixel_recovery_hint(column, expected):
    mask = np.zeros((100, 100), np.uint8)
    mask[70:95, column-2:column+3] = 1
    assert near_pixel_side(mask) == expected
    mask[65:] = 0
    mask[20:40, column-2:column+3] = 1
    assert near_pixel_side(mask) is None


@pytest.mark.parametrize('column,expected', [(15, 'left'), (85, 'right')])
def test_near_pixel_expanded_band_uses_pixels_above_sparse_tip(column, expected):
    mask = np.zeros((100, 100), np.uint8)
    mask[80:89, column-2:column+3] = 1
    mask[89:100, column] = 1
    # The old 8% band contained only nine pixels: too few for a hint.
    assert near_pixel_side(mask) == expected


def test_near_pixel_expansion_preserves_far_cutoff_and_centre_deadband():
    mask = np.zeros((100, 100), np.uint8)
    mask[50:65, 10:20] = 1
    assert near_pixel_side(mask) is None
    mask[:] = 0
    mask[80:100, 48:53] = 1
    assert near_pixel_side(mask) is None


def test_pixel_side_recovers_missing_lock_only_after_three_frames():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.requires_pair = True  # Pair was lost before either side was locked.
    for now in (1., 1.2):
        with pytest.raises(ValueError, match='reacquiring right'):
            tracker.update([masks[1]], c, now, lane_width=.16, recovery_side_hint='right')
        assert tracker.confirmed is None
    target = tracker.update([masks[1]], c, 1.4, lane_width=.16, recovery_side_hint='right')
    assert target['visible_side'] == 'right' and target['inferred']


@pytest.mark.parametrize('index,side', [(0,'left'), (1,'right')])
def test_returning_pair_keeps_current_boundary_until_three_matches(index,side):
    tracker=MetricLaneTracker()
    masks=masks_for_lines(); c=calibration()
    tracker.update([masks[index]],c,0.,lane_width=.16)
    for now,count in [(.2,1),(.4,2)]:
        result=tracker.update(masks,c,now,lane_width=.16)
        assert result['boundary_count']==1
        assert result['pair_transition_count']==count
        assert tracker.last_observation['side']==side
    result=tracker.update(masks,c,.6,lane_width=.16)
    assert result['boundary_count']==2 and result['pair_transition_count']==3
    assert tracker.last_observation['side']==side
    tracker.update(masks,c,.8,lane_width=.16)
    assert tracker.last_observation['side']==side


def test_returning_pair_confirmation_resets_on_missing_opposite():
    tracker=MetricLaneTracker(); masks=masks_for_lines(); c=calibration()
    tracker.update([masks[1]],c,0.,lane_width=.16)
    assert tracker.update(masks,c,.2,lane_width=.16)['pair_transition_count']==1
    tracker.update([masks[1]],c,.4,lane_width=.16)
    assert tracker.update(masks,c,.6,lane_width=.16)['pair_transition_count']==1


def test_new_pair_cannot_bypass_width_stop_while_boundary_is_locked():
    tracker=MetricLaneTracker(minimum_lane_width_m=.152)
    masks=masks_for_lines(); c=calibration()
    tracker.update([masks[1]],c,0.,lane_width=.16)
    with pytest.raises(ValueError,match='implausibly narrow'):
        tracker.update(masks_for_lines(half_width=.05),c,.2,lane_width=.16)


def test_shifted_new_pair_does_not_replace_continuing_centre(monkeypatch):
    import pinky_move.metric_lane as lane
    tracker=MetricLaneTracker(); masks=masks_for_lines(); c=calibration()
    original=lane.metric_target
    first=tracker.update([masks[1]],c,0.,lane_width=.16)
    def shifted(*args,**kwargs):
        candidate=original(*args,**kwargs)
        candidate['center_path']=(np.asarray(candidate['center_path'])+[0.,.08]).tolist()
        candidate['y_m']+=.08
        return candidate
    monkeypatch.setattr(lane,'metric_target',shifted)
    for now in (.2,.4,.6):
        result=tracker.update(masks,c,now,lane_width=.16)
        assert result['boundary_count']==1 and result['pair_transition_count']==0
        assert result['y_m']==pytest.approx(first['y_m'])
        assert tracker.last_observation['side']=='right'


def test_pixel_recovery_side_change_restarts_confirmation():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.requires_pair = True
    for now, side, index in ((1., 'right', 1), (1.2, 'left', 0), (1.4, 'right', 1)):
        with pytest.raises(ValueError, match='1/3'):
            tracker.update([masks[index]], c, now, lane_width=.16, recovery_side_hint=side)
    with pytest.raises(ValueError, match='locked boundary side'):
        tracker.update([masks[1]], c, 1.6, lane_width=.16)


def test_pixel_hint_never_overrides_matching_locked_geometry():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    for now in (1.2, 1.4):
        with pytest.raises(ValueError, match='reacquiring left'):
            tracker.update([masks[0]], c, now, lane_width=.16, recovery_side_hint='right')
    target = tracker.update([masks[0]], c, 1.6, lane_width=.16, recovery_side_hint='right')
    assert target['visible_side'] == 'left'


def test_width_estimator_fixed_tick_and_sustained_remeasurement():
    estimator = LaneWidthEstimator(interval_s=.5, tolerance_m=.02, minimum_m=.152)
    assert estimator.observe(.16, 0.) is None
    assert estimator.observe(.25, .2) is None  # no new measurement before the tick
    assert estimator.observe(.161, .5) is None
    assert estimator.observe(.159, 1.) == pytest.approx(.16, abs=.001)
    with pytest.raises(ValueError, match='implausibly narrow'):
        estimator.observe(.14, 1.5)
    assert estimator.value == pytest.approx(.16, abs=.001)
    assert estimator.observe(.20, 1.5) == pytest.approx(.16, abs=.001)
    assert estimator.observe(.201, 2.) == pytest.approx(.16, abs=.001)
    assert estimator.observe(.199, 2.5) == pytest.approx(.20, abs=.001)


def test_robot_narrow_pair_cannot_become_single_line_fallback():
    tracker = MetricLaneTracker(minimum_lane_width_m=.152)
    with pytest.raises(ValueError, match='implausibly narrow'):
        tracker.update(masks_for_lines(half_width=.05), calibration(), 0.)
    assert tracker.last_observation is None
    assert tracker.confirmed is None


def masks_for_lines(offset=0., slope=0., half_width=.08, far=.6):
    c = calibration()
    t = np.array(c['base_from_upright_optical'])
    r = t[:3, :3].T; tv = -r@t[:3, 3]
    rv, _ = cv2.Rodrigues(r)
    masks = []
    for y in (half_width+offset, -half_width+offset):
        world = np.array([[x, y+slope*(x-.28), c.get('ground_plane_z_m', 0.)]
                          for x in np.linspace(.10, far, 200)])
        uv, _ = cv2.projectPoints(world, rv, tv, np.array(c['camera_matrix']),
                                  np.array(c['distortion_coefficients']))
        mask = np.zeros((480, 640), np.uint8)
        cv2.polylines(mask, [uv.astype(np.int32)], False, 1, 5)
        masks.append(mask)
    return masks


def stopped_corner_masks():
    """Approximate debug outlines, NOT original masks or metric ground truth.

    Near right stripe is too thick for column sampling; far left stripe splits
    into similar-length column tracks. Row extraction retains both real spans.
    """
    polygons = [
        [[639,330],[581,330],[564,333],[538,342],[505,362],[481,381],
         [449,412],[434,434],[430,443],[428,453],[428,479],[544,479],
         [547,460],[551,454],[554,444],[568,419],[588,395],[594,380],
         [603,373],[613,369],[626,359],[633,356],[639,356]],
        [[639,258],[619,258],[612,255],[604,255],[596,252],[528,249],
         [466,251],[447,255],[416,254],[353,260],[307,259],[275,262],
         [228,263],[188,267],[107,283],[59,297],[40,301],[5,313],[0,313],
         [0,343],[9,343],[12,341],[27,338],[28,336],[33,336],[42,332],
         [53,330],[69,323],[174,292],[200,282],[236,274],[327,270],
         [341,267],[343,262],[346,260],[403,264],[462,262],[571,263],
         [608,266],[639,265]]]
    masks = []
    for polygon in polygons:
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
        masks.append(mask)
    return masks


def test_stopped_corner_dual_axis_and_locked_single_fallback():
    c = calibration(); masks = stopped_corner_masks()
    curves = floor_curves(masks, c)
    assert len(curves) == 2
    assert all(np.ptp(p[:, 0]) > .04 for p in curves)
    # Their real x supports barely overlap: never fabricate a measured pair.
    with pytest.raises(ValueError):
        metric_target(masks, c, path_min_m=.14, path_max_m=.48)
    tracker = MetricLaneTracker()
    tracker.observed_side = 'left'
    tracker.observed_curve = curves[1]
    tracker.observed_at = 0.
    tracker.requires_pair = True
    for now, message in ((.3, '1/3'), (.6, '2/3')):
        with pytest.raises(ValueError, match=message):
            tracker.update(masks, c, now, lane_width=.154,
                           path_min_m=.14, path_max_m=.48, image_match=(1, 'left'))
    target = tracker.update(masks, c, .9, lane_width=.154,
                            path_min_m=.14, path_max_m=.48, image_match=(1, 'left'))
    assert target['inferred'] and target['visible_side'] == 'left'
    assert target['source_mask_index'] == 1
    assert abs(target['y_m']) < .03 and .25 < target['x_m'] < .30


def test_multi_mask_flow_preserves_only_locked_identity(monkeypatch):
    c = calibration(); mask = masks_for_lines()[0]
    tracker = MetricLaneTracker()
    tracker.update([mask], c, 0., lane_width=.16)
    monkeypatch.setattr(tracker, '_same_curve', lambda *args: False)
    empty = np.zeros_like(mask)
    # Model instance order can change; match index is carried into the target.
    target = tracker.update([empty, mask], c, .3, lane_width=.16,
                            image_match=(1, 'left'))
    assert target['source_mask_index'] == 1 and target['visible_side'] == 'left'
    with pytest.raises(ValueError):
        tracker.update([empty, mask], c, .6, lane_width=.16,
                       image_match=(1, 'right'))


def test_straight_and_offset():
    centre = metric_target(masks_for_lines(), calibration())
    assert abs(centre['y_m']) < .003
    assert abs(centre['width_m']-.16) < .006
    left = metric_target(masks_for_lines(.02), calibration())
    assert .017 < left['y_m'] < .023
    assert pursuit(left, .02) > 0


def test_loss_and_limit():
    with pytest.raises(ValueError):
        metric_target(masks_for_lines()[:1], calibration())
    assert abs(pursuit(dict(x_m=.15, y_m=.4), .2)) <= .15
    with pytest.raises(ValueError):
        robot_floor_point(320, 0, calibration(), (640, 480))
    with pytest.raises(ValueError):
        robot_floor_point(320, 400, calibration(), (320, 240))


def test_adapts_to_observed_overlap_without_extrapolation():
    c = calibration(); masks = masks_for_lines()
    for row in range(480):
        try:
            x, _ = robot_floor_point(320, row, c, (640, 480))
            if x > .265:
                masks[1][row] = 0
        except ValueError:
            masks[1][row] = 0
    target = metric_target(masks, c)
    assert .18 <= target['x_m'] < .265
    assert target['adaptive']
    assert abs(target['y_m']) < .005


def test_rejects_abrupt_pair_switch():
    with pytest.raises(ValueError):
        metric_target(masks_for_lines(), calibration(),
                      previous={'y_m': .15, 'width_m': .16})


def test_target_pixel_round_trip_and_drawing():
    c = calibration()
    u, v = robot_floor_pixel(.24, -.022, c, (640, 480))
    np.testing.assert_allclose(robot_floor_point(u, v, c, (640, 480)), [.24, -.022], atol=1e-5)
    frame = np.zeros((480, 640, 3), np.uint8)
    assert not draw_metric_target(frame, None, c)
    assert not frame.any()
    assert draw_metric_target(frame, dict(x_m=.24, y_m=-.022), c)
    assert frame[int(v)-10:int(v)+11, int(u)-10:int(u)+11].any()
    with pytest.raises(ValueError):
        robot_floor_pixel(.24, 10., c, (640, 480))
    with pytest.raises(ValueError):
        robot_floor_pixel(.24, 0., c, (320, 240))


@pytest.mark.parametrize('side', [0, 1])
def test_virtual_boundary_and_expiry(side):
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    with pytest.raises(ValueError):
        tracker.update([masks[side]], c, 0.)
    tracker.update(masks, c, .1)
    with pytest.raises(ValueError):
        tracker.update([masks[side]], c, .2)  # one pair is insufficient
    tracker.update(masks, c, .3)
    tracker.update(masks, c, .4)
    target = tracker.update([masks[side]], c, .7, timeout_s=1.)
    assert target['inferred'] and target['boundary_count'] == 1
    assert target['visible_side'] == ('left' if side == 0 else 'right')
    assert abs(target['y_m']) < .005
    assert tracker.confirmed_at == .4
    target = tracker.update([masks[side]], c, 1.3, timeout_s=1.)
    assert tracker.confirmed_at == .4
    with pytest.raises(ValueError):
        tracker.update([masks[side]], c, 1.5, timeout_s=1.)


def test_ten_second_limit_not_renewed_by_tracking():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    for now in np.arange(.4, 10.0, .3):
        result = tracker.update([masks[0]], c, float(now), timeout_s=10.)
        assert result['visible_side'] == 'left'
        assert tracker.confirmed_at == .1
    with pytest.raises(ValueError):
        tracker.update([masks[0]], c, 10.2, timeout_s=10.)


def test_continuous_single_line_updates_beyond_ten_seconds():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    ys = []
    for now in np.arange(.4, 15., .3):
        moving = masks_for_lines(offset=.008*np.sin(now*.3))
        target = tracker.update([moving[0]], c, float(now))
        assert target['visible_side'] == 'left'
        assert target['inferred']
        ys.append(target['y_m'])
    assert np.ptp(ys) > .01  # New frame geometry, not a frozen target.
    assert tracker.confirmed_at == .1  # Never mislabel inferred width as measured.
    with pytest.raises(ValueError):
        tracker.update([moving[0]], c, 16.5)  # stale reference still stops


def test_first_single_line_cannot_reuse_stale_pair():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 2.)


def test_configured_single_reacquires_after_three_stable_observations():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    for now in (1.2, 1.4):
        with pytest.raises(ValueError, match='reacquiring'):
            tracker.update([masks[0]], c, now, lane_width=.16)
    result = tracker.update([masks[0]], c, 1.6, lane_width=.16)
    assert result['visible_side'] == 'left'
    assert result['width_source'] == 'configured'
    assert not tracker.requires_pair


def test_reacquisition_opposite_side_and_empty_reset_confirmation():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    for now in (1.2, 1.4, 1.6, 1.8):
        with pytest.raises(ValueError, match='locked boundary side'):
            tracker.update([masks[1]], c, now, lane_width=.16)
    with pytest.raises(ValueError, match='1/3'):
        tracker.update([masks[0]], c, 2., lane_width=.16)
    with pytest.raises(ValueError):
        tracker.update([], c, 2.1, lane_width=.16)
    with pytest.raises(ValueError, match='1/3'):
        tracker.update([masks[0]], c, 2.2, lane_width=.16)


def test_extra_instance_preserves_unique_locked_boundary_not_duplicates():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    empty = np.zeros_like(masks[0])
    result = tracker.update([masks[0], empty], c, .3, lane_width=.16)
    assert result['boundary_count'] == 1
    with pytest.raises(ValueError):
        tracker.update([masks[0], masks[0]], c, .6, lane_width=.16)


def test_positive_width_timeout_cannot_be_bypassed_by_reacquisition():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., timeout_s=1., lane_width=.16)
    for now in (1.1, 1.3, 1.5, 1.7):
        with pytest.raises(ValueError):
            tracker.update([masks[0]], c, now, timeout_s=1., lane_width=.16)


def test_recovery_checks_history_before_initial_sign(monkeypatch):
    import pinky_move.metric_lane as module
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    def ambiguous(*args):
        raise AssertionError('cold-start classifier must not override matching history')
    monkeypatch.setattr(module, 'initial_boundary_side', ambiguous)
    for now in (1.2, 1.4):
        with pytest.raises(ValueError, match='reacquiring'):
            tracker.update([masks[0]], c, now, lane_width=.16)
    assert tracker.update([masks[0]], c, 1.6, lane_width=.16)['visible_side'] == 'left'


def test_fold_diagnostics_identify_center_and_virtual():
    x = np.linspace(.1, .5, 80)
    curve = np.column_stack((x, 5*(x-.3)**2))
    for role in ('center_path', 'virtual_boundary'):
        with pytest.raises(ValueError, match=role+' curve folds back'):
            normal_offset(curve, .3, role)


def test_failed_recovery_does_not_commit_state(monkeypatch):
    import pinky_move.metric_lane as module
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    original = module.normal_offset
    def fail_center(curve, distance, role='offset'):
        if role == 'center_path':
            raise ValueError('center_path curve folds back')
        return original(curve, distance, role)
    monkeypatch.setattr(module, 'normal_offset', fail_center)
    for now in (1.2, 1.4, 1.6, 1.8):
        with pytest.raises(ValueError):
            tracker.update([masks[0]], c, now, lane_width=.16)
        assert tracker.requires_pair and tracker.confirmed is None
        assert tracker.observed_side == 'left' and tracker.observed_at == 0.
    monkeypatch.setattr(module, 'normal_offset', original)
    result = None
    for now in (2., 2.2, 2.4):
        try:
            result = tracker.update([masks[0]], c, now, lane_width=.16)
        except ValueError:
            pass
    assert result is not None and result['visible_side'] == 'left'


def test_virtual_fold_is_optional_but_center_fold_is_not(monkeypatch):
    import pinky_move.metric_lane as module
    original = module.normal_offset
    def fail_virtual(curve, distance, role='offset'):
        if role == 'virtual_boundary':
            raise ValueError('virtual_boundary curve folds back')
        return original(curve, distance, role)
    monkeypatch.setattr(module, 'normal_offset', fail_virtual)
    tracker = MetricLaneTracker()
    target = tracker.update([masks_for_lines()[0]], calibration(), 0., lane_width=.16)
    assert target['virtual_curve'] == []
    assert target['virtual_boundary_warning']
    assert len(target['center_path']) > 5 and 'right_curve' not in target


def test_folded_path_keeps_fresh_identity_but_never_returns_target(monkeypatch):
    import pinky_move.metric_lane as module
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    valid = tracker.update([masks[0]], c, 0., lane_width=.16)
    original = module.normal_offset
    def fail_center(curve, distance, role='offset'):
        if role == 'center_path':
            raise ValueError('center_path curve folds back')
        return original(curve, distance, role)
    monkeypatch.setattr(module, 'normal_offset', fail_center)
    for now in (.3, .6, .9, 1.2, 1.5):
        with pytest.raises(ValueError, match='center_path curve folds back'):
            tracker.update([masks[0]], c, now, lane_width=.16)
        assert tracker.observed_at == now
        assert tracker.observed_side == 'left'
        assert tracker.previous is valid
        assert not tracker.requires_pair
    monkeypatch.setattr(module, 'normal_offset', original)
    assert tracker.update([masks[0]], c, 1.8, lane_width=.16)['visible_side'] == 'left'
    # True absence still expires identity tracking; no stale-target bypass.
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 3., lane_width=.16)


def test_recovery_identity_confirmation_survives_invalid_path(monkeypatch):
    import pinky_move.metric_lane as module
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update([masks[0]], c, 0., lane_width=.16)
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([masks[0]], c, 1., lane_width=.16)
    original = module.normal_offset
    def fail_center(curve, distance, role='offset'):
        if role == 'center_path':
            raise ValueError('center_path curve folds back')
        return original(curve, distance, role)
    monkeypatch.setattr(module, 'normal_offset', fail_center)
    for now, expected in ((1.2, '1/3'), (1.4, '2/3'),
                          (1.6, 'center_path curve folds back'),
                          (1.8, 'center_path curve folds back')):
        with pytest.raises(ValueError, match=expected):
            tracker.update([masks[0]], c, now, lane_width=.16)
        assert tracker.confirmed is None and tracker.requires_pair
    monkeypatch.setattr(module, 'normal_offset', original)
    assert tracker.update([masks[0]], c, 2., lane_width=.16)['visible_side'] == 'left'
    assert tracker.recovery is None


def test_configured_forward_interval_clips_single_boundary():
    c = calibration(); mask = masks_for_lines(far=.8)[0]
    tracker = MetricLaneTracker()
    target = tracker.update([mask], c, 0., lane_width=.16,
                            path_min_m=.14, path_max_m=.48)
    actual = np.asarray(target['actual_curve'])
    assert actual[0, 0] >= .14
    assert actual[-1, 0] <= .48
    assert target['x_m'] >= .05 and target['x_m'] <= .48
    with pytest.raises(ValueError, match='interval'):
        MetricLaneTracker().update([mask], c, 0., lane_width=.16,
                                   path_min_m=.3, path_max_m=.48)


def test_last_segmentation_image_carries_unique_role_across_camera_motion():
    frame = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    cv2.line(frame, (70, 450), (520, 260), (220, 220, 220), 20)
    cv2.line(mask, (70, 450), (520, 260), 1, 20)
    history = LaneImageIdentity()
    assert history.match(frame, [mask], 0.) is None
    history.commit(0., {0: 'left'})
    shift = np.float32([[1, 0, 20], [0, 1, 0]])
    moved = cv2.warpAffine(frame, shift, (640, 480))
    moved_mask = cv2.warpAffine(mask, shift, (640, 480))
    assert history.match(moved, [moved_mask], .3) == (0, 'left')
    assert history.match(moved, [moved_mask, moved_mask], .3) is None
    assert history.match(moved, [moved_mask], 1.) is None


def test_short_disjoint_white_patch_does_not_destroy_true_stripe():
    mask = np.zeros((480, 640), np.uint8)
    cv2.line(mask, (100, 470), (300, 330), 1, 9)
    clean = floor_curves([mask], calibration())[0]
    polluted = mask.copy()
    cv2.rectangle(polluted, (570, 390), (639, 414), 1, -1)
    polluted[400:402, 150:640] = 1  # Bad segmentation bridge across bare floor.
    curves = floor_curves([polluted], calibration())
    assert len(curves) == 1
    assert len(curves[0]) >= len(clean)-3
    for x, y in curves[0]:
        assert abs(y-np.interp(x, clean[:, 0], clean[:, 1])) < .005


def test_equally_supported_merged_stripes_are_not_arbitrarily_selected():
    mask = np.zeros((480, 640), np.uint8)
    cv2.line(mask, (100, 470), (200, 330), 1, 9)
    cv2.line(mask, (440, 470), (540, 330), 1, 9)
    assert floor_curves([mask], calibration()) == []


def test_image_role_can_recover_after_geometry_gap_without_opposite_side():
    c = calibration(); tracker = MetricLaneTracker(); masks = masks_for_lines()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    shifted = masks_for_lines(offset=.08)[0]
    with pytest.raises(ValueError, match='reference tracking gap'):
        tracker.update([shifted], c, 1., lane_width=.16,
                       image_side_hint='left')
    assert tracker.requires_pair and tracker.observed_side == 'left'
    for now in (1.2, 1.4):
        with pytest.raises(ValueError, match='reacquiring'):
            tracker.update([shifted], c, now, lane_width=.16,
                           image_side_hint='left')
    target = tracker.update([shifted], c, 1.6, lane_width=.16,
                            image_side_hint='left')
    assert target['visible_side'] == 'left'


def test_locked_role_cannot_switch_and_tracking_gap_stops():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    tracker.update([masks[0]], c, .3)
    with pytest.raises(ValueError):
        tracker.update([masks[1]], c, .6)
    with pytest.raises(ValueError):
        tracker.update([masks[0]], c, 1.5)


def test_initial_side_uses_robot_reference_and_rejects_centre():
    c = calibration(); masks = masks_for_lines()
    curves = floor_curves(masks, c)
    assert initial_boundary_side(curves[0]) == 'left'
    assert initial_boundary_side(curves[1]) == 'right'
    with pytest.raises(ValueError):
        initial_boundary_side(np.array([[.18, -.005], [.20, .005], [.38, .02]]))
    tracker = MetricLaneTracker()
    with pytest.raises(ValueError, match='initial boundary=right'):
        tracker.update([masks[1]], c, 0.)
    assert tracker.initial_side == 'right'


def test_initial_identity_is_independent_of_far_crossing_and_lookahead():
    curve = np.array([[.15, .06], [.17, .05], [.28, 0.], [.4, -.08]])
    for distance in (.2, .28, .4):
        assert initial_boundary_side(curve, distance) == 'left'
        mirrored = curve.copy(); mirrored[:, 1] *= -1
        assert initial_boundary_side(mirrored, distance) == 'right'


def test_invalid_longer_axis_cannot_replace_valid_shorter_axis(monkeypatch):
    import pinky_move.metric_lane as module
    row = np.column_stack((np.linspace(.15,.33,50),np.full(50,.07)))
    column = np.column_stack((np.linspace(.15,.334,50),np.full(50,.07)))
    column[25,1] += .12
    with pytest.raises(ValueError, match='residual|self intersection'):
        fit_boundary(column)
    monkeypatch.setattr(module, '_axis_floor_curves',
                        lambda masks,c,max_f,by_column:[column if by_column else row])
    selected = floor_curves([object()], None, 2., .14, .48)
    np.testing.assert_array_equal(selected[0], row)


def test_axis_fit_selection_uses_only_configured_support(monkeypatch):
    import pinky_move.metric_lane as module
    near = np.column_stack((np.linspace(.15,.4,30),np.full(30,.07)))
    outside = np.column_stack((np.linspace(.7,.9,20),.3*np.sin(np.arange(20))))
    points = np.vstack((near,outside))
    monkeypatch.setattr(module, '_axis_floor_curves', lambda *args:[points])
    assert len(floor_curves([object()],None,2.,.14,.48)) == 1
    assert floor_curves([object()],None,2.,.14,1.) == []


def test_remote_stop_outline_selects_valid_axis_and_nearer_target():
    # Approximation of saved debug outline, not original segmentation ground truth.
    polygon = [[639,271],[618,270],[609,267],[503,267],[478,271],[430,271],
               [389,275],[351,283],[340,284],[312,292],[303,293],[289,298],
               [277,300],[261,306],[194,324],[125,349],[99,361],[86,365],
               [40,388],[10,400],[4,404],[0,404],[0,472],[4,472],[19,460],
               [33,454],[43,447],[58,440],[75,428],[107,411],[118,407],
               [153,387],[183,374],[202,362],[218,355],[247,346],[262,339],
               [295,329],[309,323],[431,290],[443,285],[466,280],[488,279],
               [620,282],[639,280]]
    mask = np.zeros((480,640),np.uint8)
    cv2.fillPoly(mask,[np.array(polygon,np.int32)],1)
    c = calibration()
    targets = []
    for lookahead in (.28,.22):
        target = MetricLaneTracker().update([mask],c,0.,lookahead=lookahead,
                                             lane_width=.154,path_min_m=.14,path_max_m=.48)
        assert target['visible_side'] == 'left' and target['inferred']
        targets.append(target)
    assert targets[1]['x_m'] < targets[0]['x_m']-.03
    assert np.hypot(targets[1]['x_m'],targets[1]['y_m']) == pytest.approx(.22)
    assert .19 < targets[1]['x_m'] < .23 and -.08 < targets[1]['y_m'] < -.02
    # Symmetric real right-side geometry must offset LEFT, not apply a fixed bias.
    target = MetricLaneTracker().update([masks_for_lines()[1]],c,0.,lookahead=.22,lane_width=.16)
    assert target['visible_side'] == 'right' and abs(target['y_m']) < .008


def test_current_start_frame_can_bootstrap_left_without_history():
    # Approximate saved debug outline; regression coverage, not ground truth.
    polygon = [[639,230],[630,230],[622,228],[592,228],[560,233],[493,238],
               [404,252],[384,258],[370,260],[299,282],[286,284],[273,289],
               [232,300],[207,309],[154,331],[134,341],[99,361],[88,365],
               [69,376],[59,380],[7,412],[0,414],[0,479],[28,479],[28,476],
               [40,465],[59,455],[71,444],[86,435],[125,406],[188,364],
               [213,349],[268,323],[322,305],[333,300],[346,297],[356,292],
               [390,281],[401,276],[437,267],[451,262],[518,252],[585,245],
               [610,244],[639,240]]
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
    tracker = MetricLaneTracker()
    target = tracker.update([mask], calibration(), 0., lane_width=.154,
                            path_min_m=.14, path_max_m=.48)
    assert target['inferred'] and target['visible_side'] == 'left'
    assert target['x_m'] > .14 and np.isfinite(target['y_m'])


def test_horizontal_bend_uses_column_centres_not_wide_row_midpoints():
    c = calibration()
    mask = np.zeros((480, 640), np.uint8)
    u = np.arange(15, 626)
    v = .0002*u*u-.35*u+460
    cv2.polylines(mask, [np.column_stack((u, v)).astype(np.int32)], False, 1, 26)
    # Long row intersections exceed the old 128-pixel rejection threshold.
    assert any(np.ptp(np.flatnonzero(row)) > 128 for row in mask
               if np.count_nonzero(row) > 1)
    curves = floor_curves([mask], c)
    assert len(curves) == 1 and len(curves[0]) > 60
    expected = np.array([robot_floor_point(float(x), float(y), c, (640, 480))
                         for x, y in zip(u, v)])
    expected = expected[np.argsort(expected[:, 0])]
    measured = curves[0]
    inner = measured[(measured[:, 0] > expected[8, 0]) &
                     (measured[:, 0] < expected[-9, 0])]
    error = abs(inner[:, 1]-np.interp(inner[:, 0], expected[:, 0], expected[:, 1]))
    assert np.quantile(error, .95) < .008
    assert MetricLaneTracker._same_curve(measured, measured)


def test_column_sampler_rejects_equal_branches_and_wide_blob():
    mask = np.zeros((480, 640), np.uint8)
    cv2.line(mask, (10, 330), (630, 310), 1, 12)
    cv2.line(mask, (10, 410), (630, 390), 1, 12)
    assert floor_curves([mask], calibration()) == []
    blob = np.zeros_like(mask)
    cv2.rectangle(blob, (10, 290), (630, 475), 1, -1)
    assert floor_curves([blob], calibration()) == []


def test_normal_offset_on_slanted_lane():
    c = calibration(); masks = masks_for_lines(slope=.5); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.)
    measured = tracker.update(masks, c, .1)
    assert abs(measured['normal_width_m']-.16/np.hypot(1., .5)) < .01
    inferred = tracker.update([masks[0]], c, .3)
    assert abs(inferred['y_m']-.5*(inferred['x_m']-.28)) < .01
    actual = np.array(inferred['actual_curve']); virtual = np.array(inferred['virtual_curve'])
    assert np.median(np.abs(actual[:, 0]-virtual[:, 0])) > .03
    frame = np.zeros((480, 640, 3), np.uint8)
    assert draw_metric_target(frame, inferred, c)


def test_wrong_boundary_and_total_loss_rejected():
    c = calibration(); masks = masks_for_lines(); tracker = MetricLaneTracker()
    tracker.update(masks, c, 0.); tracker.update(masks, c, .1)
    with pytest.raises(ValueError):
        tracker.update([], c, .2)
    with pytest.raises(ValueError):
        tracker.update([masks_for_lines(offset=.08)[1]], c, .3)


@pytest.mark.parametrize('side', [0, 1])
def test_cold_start_single_with_configured_width(side):
    tracker = MetricLaneTracker()
    target = tracker.update([masks_for_lines()[side]], calibration(), 0., lane_width=.16)
    assert target['inferred'] and target['width_source'] == 'configured'
    assert target['visible_side'] == ('left' if side == 0 else 'right')
    assert abs(target['y_m']) < .006
    assert len(target['center_path']) >= 5
    # A real pair is allowed to replace the configured prior, not vice versa.
    tracker.update(masks_for_lines(), calibration(), .3, lane_width=.16)
    tracker.update(masks_for_lines(), calibration(), .6, lane_width=.16)
    # Returning opposite boundaries now require three continuous observations.
    tracker.update(masks_for_lines(), calibration(), .9, lane_width=.16)
    assert tracker.confirmed['width_source'] == 'measured'


def test_requested_example_scale_and_pure_pursuit():
    masks = masks_for_lines(half_width=.3, far=1.5)
    target = metric_target(masks, calibration(), lookahead=.7)
    assert abs(np.hypot(target['x_m'], target['y_m'])-.7) < 1e-5
    assert abs(target['width_m']-.6) < .02
    single = MetricLaneTracker().update([masks[0]], calibration(), 0., lookahead=.7, lane_width=.6)
    assert abs(single['y_m']) < .02
    alpha = np.arctan2(.2, .7); ld = np.hypot(.7, .2)
    assert np.isclose(pursuit(dict(x_m=.7, y_m=.2), .2, 1.), .2*2*np.sin(alpha)/ld)
    assert pursuit(dict(x_m=.7, y_m=-.2), .2, 1.) < 0


def test_s_curve_normal_offset_and_radial_lookahead():
    xs = np.linspace(.15, 1.2, 80)
    ys = .2*(xs-.65)**3-.05*(xs-.65)
    curve = fit_boundary(np.column_stack((xs, ys)), degree=3)
    shifted = normal_offset(curve, -.3)
    delta = shifted-curve
    tangent = np.gradient(curve, axis=0)  # Parametric tangent, not dy/dx.
    np.testing.assert_allclose(np.sum(delta*tangent, axis=1), 0., atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), .3, atol=1e-10)
    target, adaptive = select_lookahead(curve, .7)
    assert not adaptive and abs(np.linalg.norm(target)-.7) < 1e-8
    end, adaptive = select_lookahead(curve, 2.)
    np.testing.assert_allclose(end, curve[-1]); assert adaptive


def test_base_link_floor_frame_equivalence_and_validation():
    c = calibration(); validate_calibration(c)
    assert c['frame_id'] == 'base_link' and c['ground_plane_z_m'] == -.028
    footprint = json.loads(json.dumps(c))
    footprint['base_from_upright_optical'][2][3] += .028
    footprint['frame_id'] = 'base_footprint'; footprint['ground_plane_z_m'] = 0.
    np.testing.assert_allclose(robot_floor_point(350, 330, c, (640, 480)),
                               robot_floor_point(350, 330, footprint, (640, 480)))
    del c['ground_plane_z_m']
    with pytest.raises(ValueError): validate_calibration(c)


def test_loss_ramp_stops_without_new_inference():
    assert loss_speed_scale(.1, .3, .8) == 1.
    assert 0 < loss_speed_scale(.5, .3, .8) < 1.
    assert loss_speed_scale(.8, .3, .8) == 0.
    assert loss_speed_scale(10., .3, .8) == 0.
