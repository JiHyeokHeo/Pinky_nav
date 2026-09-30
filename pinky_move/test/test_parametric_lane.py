"""Motor-free regressions for lateral corners, S bends and real offset cusps."""
import cv2
import numpy as np
import pytest

from pinky_move.metric_lane import (
    MetricLaneTracker, arc_stations, center_from_pair, center_path_from_boundary,
    curve_match_error, fit_boundary, first_self_intersection, normal_offset,
    select_lookahead, supported_chain, pursuit,
)
from test_metric_lane import calibration


# Actual polygon from re-inference of /tmp/lane_distance_raw.png with the deployed
# model (320, conf=.55, iou=.7, retina masks). Not the in-flight result, and not
# reconstructed from the coloured debug overlay. Axle-to-corner measured ~19 cm.
CORNER_POLYGON = [
    [4,350],[4,386],[10,386],[11,387],[54,387],[55,388],[70,388],[71,389],
    [83,389],[84,390],[151,390],[152,391],[175,391],[176,390],[198,390],
    [199,391],[226,391],[227,392],[240,392],[241,391],[251,391],[252,392],
    [311,392],[312,393],[345,393],[346,394],[367,394],[368,395],[384,395],
    [385,396],[407,396],[408,397],[440,397],[441,398],[444,398],[445,397],
    [456,397],[457,398],[470,398],[471,399],[490,399],[491,398],[492,398],
    [493,399],[504,399],[505,400],[508,400],[509,401],[511,401],[512,402],
    [513,402],[514,403],[515,403],[525,413],[525,414],[528,417],[528,418],
    [531,421],[531,422],[533,424],[533,425],[537,429],[537,430],[550,443],
    [550,444],[558,452],[558,453],[561,456],[561,457],[565,461],[565,462],
    [566,463],[566,464],[568,466],[568,467],[569,468],[569,469],[571,471],
    [571,472],[574,475],[574,479],[638,479],[638,414],[635,414],[634,413],
    [633,413],[630,410],[629,410],[626,407],[625,407],[623,405],[622,405],
    [620,403],[619,403],[616,400],[615,400],[613,398],[612,398],[605,391],
    [604,391],[597,384],[596,384],[591,379],[590,379],[585,374],[584,374],
    [582,372],[581,372],[576,367],[575,367],[574,366],[573,366],[572,365],
    [571,365],[569,363],[568,363],[567,362],[566,362],[565,361],[564,361],
    [563,360],[560,360],[559,359],[541,359],[540,358],[539,358],[538,359],
    [502,359],[501,358],[489,358],[488,357],[469,357],[468,356],[421,356],
    [420,355],[312,355],[311,356],[298,356],[297,355],[283,355],[282,354],
    [258,354],[257,353],[246,353],[245,352],[200,352],[199,351],[183,351],[182,350]]


def corner_mask():
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [np.array(CORNER_POLYGON, np.int32)], 1)
    return mask


def test_measured_corner_preserves_right_identity_and_turns_left():
    tracker = MetricLaneTracker()
    for now in (0., .2, .4, .6):
        target = tracker.update([corner_mask()], calibration(), now, lookahead=.22,
                                lane_width=.154, path_min_m=.14, path_max_m=.48)
        assert target['visible_side'] == 'right'
        assert .08 < target['x_m'] < .18 and target['y_m'] > .04
        assert pursuit(target, .01) > 0
        assert first_self_intersection(target['center_path']) is None
        assert target['adaptive']  # Short observed support, not extrapolated .22.


@pytest.mark.parametrize('sign', [1, -1])
def test_lateral_corner_offsets_and_radial_targets(sign):
    angle = np.linspace(0., 1.8, 90)
    boundary = np.column_stack((.15+.18*np.sin(angle),
                                sign*(-.077+.18*(1-np.cos(angle)))))
    curve = fit_boundary(boundary)
    # Includes a mild decrease in x after the quarter turn. No x sorting.
    assert np.any(np.diff(curve[:, 0]) < 0)
    centre = normal_offset(curve, sign*.077)
    target, _ = select_lookahead(centre, .24)
    assert sign*target[1] > .01
    delta = centre-curve
    np.testing.assert_allclose(np.sum(delta*np.gradient(curve, axis=0), axis=1), 0., atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), .077, atol=1e-10)
    assert curve_match_error(curve, curve) < 1e-8
    assert np.isinf(curve_match_error(curve, curve[::-1]))


def test_short_forward_overlap_is_used_without_extrapolation():
    left = np.column_stack((np.linspace(.15, .25, 20), np.full(20, .08)))
    right = np.column_stack((np.linspace(.225, .35, 20), np.full(20, -.08)))
    centre, widths = center_from_pair(left, right)
    assert centre[0, 0] == pytest.approx(.225)
    assert centre[-1, 0] == pytest.approx(.25)
    target, adaptive = select_lookahead(centre, .3)
    np.testing.assert_allclose(target, centre[-1])
    assert adaptive
    np.testing.assert_allclose(widths, .16)


@pytest.mark.parametrize('start', [.24, .26])
def test_tiny_or_disjoint_forward_overlap_still_rejected(start):
    left = np.column_stack((np.linspace(.15, .25, 20), np.full(20, .08)))
    right = np.column_stack((np.linspace(start, .4, 20), np.full(20, -.08)))
    with pytest.raises(ValueError, match='x_overlap='):
        center_from_pair(left, right)


def test_three_normal_pairs_with_short_support_are_accepted():
    y = np.array([0., .012, .025])
    left = np.column_stack((np.full(3, .15), y))
    right = np.column_stack((np.full(4, .31), np.r_[y, .035]))
    centre, widths = center_from_pair(left, right)
    assert len(centre) == 3
    assert arc_stations(centre)[-1] == pytest.approx(.025)
    np.testing.assert_allclose(widths, .16)
    with pytest.raises(ValueError, match='normal_pairs=2'):
        center_from_pair(left[:2], right)
    with pytest.raises(ValueError, match='inconsistent normal lane pair'):
        center_from_pair(left * [1., .5], right * [1., .5])


def test_horizontal_pair_uses_normal_correspondence():
    y = np.linspace(-.04, .16, 60)
    left = np.column_stack((np.full(60, .15), y))
    right = np.column_stack((np.full(60, .31), y))
    centre, widths = center_from_pair(left, right)
    np.testing.assert_allclose(centre[:, 0], .23, atol=1e-9)
    np.testing.assert_allclose(widths, .16, atol=1e-9)
    assert arc_stations(centre)[-1] > .15
    with pytest.raises(ValueError):
        center_from_pair(right, left)  # Wrong side, not a lane pair.


def test_normal_pair_skips_leading_singleton_without_bridging_gap():
    left = np.column_stack((np.full(6, .15), [0., .035, .06, .08, .10, .12]))
    # Segmentation spur: y=.035 has two normal intersections, but the later
    # part is unambiguous. Old code stopped at its first one-point run.
    right = np.column_stack((np.full(4, .31), [-.01, .04, .03, .2]))
    centre, widths = center_from_pair(left, right)
    assert len(centre) == 4
    assert centre[0, 1] == pytest.approx(.06)
    assert centre[-1, 1] == pytest.approx(.12)
    np.testing.assert_allclose(widths, .16)
    with pytest.raises(ValueError, match='normal_pairs=2'):
        center_from_pair(left[:4], right)  # 1 + 2 is NOT 3 connected pairs.


@pytest.mark.parametrize('side', ['left', 'right'])
def test_s_bend_single_boundary_changes_steering_sign(monkeypatch, side):
    import pinky_move.metric_lane as module
    x = np.linspace(.14, .74, 120)
    centre = np.column_stack((x, .03*np.sin(2*np.pi*(x-.14)/.60)))
    sign = 1 if side == 'left' else -1
    boundary = normal_offset(centre, sign*.077)
    monkeypatch.setattr(module, 'floor_curves', lambda *args, **kwargs: [boundary])
    for lookahead, turn_sign in ((.25, 1), (.60, -1)):
        result = MetricLaneTracker().update([object()], None, 0., lookahead=lookahead,
                       lane_width=.154, path_min_m=.10, path_max_m=.8)
        assert result['visible_side'] == side
        assert turn_sign*pursuit(result, .01) > 0


def test_actual_cusp_at_near_end_stops_but_distant_prefix_is_usable():
    angle = np.linspace(0., np.pi/2, 60)
    tight = np.column_stack((.2+.02*np.sin(angle), .02*(1-np.cos(angle))))
    with pytest.raises(ValueError, match='folds back'):
        center_path_from_boundary(tight, .077)
    straight = np.column_stack((np.linspace(.08, .198, 80), np.zeros(80)))
    bent = np.vstack((straight, tight))
    path, warning = center_path_from_boundary(bent, .077)
    assert warning and path[-1, 0] < .205
    assert arc_stations(path)[-1] >= .06
    assert first_self_intersection(path) is None


def test_crossing_and_disjoint_support_never_become_shortcuts():
    bow = np.array([[.1,0],[.2,.1],[.3,0],[.2,0],[.1,.1],[.3,.1]])
    with pytest.raises(ValueError, match='intersection'):
        fit_boundary(bow)
    p = np.array([[.15,0],[.2,0],[.5,0],[.2,.1],[.3,.1]])
    np.testing.assert_array_equal(supported_chain(p,.14,.48), p[:2])
