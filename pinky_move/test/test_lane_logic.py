"""Unit tests for segmentation-based lane geometry."""


import numpy as np
from pinky_move.lane_logic import (
    crossline_is_close,
    estimate_lane_center,
    sample_mask_x,
    select_near_lane,
    steering_command,
)
import pytest
from pinky_move.lane_logic import near_priority_command


def test_distant_s_bend_does_not_turn_centered_robot():
    for far in [10, 50, 150, 190]:
        angular, preview = near_priority_command(100, far, 200, .85, .65)
        assert angular == 0
        assert preview > 0


def test_near_error_controls_direction_not_opposite_far_bend():
    right, _ = near_priority_command(130, 50, 200, .85, .65)
    left, _ = near_priority_command(70, 150, 200, .85, .65)
    assert right < 0 < left


def test_near_deadband_and_limit():
    assert near_priority_command(103, 170, 200, .85, .65)[0] == 0
    assert near_priority_command(200, 200, 200, 10., .15)[0] == -.15
from pinky_move.lane_logic import consistent_single_far_center


def test_curved_left_boundary_keeps_left_side_at_far_row():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[75:82, 48:53] = 1
    mask[55:62, 138:143] = 1
    # Far boundary crossed the image centre, but is still the LEFT boundary.
    for _ in range(8):
        assert consistent_single_far_center([mask], 100., .78, .58, 60.) == 170.


def test_curved_right_boundary_keeps_right_side():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[75:82, 148:153] = 1
    mask[55:62, 58:63] = 1
    assert consistent_single_far_center([mask], 100., .78, .58, 60.) == 30.


def test_far_only_mask_does_not_invent_boundary_side():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[55:62, 58:63] = 1
    assert consistent_single_far_center([mask], 100., .78, .58, 60.) is None


def test_sample_mask_uses_requested_horizontal_band():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[74:81, 28:33] = 1
    assert sample_mask_x(mask, 0.78, 0.06) == pytest.approx(30.0)


def test_two_boundaries_define_lane_center():
    estimate = estimate_lane_center(
        [60.0, 140.0], 200, 100.0, 80.0)
    assert estimate.center_x == pytest.approx(100.0)
    assert estimate.lane_width == pytest.approx(80.0)
    assert estimate.boundary_count == 2


def test_one_left_boundary_uses_remembered_lane_width():
    estimate = estimate_lane_center(
        [60.0], 200, 100.0, 80.0)
    assert estimate.center_x == pytest.approx(100.0)
    assert estimate.boundary_count == 1


def test_pair_nearest_previous_path_wins_with_extra_distant_lane():
    estimate = estimate_lane_center(
        [15.0, 65.0, 145.0], 200, 105.0, 80.0)
    assert estimate.center_x == pytest.approx(105.0)
    assert estimate.lane_width == pytest.approx(80.0)


def test_s_curve_lookahead_turns_toward_far_lane_center():
    right_curve = steering_command(100, 135, 200, 0.8, 1.2, 0.7)
    left_curve = steering_command(100, 65, 200, 0.8, 1.2, 0.7)
    assert right_curve < 0.0
    assert left_curve > 0.0


def test_crossline_only_triggers_when_near_bottom():
    distant = np.zeros((100, 200), dtype=np.uint8)
    distant[45:55, 20:180] = 1
    close = np.zeros((100, 200), dtype=np.uint8)
    close[70:76, 20:180] = 1
    assert not crossline_is_close(distant, 0.68)
    assert crossline_is_close(close, 0.68)


def test_pair_above_single_boundary_is_preferred():
    left = np.zeros((100, 200), dtype=np.uint8)
    right = np.zeros_like(left)
    left[50:100, 48:53] = 1
    right[50:74, 148:153] = 1
    estimate, ratio, count = select_near_lane(
        [left, right], 200, 100, 100)
    assert estimate.boundary_count == 2
    assert estimate.center_x == pytest.approx(100)
    assert ratio == pytest.approx(0.74)
    assert count == 2


def test_single_boundary_still_used_if_pair_only_exists_outside_near_range():
    left = np.zeros((100, 200), dtype=np.uint8)
    right = np.zeros_like(left)
    left[50:100, 48:53] = 1
    right[50:60, 148:153] = 1
    estimate, ratio, count = select_near_lane(
        [left, right], 200, 100, 100)
    assert estimate.boundary_count == 1
    assert ratio == pytest.approx(0.78)
    assert count == 1


def test_two_overlapping_instances_do_not_create_a_fake_pair():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[50:100, 48:53] = 1
    estimate, _, count = select_near_lane([mask, mask], 200, 100, 100)
    assert estimate.boundary_count == 1
    assert count == 2


def test_empty_near_band_has_no_usable_boundary():
    estimate, _, count = select_near_lane([], 200, 100, 100)
    assert estimate is None
    assert count == 0
