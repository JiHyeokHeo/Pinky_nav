"""Failure-only S-curve extraction and temporal identity, without motors."""
import json

import cv2
import numpy as np
import pytest

from test_lane_edge_replays import ROOT, recorded, update
from pinky_move.metric_lane import (
    MetricLaneTracker, connected_floor_curve, floor_curves,
)


def calibration():
    return json.loads((ROOT/'config/robot_floor_calibration.json').read_text())


def test_recorded_s_mask_has_connected_observed_curve():
    curves = floor_curves(recorded('e50d4096a2902bbe'), calibration(),
                          path_min_m=.14, path_max_m=.48)
    assert len(curves) == 1
    assert np.isfinite(curves[0]).all()
    assert np.max(np.linalg.norm(np.diff(curves[0], axis=0), axis=1)) < .02


def test_recorded_overlap_requires_three_fresh_frames():
    masks = recorded('615202e1d2694f17')
    tracker = MetricLaneTracker(minimum_lane_width_m=.15)
    for now, count in ((1., 1), (1.2, 2)):
        with pytest.raises(ValueError, match=f'confirming near-pixel right: {count}/3'):
            update(tracker, masks, now)
    target = update(tracker, masks, 1.4)
    assert target['visible_side'] == 'right'
    assert np.isfinite([target['x_m'], target['y_m']]).all()


def test_repeated_timestamp_gap_and_missing_frame_do_not_confirm():
    masks = recorded('615202e1d2694f17')
    tracker = MetricLaneTracker()
    for now in (1., 1., 3.):
        with pytest.raises(ValueError, match='1/3'):
            update(tracker, masks, now)
    with pytest.raises(ValueError, match='no boundaries'):
        update(tracker, [], 3.1)
    with pytest.raises(ValueError, match='1/3'):
        update(tracker, masks, 3.2)


def test_disconnected_stripes_and_closed_loop_are_not_guessed():
    mask = np.zeros((480, 640), np.uint8)
    cv2.line(mask, (10, 330), (630, 310), 1, 12)
    cv2.line(mask, (10, 410), (630, 390), 1, 12)
    assert connected_floor_curve(mask, calibration()) == []
    mask[:] = 0
    cv2.circle(mask, (320, 320), 90, 1, 12)
    assert connected_floor_curve(mask, calibration()) == []
