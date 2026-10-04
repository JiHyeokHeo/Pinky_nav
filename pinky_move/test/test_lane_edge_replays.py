"""Recorded masks, motor-free geometry checks; repeated frames are synthetic."""
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_lane_dataset import masks_for
from pinky_move.metric_lane import MetricLaneTracker, distinct_near_masks
from pinky_move.lane_corner import CornerConfig, CornerPolicy


ROOT = Path(__file__).resolve().parents[1]


def recorded(key):
    data = json.loads((ROOT/'test/fixtures/lane_edges'/f'{key}.json').read_text())
    return masks_for(data['instances'], (480, 640))


def update(tracker, masks, now=1.):
    calibration = json.loads((ROOT/'config/robot_floor_calibration.json').read_text())
    return tracker.update(masks, calibration, now, lookahead=.22, lane_width=.154,
                          path_min_m=.14, path_max_m=.48)


@pytest.mark.parametrize('key', ['f6376a93bf530b01', 'e9479a034914c80d'])
def test_recorded_same_boundary_duplicates_keep_real_pair(key):
    masks = recorded(key)
    assert len(masks) == 3
    assert len(distinct_near_masks(masks)) == 2
    target = update(MetricLaneTracker(minimum_lane_width_m=.15), masks)
    assert not target['inferred']


def test_near_overlap_does_not_erase_different_far_branch():
    left = np.zeros((480, 640), np.uint8)
    left[320:, 100:110] = 1
    other = left.copy()
    left[:300, 100:110] = 1
    other[:300, 220:230] = 1
    right = np.fliplr(left).copy()
    assert len(distinct_near_masks([left, left.copy(), other, right])) == 3


def test_recorded_endpoint_hook_prefers_long_observed_stripe():
    target = update(MetricLaneTracker(minimum_lane_width_m=.15),
                    recorded('eb742497f3df51c2'))
    assert target['inferred']
    assert np.isfinite([target['x_m'], target['y_m']]).all()


@pytest.mark.parametrize('key', [
    '6d25c6f46a0bca2c', '8867606cef0c1698', 'a15d9daefdb3aa3b',
    '45ac22233aa8fafa', '230e5006b70075fc', '802d3555c734b0e1',
])
def test_true_corner_keeps_fold_guard_and_requires_confirmation(key):
    tracker = MetricLaneTracker(minimum_lane_width_m=.15)
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    masks = recorded(key)
    # Zero odometry and repeated still image only exercise policy integration;
    # they do not establish independent observations or real driving safety.
    for index, now in enumerate((1., 1.2, 1.4)):
        with pytest.raises(ValueError, match='center_path curve folds back'):
            update(tracker, masks, now)
        target = policy.update(tracker.last_observation, None, now,
                               np.zeros(3), .22, .25)
        if index < 2:
            assert target is None
    assert target['corner_staged']
    assert not target.get('corner_stationary')
    assert policy.update(None, None, 1.6, np.zeros(3), .22, .25) is None
