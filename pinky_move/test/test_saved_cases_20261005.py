"""Saved-photo masks plus synthetic history: never a driving success claim."""
import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_lane_dataset import masks_for
from test_metric_lane import calibration, masks_for_lines
from pinky_move.metric_lane import MetricLaneTracker, equivalent_boundary

ROOT = Path(__file__).resolve().parents[1]


def saved_masks(case):
    data = json.loads((ROOT/'reports/cases'/case/'replay_masks.json').read_text())
    return masks_for(data['instances'], (480, 640))


def test_saved_duplicate_ground_boundaries_confirm_and_continue():
    tracker = MetricLaneTracker(tracking_gap_s=2.)
    masks = saved_masks('20261005_stopped_side_ambiguous')
    assert len(masks) == 2
    for now in (1., 1.2):
        with pytest.raises(ValueError, match='confirming unique near boundary left'):
            tracker.update(masks, calibration(), now, lane_width=.154,
                           lookahead=.22, path_min_m=.14, path_max_m=.48)
    for now in (1.4, 1.6, 1.8):
        target = tracker.update(masks, calibration(), now, lane_width=.154,
                                lookahead=.22, path_min_m=.14, path_max_m=.48)
        assert target['visible_side'] == 'left' and target['inferred']
        assert np.isfinite([target['x_m'], target['y_m']]).all()


def test_saved_second_photo_cold_start_is_not_temporal_failure():
    # Raw/debug frames differ by 5.33 seconds. This photograph alone cannot
    # reconstruct the original rejected inferred target or the stored target.
    tracker = MetricLaneTracker()
    target = tracker.update(saved_masks('20261005_005730_edge02'), calibration(),
                            1., lane_width=.154, lookahead=.22,
                            path_min_m=.14, path_max_m=.48)
    assert target['boundary_count'] == 2


def test_same_boundary_target_jump_confirms_instead_of_permanent_stop():
    tracker = MetricLaneTracker(tracking_gap_s=2.)
    mask = masks_for_lines()[1]
    initial = tracker.update([mask], calibration(), 1., lane_width=.16)
    tracker.previous = dict(initial, y_m=initial['y_m']+.10)
    for now in (1.2, 1.4):
        with pytest.raises(ValueError, match='inferred target discontinuity'):
            tracker.update([mask], calibration(), now, lane_width=.16)
        assert tracker.previous['y_m'] == initial['y_m']+.10
    target = tracker.update([mask], calibration(), 1.6, lane_width=.16)
    assert target['target_reconfirmed_frames'] == 3
    assert target['y_m'] == pytest.approx(initial['y_m'])
    assert tracker.update([mask], calibration(), 1.8, lane_width=.16)['y_m'] == target['y_m']


@pytest.mark.parametrize('interruption', ['empty', 'opposite', 'timestamp', 'gap'])
def test_target_reconfirmation_requires_fresh_uninterrupted_identity(interruption):
    tracker = MetricLaneTracker(tracking_gap_s=2.)
    masks = masks_for_lines()
    initial = tracker.update([masks[1]], calibration(), 1., lane_width=.16)
    tracker.previous = dict(initial, y_m=.10)
    with pytest.raises(ValueError, match='inferred target discontinuity'):
        tracker.update([masks[1]], calibration(), 1.2, lane_width=.16)
    if interruption in ('empty', 'opposite'):
        with pytest.raises(ValueError):
            tracker.update([] if interruption == 'empty' else [masks[0]],
                           calibration(), 1.3, lane_width=.16)
        assert tracker.target_reconfirmation is None
    else:
        # Equal timestamp or a long gap cannot increase the confirmation count.
        target = dict(initial)
        with pytest.raises(ValueError, match='inferred target discontinuity'):
            tracker._validate_target_continuity(target, 1.2 if interruption == 'timestamp' else 4., 'right')
        assert tracker.target_reconfirmation['count'] == 1


def test_common_entry_with_different_branch_is_not_duplicate():
    a = np.column_stack([np.linspace(.15,.4,60), np.zeros(60)])
    b = a.copy()
    b[35:,1] = np.linspace(0.,.05,25)
    assert not equivalent_boundary(a, b)
