import pytest
from test_lane_edge_replays import recorded, update
from pinky_move.metric_lane import MetricLaneTracker


def test_two_second_identity_grace_accepts_new_matching_observation():
    tracker = MetricLaneTracker(minimum_lane_width_m=.15, tracking_gap_s=2.)
    masks = recorded('eb742497f3df51c2')
    update(tracker, masks, 1.)
    assert update(tracker, masks, 2.9)['inferred']


def test_identity_older_than_two_seconds_requires_reacquisition():
    tracker = MetricLaneTracker(minimum_lane_width_m=.15, tracking_gap_s=2.)
    masks = recorded('eb742497f3df51c2')
    update(tracker, masks, 1.)
    with pytest.raises(ValueError, match='reference tracking gap'):
        update(tracker, masks, 3.01)


def test_empty_masks_do_not_become_a_new_target_during_grace():
    tracker = MetricLaneTracker(tracking_gap_s=2.)
    update(tracker, recorded('eb742497f3df51c2'), 1.)
    with pytest.raises(ValueError, match='no boundaries'):
        update(tracker, [], 2.)
