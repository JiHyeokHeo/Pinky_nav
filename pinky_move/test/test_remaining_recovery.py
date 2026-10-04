"""Recorded remaining failures; no ROS publishing or physical driving."""
import numpy as np
import pytest
from test_lane_edge_replays import recorded, update
from pinky_move.metric_lane import MetricLaneTracker, short_curve_match_error


@pytest.mark.parametrize('key', ['ca83b0c46461ad18', '65eca2c528317038',
                               '25d165aed3e2368d', 'd1281956c1f01266'])
def test_s_bend_uses_only_valid_observed_prefix(key):
    result = update(MetricLaneTracker(minimum_lane_width_m=.15), recorded(key))
    assert result['inferred']
    assert result['center_path_warning'] == 'S-bend observed prefix refit'
    assert np.isfinite(result['center_path']).all()


@pytest.mark.parametrize('key', ['64b29be82a3b147a', 'd7d40278b971d792'])
def test_relative_pair_and_individual_extraction_are_order_independent(key):
    masks = recorded(key)
    first = update(MetricLaneTracker(minimum_lane_width_m=.15), masks)
    second = update(MetricLaneTracker(minimum_lane_width_m=.15), masks[::-1])
    assert not first['inferred']
    assert .15 < first['normal_width_m'] < .21
    np.testing.assert_allclose([first['x_m'], first['y_m']],
                               [second['x_m'], second['y_m']])


def test_short_bootstrap_is_not_required_to_match_itself_again():
    tracker = MetricLaneTracker(minimum_lane_width_m=.15)
    masks = recorded('3906545c0cf63166')
    first = update(tracker, masks)
    assert first['visible_side'] == 'left'
    second = update(tracker, masks, 1.2)
    assert second['visible_side'] == 'left'


def test_short_tracking_requires_whole_curve_not_just_one_close_end():
    a = np.column_stack([np.linspace(.2,.245,20), np.full(20,.1)])
    assert short_curve_match_error(a,a) == 0.
    b = a.copy(); b[:,1] += np.linspace(0,.01,20)
    assert np.isinf(short_curve_match_error(a,b))
    assert np.isinf(short_curve_match_error(a,a[::-1]))


@pytest.mark.parametrize('key', ['9114047c21ad4d24', '11435fc1441f581b'])
def test_distant_unpaired_mask_requires_three_near_confirmations(key):
    masks = recorded(key)
    tracker = MetricLaneTracker(minimum_lane_width_m=.15)
    for now, count in ((1.,1), (1.2,2)):
        with pytest.raises(ValueError, match=f'confirming unique near boundary right: {count}/3'):
            update(tracker, masks, now)
    target = update(tracker, masks, 1.4)
    assert target['inferred'] and target['visible_side'] == 'right'
    assert target['pair_fallback'] == 'temporally confirmed near boundary'


def test_near_pair_confirmation_gap_does_not_accumulate():
    tracker = MetricLaneTracker()
    masks = recorded('9114047c21ad4d24')
    for now in (1., 1., 3.):
        with pytest.raises(ValueError, match='1/3'):
            update(tracker, masks, now)
