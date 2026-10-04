"""Detached upper detections must not take over a verified near boundary."""
import pytest
from test_metric_lane import calibration, masks_for_lines
from pinky_move.metric_lane import MetricLaneTracker


@pytest.mark.parametrize('reverse',[False,True])
def test_far_upper_stripe_is_context_not_new_pair(reverse):
    c=calibration(); left,right=masks_for_lines()
    far=left.copy();far[280:]=0
    tracker=MetricLaneTracker()
    first=tracker.update([right],c,1.,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    masks=[right,far] if not reverse else [far,right]
    result=tracker.update(masks,c,1.2,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    assert result['inferred'] and result['visible_side']=='right'
    assert result['context_only_indices']==([1] if not reverse else [0])
    assert result['source_mask_index']==(0 if not reverse else 1)
    assert result['y_m']==pytest.approx(first['y_m'])


def test_real_near_opposite_line_is_not_filtered():
    c=calibration(); masks=masks_for_lines(); tracker=MetricLaneTracker()
    tracker.update([masks[1]],c,1.,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    result=tracker.update(masks,c,1.2,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    assert tracker.context_only_indices==[]
    assert result['pair_transition_count']==1


def test_missing_near_anchor_does_not_replay_old_goal():
    c=calibration(); left,right=masks_for_lines();far=left.copy();far[280:]=0
    tracker=MetricLaneTracker()
    tracker.update([right],c,1.,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    try:
        result=tracker.update([far],c,1.2,lookahead=.22,lane_width=.16,path_min_m=.14,path_max_m=.48)
    except ValueError:
        result=None
    assert tracker.context_only_indices==[]
    assert result is None  # Missing right stripe cannot become the far left stripe.
