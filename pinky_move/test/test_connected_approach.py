"""Replay the long S mask that defeated a whole-curve polynomial fit."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import pinky_move.metric_lane as lane


def scene():
    root=Path(__file__).parents[1]
    data=json.loads((root/'test/fixtures/lane_edges/long_s_prefix_20261004.json').read_text())
    cal=json.loads((root/'config/robot_floor_calibration.json').read_text())
    w,h=data['image_size'];mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.asarray(data['polygon'],np.int32)],1)
    return mask,cal


def test_long_s_retains_verified_near_prefix():
    mask,cal=scene()
    full=lane.supported_chain(lane.connected_floor_curve(mask,cal)[0],.14,.48)
    fit=lane.fit_boundary(full)
    error=np.linalg.norm(full-lane.nearest_on_chain(full,fit)[0],axis=1)
    assert lane.arc_stations(full)[-1]>.8
    assert np.sqrt(np.mean(error**2))>.006
    chosen=lane.floor_curves([mask],cal,2.,.14,.48)[0]
    assert chosen[0,0]<.15
    assert .20<lane.arc_stations(chosen)[-1]<.28
    # All points belong to the FIRST contiguous observation, not a new spline
    # bridge from the body-side segment to the far outgoing leg.
    np.testing.assert_array_equal(chosen,full[:len(chosen)])


def test_real_long_s_produces_current_target_and_preserves_left_identity():
    from pinky_move.lane_corner import CornerPolicy
    mask,cal=scene();tracker=lane.MetricLaneTracker();policy=CornerPolicy()
    for now in (1.,1.2,1.4):
        result=tracker.update([mask],cal,now,lookahead=.22,lane_width=.168,
                              path_min_m=.14,path_max_m=.48)
        assert result['visible_side']=='left' and not result.get('held',False)
        assert policy.update(tracker.last_observation,result,now,np.zeros(3),.22,.25) is not None


def test_prefix_growth_stops_at_first_failed_section(monkeypatch):
    points=np.column_stack((np.linspace(.14,.6,400),np.zeros(400)))
    original=lane.fit_boundary;attempts=[]
    def fail_middle(prefix,degree):
        length=lane.arc_stations(prefix)[-1];attempts.append(length)
        if .10<length<.15:raise ValueError('unreliable section')
        return original(prefix,degree)
    monkeypatch.setattr(lane,'fit_boundary',fail_middle)
    chosen,_,_,length=lane.fit_connected_approach(points)
    assert .075<length<.081 and len(attempts)==2
    np.testing.assert_array_equal(chosen,points[:len(chosen)])


def test_missing_near_segment_cannot_be_bridged():
    points=np.column_stack((np.linspace(.14,.5,100),np.zeros(100)))
    points[3:,0]+=.08
    with pytest.raises(ValueError,match='no supported connected approach'):
        lane.fit_connected_approach(points)


@pytest.mark.parametrize('points',[np.empty((0,2)),np.zeros((4,2)),np.full((20,2),np.nan)])
def test_invalid_prefix_is_rejected(points):
    with pytest.raises(ValueError):lane.fit_connected_approach(points)
