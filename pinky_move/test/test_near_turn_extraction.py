"""Real stopped-turn mask: axis scans must not discard the visible near leg."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import pinky_move.metric_lane as lane


def scene():
    root=Path(__file__).parents[1]
    data=json.loads((root/'test/fixtures/lane_edges/near_turn_20261004.json').read_text())
    cal=json.loads((root/'config/robot_floor_calibration.json').read_text())
    w,h=data['image_size']; mask=np.zeros((h,w),np.uint8)
    cv2.fillPoly(mask,[np.asarray(data['polygon'],np.int32)],1)
    return mask,cal


def test_real_mask_selects_connected_near_approach_not_far_exit():
    mask,cal=scene()
    row=lane._axis_floor_curves([mask],cal,2.,False)[0]
    chosen=lane.floor_curves([mask],cal,2.,.14,.48)[0]
    assert row[0,0]>.24
    assert .13<chosen[0,0]<.15
    assert row[0,0]-chosen[0,0]>.10
    near=lane.supported_chain(chosen,.14,.48)
    assert lane.arc_stations(near)[-1]>.25
    assert np.isfinite(lane.fit_boundary(near)).all()


def test_real_scene_produces_repeatable_current_centre_target():
    from pinky_move.lane_corner import CornerPolicy
    mask,cal=scene(); tracker=lane.MetricLaneTracker(); policy=CornerPolicy()
    for now in (1.,1.2,1.4):
        target=tracker.update([mask],cal,now,lookahead=.22,lane_width=.168,
                              path_min_m=.14,path_max_m=.48)
        assert target['visible_side']=='left' and not target.get('held',False)
        assert target['x_m']>.05 and len(target['center_path'])>=5
        assert policy.update(tracker.last_observation,target,now,np.zeros(3),.22,.25) is not None


@pytest.mark.parametrize('case',['absent','short','far'])
def test_invalid_connected_candidate_does_not_replace_axis(monkeypatch,case):
    mask,cal=scene()
    row=lane._axis_floor_curves([mask],cal,2.,False)[0]
    if case=='absent':replacement=[]
    elif case=='short':replacement=[np.column_stack((np.linspace(.14,.15,10),np.zeros(10)))]
    else:replacement=[row]
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *args:replacement)
    actual=lane.floor_curves([mask],cal,2.,.14,.48)[0]
    np.testing.assert_array_equal(actual,row)


def test_existing_near_axis_curve_does_not_trigger_connected_replacement(monkeypatch):
    mask,cal=scene()
    near=np.column_stack((np.linspace(.14,.4,50),np.full(50,.08)))
    monkeypatch.setattr(lane,'_axis_floor_curves',lambda masks,cal,maxdist,column:[] if column else [near])
    monkeypatch.setattr(lane,'connected_floor_curve',lambda *a:pytest.fail('unnecessary replacement'))
    np.testing.assert_array_equal(lane.floor_curves([mask],cal,2.,.14,.48)[0],near)
