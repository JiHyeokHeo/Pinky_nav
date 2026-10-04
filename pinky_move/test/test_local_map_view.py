import numpy as np
from pinky_move.local_map_view import map_pixels, visible_curves, render_local_map
from pinky_move.lane_history import LaneHistory, rotation
from pinky_move.lane_history import SpatialLaneArchive


def test_pixel_axes_and_metric_scale():
    np.testing.assert_array_equal(map_pixels([[0,0],[.1,0],[0,.1]]),
                                  [[240,240],[240,220],[220,240]])


def test_accumulated_world_boundary_uses_current_capture_pose():
    h=LaneHistory(max_frames=50,max_age_s=5.)
    curve=np.column_stack((np.linspace(.1,.5,30),np.full(30,.15)))
    h.remember(1_000_000_000,np.zeros(3),dict(side='left',curve=curve))
    pose=np.array([.05,0.,.2])
    side,local,age=visible_curves(h.frames,2_000_000_000,pose)[0]
    assert side=='left' and age==1.
    np.testing.assert_allclose(local,(curve-pose[:2])@rotation(pose[2]))
    assert visible_curves(h.frames,7_000_000_000,pose)
    assert not visible_curves(h.frames,12_000_000_000,pose)
    assert not visible_curves(h.frames,0,pose)
    assert not visible_curves(h.frames,2_000_000_000,None)


def test_render_is_bounded_and_does_not_mutate_observation():
    curve=np.column_stack((np.linspace(.1,.5,30),np.full(30,.15)))
    before=curve.copy()
    frames=[(1_000_000_000,dict(left=curve))]
    frame=render_local_map(frames,1_000_000_000,np.zeros(3),
                           dict(x_m=.2,y_m=0.,center_path=[[.1,0],[.2,0]]),
                           dict(mode='ENTRY',pivot_world=[.3,0]))
    assert frame.shape==(480,480,3) and frame.dtype==np.uint8
    np.testing.assert_array_equal(curve,before)
    assert render_local_map(frames,1_000_000_000,None).shape==frame.shape


def test_archive_survives_time_expiry_and_draws_previous_lane_behind_robot():
    archive = SpatialLaneArchive()
    curve = np.column_stack((np.linspace(.1,.4,30), np.full(30,.15)))
    archive.remember(1_000_000_000, np.zeros(3), dict(side='left', curve=curve))
    pose = np.array([.6,0.,0.])
    archive.remember(60_000_000_000, pose, None)
    points = archive.snapshot()['left']
    assert len(points) > 0
    assert np.all(points[:,0]-pose[0] < 0)
    canvas = render_local_map([], 60_000_000_000, pose, archive=archive.snapshot())
    pixel = map_pixels([(points[10]-pose[:2])])[0]
    np.testing.assert_array_equal(canvas[pixel[1],pixel[0]], [30,110,30])
    archive.remember(61_000_000_000, np.array([1.5,0.,0.]), None)
    assert len(archive.snapshot()['left']) == 0


def test_archive_deduplicates_repeated_frames_and_only_keeps_spatial_box():
    archive = SpatialLaneArchive()
    curve = np.column_stack((np.linspace(.1,.4,30), np.full(30,.15)))
    obs = dict(side='right', curve=curve, other=curve+[0.,-.3])
    for stamp in range(1,101):
        archive.remember(stamp, np.zeros(3), obs)
    assert len(archive.snapshot()['right']) == 30
    assert len(archive.snapshot()['left']) == 30
    archive.remember(101, np.array([0.,0.,np.pi]), None)
    assert len(archive.snapshot()['right']) == 30
    archive.clear()
    assert not len(archive.snapshot()['right'])
