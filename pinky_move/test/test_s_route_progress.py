"""No motors: S targets progress with odometry, never detector endpoint jumps."""
import numpy as np
import pytest
from pinky_move.lane_corner import CornerPolicy, CornerConfig


def path():
    x = np.linspace(.1, .9, 120)
    return np.column_stack((x, .035*np.sin((x-.1)*2*np.pi/.8)))


def observation():
    return dict(side='right', width=.22)


def target(world, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    local = (world-pose[:2])@np.array([[c,-s],[s,c]])
    return dict(center_path=local.tolist(), x_m=.7, y_m=.2, boundary_count=1)


def seeded():
    p = CornerPolicy(CornerConfig(staged_turn=True))
    pose = np.zeros(3)
    result = p._track_s_route(target(path(), pose), observation(), pose, 1., .22)
    return p, result


def test_stationary_detector_updates_cannot_jump_goal():
    p, first = seeded()
    initial = p.s_route['path'].copy()
    for now in (1.2, 1.4, 1.6):
        noisy = path()+[0., .003]
        result = p._track_s_route(target(noisy, np.zeros(3)), observation(), np.zeros(3), now, .22)
        np.testing.assert_allclose([result['x_m'],result['y_m']], [first['x_m'],first['y_m']], atol=1e-9)
        np.testing.assert_allclose(p.s_route['path'][:len(initial)], initial, atol=1e-9)


def test_target_advances_from_actual_robot_motion():
    p, first = seeded()
    pose = np.array([.025, 0., .05])
    result = p._track_s_route(target(path(), pose), observation(), pose, 1.2, .22)
    c,s = np.cos(pose[2]),np.sin(pose[2])
    goal = np.array([[c,-s],[s,c]])@np.array([result['x_m'], result['y_m']])+pose[:2]
    advance = np.linalg.norm(goal-[first['x_m'],first['y_m']])
    assert 0 < advance <= .03


@pytest.mark.parametrize('case', ['far_leg','shift','reverse','gap','jump','side','width','missing_pose'])
def test_bad_update_does_not_replace_pending_bend(case):
    p, _ = seeded()
    before = p.s_route['path'].copy()
    fresh, pose, now, obs = path(), np.zeros(3), 1.2, observation()
    if case == 'far_leg': fresh = fresh[65:]
    if case == 'shift': fresh = fresh+[0., .06]
    if case == 'reverse': fresh = fresh[::-1]
    if case == 'gap': now = 3.1
    if case == 'jump': pose[0] = .3
    if case == 'side': obs['side'] = 'left'
    if case == 'width': obs['width'] = .5
    candidate = target(fresh, pose)
    if case == 'missing_pose': pose = None
    with pytest.raises(ValueError):
        p._track_s_route(candidate, obs, pose, now, .22)
    np.testing.assert_array_equal(p.s_route['path'], before)


def test_extension_keeps_original_untraversed_geometry():
    p, _ = seeded()
    before = p.s_route['path'].copy()
    extra = np.column_stack((np.linspace(.91,1.1,30), np.linspace(.002,.04,30)))
    p._track_s_route(target(np.vstack((path(),extra)),np.zeros(3)),
                     observation(),np.zeros(3),1.2,.22)
    assert p.s_route['path'][-1,0] > .95
    np.testing.assert_allclose(p.s_route['path'][:len(before)],before,atol=1e-9)


def test_odometry_projection_cannot_jump_to_returning_leg():
    p = CornerPolicy()
    world = np.vstack((np.column_stack((np.linspace(.1,.4,40), np.zeros(40))),
                       np.column_stack((np.full(20,.4), np.linspace(.005,.1,20))),
                       np.column_stack((np.linspace(.395,.01,40), np.full(40,.1)))))
    p._track_s_route(target(world,np.zeros(3)),observation(),np.zeros(3),1.,.22)
    # Local progress projection is limited to physical travel even if a later
    # return branch would be the globally nearest point to the robot.
    initial = p.s_route['path'].copy()
    pose = np.array([.001,.02,0.])
    # The end of the returning leg is nearer than the entry, but untraversed.
    assert np.linalg.norm(world[-1]-pose[:2]) < np.linalg.norm(world[0]-pose[:2])
    p._track_s_route(target(world,pose),observation(),pose,1.2,.22)
    assert np.linalg.norm(p.s_route['path'][0]-initial[0]) <= .036
