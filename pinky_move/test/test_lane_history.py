import numpy as np
from pinky_move.lane_history import LaneHistory, rotation
from test_lane_controller import controller, detection
from types import SimpleNamespace
from std_msgs.msg import Header


def observed(side='left', y=.15):
    return dict(side=side,curve=np.column_stack((np.linspace(.12,.5,60),np.full(60,y))))


def test_history_compensates_translation_and_rotation_and_keeps_role():
    history=LaneHistory()
    obs=observed()
    history.remember(1_000_000_000,np.zeros(3),obs)
    history.remember(1_100_000_000,np.zeros(3),obs)
    pose=np.array([.04,0.,.4])
    current=(obs['curve']-pose[:2])@rotation(pose[2])
    assert history.associate_curves([(0,current)],1_500_000_000,pose)==(0,'left')
    assert history.associate_curves([(0,current)],1_500_000_000,None) is None


def test_history_needs_two_distinct_frames_and_expires():
    history=LaneHistory()
    obs=observed()
    history.remember(1_000_000_000,np.zeros(3),obs)
    history.remember(1_000_000_000,np.zeros(3),obs)
    assert len(history.frames)==1
    assert history.associate_curves([(0,obs['curve'])],1_100_000_000,np.zeros(3)) is None
    history.remember(1_200_000_000,np.zeros(3),obs)
    assert history.associate_curves([(0,obs['curve'])],3_200_000_001,np.zeros(3)) is None


def test_history_never_guesses_between_duplicate_candidates_or_uses_empty_input():
    history=LaneHistory(); obs=observed()
    for stamp in (1_000_000_000,1_100_000_000):
        history.remember(stamp,np.zeros(3),obs)
    assert history.associate_curves([(0,obs['curve']),(1,obs['curve'])],1_200_000_000,np.zeros(3)) is None
    assert history.associate_curves([],1_200_000_000,np.zeros(3)) is None


def test_history_is_bounded_and_reset_clears_world_coordinates():
    history=LaneHistory()
    for stamp in range(1,20):
        history.remember(stamp,np.zeros(3),observed())
    assert len(history.frames)==5
    history.clear()
    assert not history.frames


def test_history_hint_is_applied_before_exactly_one_geometry_update(controller):
    node = controller
    node.robot_calibration = {'present': True}
    node.image_identity = SimpleNamespace(match=lambda *a: None, commit=lambda *a: None)
    node.lane_history = SimpleNamespace(match=lambda *a: (0, 'left'))
    node._corner_pose_for_frame = lambda *a: np.zeros(3)
    node._image_roles_from_target = lambda *a: {}
    node._update_turn_observation = lambda *a: None
    calls = []
    def update(*args):
        calls.append(node.image_match)
        return 0
    node._update_lane_command = update
    node._process_result(np.zeros((100,200,3),np.uint8), detection(),
                         node.safety_clock.now(), Header())
    assert calls == [(0, 'left')]
    assert node.history_match == (0, 'left')
