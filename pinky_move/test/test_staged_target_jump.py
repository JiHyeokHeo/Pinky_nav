"""Only confirmed staged control may ignore an ordinary target jump."""
import numpy as np
import pytest
from types import SimpleNamespace
from test_lane_controller import controller
from test_staged_corner import prepared
from test_lane_corner import observation
from pinky_move.lane_corner import CornerConfig, CornerPolicy


@pytest.mark.parametrize('case', ['valid','unconfirmed','wrong_side','wrong_curve','no_pose','no_obs','other_error','multi_mask'])
@pytest.mark.parametrize('error', ['inferred target discontinuity','no forward centre path'])
def test_target_jump_separation_preserves_all_corner_guards(controller, case, error):
    node = controller
    node.safety_clock.seconds = 1.6
    node.robot_calibration = {'test':True}
    node.parameters['corner_enabled'] = True
    policy, obs, _ = prepared()
    if case == 'unconfirmed':
        policy = CornerPolicy(CornerConfig(staged_turn=True,clearance_m=0.))
    if case == 'wrong_side': obs = dict(obs,side='left')
    if case == 'wrong_curve': obs = dict(obs,curve=obs['curve']+[0.,.1])
    if case == 'no_obs': obs = None
    node.corner_policy = policy
    node._corner_pose_for_frame = lambda *a: None if case=='no_pose' else np.zeros(3)
    def reject(*args, **kwargs):
        raise ValueError('visible boundary identity ambiguous' if case=='other_error'
                         else error)
    node.metric_tracker = SimpleNamespace(update=reject,last_observation=obs)
    masks = [np.zeros((20,20),np.uint8)]*(2 if case=='multi_mask' else 1)
    count = node._update_lane_command(masks,20,node.safety_clock.now())
    if case == 'valid':
        assert count == 1 and node.metric_target['corner_staged']
        assert node.corner_policy.debug['ordinary_target_ignored']
    else:
        assert count == 0 and node.metric_target is None
