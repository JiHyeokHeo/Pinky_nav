import numpy as np
import cv2
import pytest
from types import SimpleNamespace
from test_lane_controller import controller
from test_lane_corner import observation, ordinary
from pinky_move.lane_corner import CornerConfig, CornerPolicy, classify_boundary
from pinky_move.metric_lane import near_connected_component


def test_ordinary_pursuit_slows_forward_speed_to_preserve_curvature(controller):
    target = dict(x_m=.1,y_m=.15,inferred=True,boundary_count=1,width_m=.154)
    controller.robot_calibration = {'test': True}
    controller.parameters['corner_enabled'] = False
    controller.metric_tracker = SimpleNamespace(update=lambda *a,**kw: target)
    assert controller._update_lane_command([],640,controller.safety_clock.now()) == 1
    k = 2*.15/(.1**2+.15**2)
    assert controller.latest_linear < .03
    assert controller.latest_angular == pytest.approx(controller.latest_linear*k)
    assert abs(controller.latest_angular) <= .15


@pytest.mark.parametrize('sign', [1,-1])
def test_confirmed_right_angle_never_calls_swept_footprint_planner(monkeypatch, sign):
    def forbidden(*args, **kwargs):
        raise AssertionError('right-angle staging must not call Bezier footprint planner')
    monkeypatch.setattr('pinky_move.lane_corner.plan_corner', forbidden)
    policy = CornerPolicy(CornerConfig(staged_turn=True, clearance_m=0.))
    for now in (1.,1.2,1.4):
        target = policy.update(observation(sign), ordinary(), now,np.zeros(3),.22,.25)
    assert target['corner_staged']


@pytest.mark.parametrize('sign', [1,-1])
def test_rounded_right_angle_with_distributed_heading_has_two_supported_legs(sign):
    radius=.025
    angle=np.linspace(-np.pi/2,0,35)
    curve=np.vstack((np.column_stack((np.linspace(.1,.3,40),np.zeros(40))),
                     np.column_stack((.3+radius*np.cos(angle),radius+radius*np.sin(angle)))[1:],
                     np.column_stack((np.full(40,.3+radius),np.linspace(radius,.22,40)))[1:]))
    curve[:,1] *= sign
    result=classify_boundary(curve,CornerConfig(segment_m=.02,window_m=.02))
    assert result['kind']=='CORNER'
    assert sign*result['angle_deg'] > 80


def test_thin_bridge_removal_never_selects_between_two_near_lines():
    mask=np.zeros((480,640),np.uint8)
    cv2.line(mask,(40,240),(600,240),1,10)
    cv2.line(mask,(100,470),(600,330),1,20)
    cv2.line(mask,(400,240),(400,385),1,1)
    near=near_connected_component(mask)
    assert near is not None and np.count_nonzero(near[:300]) == 0
    mask[:]=0
    cv2.line(mask,(100,470),(200,330),1,20)
    cv2.line(mask,(400,470),(500,330),1,20)
    assert near_connected_component(mask) is None
