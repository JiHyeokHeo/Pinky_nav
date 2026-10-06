"""Regression for fresh odometry received after inference completion."""
import numpy as np
import pytest

from test_lane_controller import controller  # noqa: F401: shared transport-free fixture


def prepare(node):
    node.safety_clock.seconds = 10.2
    node.corner_odom = (10.18, np.array([.02, 0., 0.]))
    node.corner_capture_stamp = 1_100_000_000
    node.corner_odom_history = [(1_000_000_000, np.zeros(3)),
                                (1_200_000_000, np.array([.02, 0., 0.]))]


def test_new_odom_after_completion_is_accepted(controller):
    prepare(controller)
    np.testing.assert_allclose(controller._corner_pose_for_frame(10.15), [.01, 0., 0.])
    assert controller.corner_pose_diagnostic['reason'] == 'interpolated'
    assert controller.corner_pose_diagnostic['odom_age_s'] == pytest.approx(.02)


def test_true_outage_is_not_hidden_by_old_completion_time(controller):
    prepare(controller)
    controller.safety_clock.seconds = 10.6
    assert controller._corner_pose_for_frame(10.15) is None
    assert controller.corner_pose_diagnostic['reason'] == 'odometry stale or future reception'


def test_uncovered_camera_time_still_rejected(controller):
    prepare(controller)
    controller.corner_odom_history = controller.corner_odom_history[:1]
    assert controller._corner_pose_for_frame(10.15) is None
    assert controller.corner_pose_diagnostic['reason'] == 'camera time not covered by odometry'
