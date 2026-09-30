"""Pure unit tests: no DDS participants or robot commands."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from geometry_msgs.msg import TransformStamped
from rclpy.clock import ClockType
from rclpy.time import Time
from tf2_ros import TransformException

from pinky_multi_robot.localization_guard import LocalizationGuard
from test_localization_ready_latch import FakeGuard


def guard_with_tf(stamp=99):
    guard = FakeGuard(latch=False)
    guard._tf_stamp = None
    guard._tf_advanced_at = None
    guard.parameters.update(base_frame='base_footprint', tf_timeout=2.0)
    transform = TransformStamped()
    transform.header.frame_id = 'map'
    transform.header.stamp.sec = stamp
    transform.transform.translation.x = 1.25
    transform.transform.rotation.w = 1.0
    guard._tf_buffer = SimpleNamespace(lookup_transform=Mock(return_value=transform))
    guard._tracked_publisher = SimpleNamespace(publish=Mock())
    guard.get_clock = lambda: SimpleNamespace(
        clock_type=ClockType.ROS_TIME,
        now=lambda: Time(seconds=100, clock_type=ClockType.ROS_TIME))
    return guard


def advance(guard):
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tf_buffer.lookup_transform.return_value.header.stamp.nanosec += 100000000
    LocalizationGuard._publish_tracked_pose(guard)


def test_old_amcl_with_fresh_tf_publishes_map_pose():
    guard = guard_with_tf()
    guard._last_pose -= 3600
    advance(guard)
    pose = guard._tracked_publisher.publish.call_args.args[0]
    assert pose.header.frame_id == 'map'
    assert pose.header.stamp.sec == 99  # preserve actual TF timestamp
    assert pose.pose.pose.position.x == 1.25


@pytest.mark.parametrize('stamp', [1, 1789834527, 2000000000])
def test_clock_offset_allows_advancing_tf(stamp):
    guard = guard_with_tf(stamp)
    advance(guard)
    guard._tracked_publisher.publish.assert_called_once()


def test_cached_first_sample_is_not_live():
    guard = guard_with_tf()
    LocalizationGuard._publish_tracked_pose(guard)
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()


def test_tf_stall_blocks_even_with_live_scan():
    guard = guard_with_tf()
    advance(guard)
    guard._tracked_publisher.publish.reset_mock()
    future = guard._tf_advanced_at + 2.1
    guard._last_scan = future
    with patch('pinky_multi_robot.localization_guard.time.monotonic', return_value=future):
        LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()


def test_backward_jump_requires_progress_again():
    guard = guard_with_tf()
    advance(guard)
    guard._tracked_publisher.publish.reset_mock()
    guard._tf_buffer.lookup_transform.return_value.header.stamp.sec = 1
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()
    advance(guard)
    guard._tracked_publisher.publish.assert_called_once()


def test_lost_scan_blocks_pose_even_before_status_timer():
    guard = guard_with_tf()
    guard._last_scan -= 3
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()


def test_reset_blocks_pose():
    guard = guard_with_tf()
    LocalizationGuard._initial_pose_callback(guard, None)
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()


def test_missing_tf_blocks_pose():
    guard = guard_with_tf()
    guard._tf_buffer.lookup_transform.side_effect = TransformException('missing')
    LocalizationGuard._publish_tracked_pose(guard)
    guard._tracked_publisher.publish.assert_not_called()
