"""State-transition checks without starting DDS or moving a robot."""

import time
import unittest
from types import SimpleNamespace

from geometry_msgs.msg import PoseWithCovarianceStamped
from pinky_multi_robot.localization_guard import LocalizationGuard


class FakeGuard:
    def __init__(self, latch):
        now = time.monotonic()
        self._last_scan = now
        self._last_pose = now - 31.0
        self._started = now - 40.0
        self._mode = 'manual'
        self._state = 'READY'
        self._samples = 5
        self._request_pending = False
        self.published = []
        self._publisher = SimpleNamespace(
            publish=lambda message: self.published.append(message.data))
        self.parameters = {
            'sensor_timeout': 2.0,
            'pose_timeout': 30.0,
            'latch_ready_on_pose_timeout': latch,
            'auto_global_localization': False,
            'manual_required_samples': 1,
            'required_samples': 5,
            'map_frame': 'map',
            'max_xy_variance': 0.04,
            'max_yaw_variance': 0.12,
        }

    def get_parameter(self, name):
        return SimpleNamespace(value=self.parameters[name])


class LocalizationReadyLatchTest(unittest.TestCase):
    def test_disabled_timeout_keeps_validated_pose_ready(self):
        guard = FakeGuard(latch=False)
        guard.parameters['pose_timeout'] = 0.0
        guard._last_pose -= 3600.0
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'READY')

    def test_disabled_timeout_still_requires_first_pose(self):
        guard = FakeGuard(latch=False)
        guard.parameters['pose_timeout'] = 0.0
        guard._last_pose = None
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'WAITING_FOR_AMCL_POSE')

    def test_disabled_timeout_still_blocks_lost_scan(self):
        guard = FakeGuard(latch=False)
        guard.parameters['pose_timeout'] = 0.0
        guard._last_scan -= 3.0
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'WAITING_FOR_SCAN')

    def test_sim_keeps_ready_when_only_pose_is_stale(self):
        guard = FakeGuard(latch=True)
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'READY')

    def test_real_robot_still_expires_stale_pose(self):
        guard = FakeGuard(latch=False)
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'WAITING_FOR_AMCL_POSE')

    def test_lost_scan_overrides_latch(self):
        guard = FakeGuard(latch=True)
        guard._last_scan -= 3.0
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'WAITING_FOR_SCAN')

    def test_explicit_initial_pose_clears_ready(self):
        guard = FakeGuard(latch=True)
        LocalizationGuard._initial_pose_callback(guard, None)
        LocalizationGuard._tick(guard)
        self.assertNotEqual(guard.published[-1], 'READY')

    def test_startup_does_not_reset_amcl(self):
        guard = FakeGuard(latch=False)
        guard._mode = None
        guard._last_pose = None
        guard._state = 'WAITING_FOR_AMCL'
        guard._global_client = SimpleNamespace(
            service_is_ready=lambda: (_ for _ in ()).throw(
                AssertionError('AMCL reset service must not be queried')))
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'NEEDS_MANUAL_POSE')

    def test_one_valid_manual_pose_becomes_ready(self):
        guard = FakeGuard(latch=False)
        LocalizationGuard._initial_pose_callback(guard, None)
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = 'map'
        LocalizationGuard._pose_callback(guard, pose)
        LocalizationGuard._tick(guard)
        self.assertEqual(guard.published[-1], 'READY')


if __name__ == '__main__':
    unittest.main()
