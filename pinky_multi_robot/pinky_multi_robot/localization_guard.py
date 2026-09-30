"""Request AMCL global localization and report whether navigation may start.

Run once in each robot's ROS domain. This node never commands robot motion.
The readiness test is deliberately conservative and is only a heuristic:
repeated-looking parts of a map can still fool AMCL.
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener, TransformException


class LocalizationGuard(Node):
    def __init__(self):
        super().__init__('localization_guard')
        self.declare_parameter('max_xy_variance', 0.04)  # 20 cm standard deviation
        self.declare_parameter('max_yaw_variance', 0.12)
        self.declare_parameter('required_samples', 5)
        self.declare_parameter('manual_required_samples', 1)
        self.declare_parameter('sensor_timeout', 2.0)
        # Zero disables pose-age expiry; a first validated AMCL pose is still required.
        self.declare_parameter('pose_timeout', 0.0)
        # Simulation-only option: a stationary AMCL may stop publishing poses.
        # Never use this to ignore a lost laser scan or an explicit reset.
        self.declare_parameter('latch_ready_on_pose_timeout', False)
        self.declare_parameter('search_timeout', 30.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('tf_timeout', 2.0)
        # Never reset a real robot's AMCL merely because central control started.
        # Global localization remains available through an explicit START command.
        self.declare_parameter('auto_global_localization', False)
        self._started = time.monotonic()
        self._last_scan = None
        self._last_pose = None
        self._tf_stamp = None
        self._tf_advanced_at = None
        self._samples = 0
        self._mode = None
        self._request_pending = False
        self._state = 'WAITING_FOR_AMCL'
        self._publisher = self.create_publisher(String, 'localization_status', 10)
        self._tracked_publisher = self.create_publisher(
            PoseWithCovarianceStamped, 'tracked_pose', 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._global_client = self.create_client(
            Empty, 'reinitialize_global_localization')
        self.create_subscription(
            LaserScan, 'scan', self._scan_callback, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 self._pose_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose',
                                 self._initial_pose_callback, 10)
        self.create_timer(0.5, self._tick)
        self.create_timer(0.2, self._publish_tracked_pose)

    def _publish_tracked_pose(self):
        # Never relabel a cached AMCL pose as fresh. TF combines AMCL's map
        # correction with ongoing odometry. Check timestamp progress against
        # local monotonic time, never against the PC's different wall clock.
        if (self._state != 'READY' or self._last_scan is None or
                time.monotonic() - self._last_scan >=
                self.get_parameter('sensor_timeout').value):
            self._tf_stamp = None
            self._tf_advanced_at = None
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self.get_parameter('map_frame').value,
                self.get_parameter('base_frame').value, rclpy.time.Time())
        except TransformException:
            self._tf_stamp = None
            self._tf_advanced_at = None
            return
        stamp = transform.header.stamp.sec * 1000000000 + transform.header.stamp.nanosec
        now = time.monotonic()
        if self._tf_stamp is None or stamp < self._tf_stamp:
            self._tf_stamp = stamp
            self._tf_advanced_at = None
            return  # First sample / clock reset: require subsequent progress.
        if stamp > self._tf_stamp:
            self._tf_stamp = stamp
            self._tf_advanced_at = now
        if (self._tf_advanced_at is None or
                now - self._tf_advanced_at >= self.get_parameter('tf_timeout').value):
            return
        pose = PoseWithCovarianceStamped()
        pose.header = transform.header
        pose.pose.pose.position.x = transform.transform.translation.x
        pose.pose.pose.position.y = transform.transform.translation.y
        pose.pose.pose.position.z = transform.transform.translation.z
        pose.pose.pose.orientation = transform.transform.rotation
        self._tracked_publisher.publish(pose)

    def _scan_callback(self, _message):
        self._last_scan = time.monotonic()

    def _initial_pose_callback(self, _message):
        # Operator-supplied pose takes precedence over an automatic search.
        self._tf_stamp = None
        self._tf_advanced_at = None
        self._mode = 'manual'
        self._samples = 0
        self._last_pose = None
        self._started = time.monotonic()
        self._state = 'VERIFYING_MANUAL_POSE'

    def _pose_callback(self, message):
        now = time.monotonic()
        self._last_pose = now
        if self._mode is None:
            return  # Ignore the hard-coded AMCL startup pose.
        cov = message.pose.covariance
        values = (cov[0], cov[7], cov[35])
        good = (message.header.frame_id == self.get_parameter('map_frame').value and
                all(math.isfinite(v) and v >= 0.0 for v in values) and
                cov[0] <= self.get_parameter('max_xy_variance').value and
                cov[7] <= self.get_parameter('max_xy_variance').value and
                cov[35] <= self.get_parameter('max_yaw_variance').value and
                self._last_scan is not None and
                now - self._last_scan < self.get_parameter('sensor_timeout').value)
        self._samples = self._samples + 1 if good else 0

    def _global_done(self, future):
        self._request_pending = False
        if self._mode == 'manual':
            return
        try:
            future.result()
        except Exception as error:
            self.get_logger().error(f'AMCL global localization failed: {error}')
            self._state = 'NEEDS_MANUAL_POSE'
            return
        self._mode = 'global'
        self._samples = 0
        self._started = time.monotonic()
        self._state = 'SEARCHING'
        self.get_logger().info('AMCL global localization requested; no motion commanded.')

    def _tick(self):
        now = time.monotonic()
        timeout = self.get_parameter('sensor_timeout').value
        if self._last_scan is None or now - self._last_scan >= timeout:
            self._samples = 0
            self._state = 'WAITING_FOR_SCAN'
        elif self._mode is None and not self._request_pending:
            if not self.get_parameter('auto_global_localization').value:
                self._state = 'NEEDS_MANUAL_POSE'
            elif self._global_client.service_is_ready():
                self._request_pending = True
                future = self._global_client.call_async(Empty.Request())
                future.add_done_callback(self._global_done)
            else:
                self._state = 'WAITING_FOR_AMCL'
        elif self._mode is not None:
            pose_timeout = self.get_parameter('pose_timeout').value
            if self._last_pose is None:
                self._samples = 0
                self._state = 'WAITING_FOR_AMCL_POSE'
            elif pose_timeout > 0.0 and now - self._last_pose >= pose_timeout:
                if not (self.get_parameter('latch_ready_on_pose_timeout').value and
                        self._state == 'READY'):
                    self._samples = 0
                    self._state = 'WAITING_FOR_AMCL_POSE'
            elif self._samples >= (
                    self.get_parameter('manual_required_samples').value
                    if self._mode == 'manual'
                    else self.get_parameter('required_samples').value):
                self._state = 'READY'
            elif self._mode == 'global' and now - self._started >= self.get_parameter('search_timeout').value:
                self._state = 'NEEDS_MANUAL_POSE'
            else:
                self._state = 'VERIFYING_MANUAL_POSE' if self._mode == 'manual' else 'SEARCHING'
        self._publisher.publish(String(data=self._state))


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
