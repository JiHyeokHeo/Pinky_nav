"""Publish a simulated robot's odom-to-base TF from bridged Gazebo odometry."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class SimOdomTf(Node):
    def __init__(self):
        super().__init__('sim_odom_tf')
        self._broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, 'odom', self._on_odom, 10)

    def _on_odom(self, odom):
        if not odom.header.frame_id or not odom.child_frame_id:
            return
        transform = TransformStamped()
        transform.header = odom.header
        transform.child_frame_id = odom.child_frame_id
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = odom.pose.pose.position.z
        transform.transform.rotation = odom.pose.pose.orientation
        self._broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = SimOdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
