"""Publish operator localization commands directly into each robot domain."""

import math
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.context import Context
from rclpy.node import Node
from std_msgs.msg import String


class LocalizationCommandBus:
    def __init__(self, domains=None, namespaces=None):
        domains = domains or {'pinky1': 20, 'pinky2': 22}
        namespaces = namespaces or {}
        self._contexts = []
        self._nodes = []
        self._publishers = {}
        self._pose_publishers = {}
        self._robot_nodes = {}
        try:
            for robot, domain in domains.items():
                context = Context()
                rclpy.init(context=context, domain_id=domain)
                self._contexts.append(context)
                node = Node(f'{robot}_localization_operator', context=context)
                self._nodes.append(node)
                self._robot_nodes[robot] = node
                prefix = '/' + namespaces[robot].strip('/') if namespaces.get(robot) else ''
                self._publishers[robot] = node.create_publisher(
                    String, f'{prefix}/localization_command', 10)
                self._pose_publishers[robot] = node.create_publisher(
                    PoseWithCovarianceStamped, f'{prefix}/initialpose', 10)
        except Exception:
            self.close()
            raise

    def send(self, robot, command):
        publisher = self._publishers[robot]
        if publisher.get_subscription_count() == 0:
            return False
        publisher.publish(String(data=command))
        return True

    def set_initial_pose(self, robot, x, y, yaw, frame_id='map'):
        publisher = self._pose_publishers[robot]
        if publisher.get_subscription_count() == 0:
            return False
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self._robot_nodes[robot].get_clock().now().to_msg()
        pose.pose.pose.position.x = float(x)
        pose.pose.pose.position.y = float(y)
        pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
        pose.pose.covariance[0] = 0.04
        pose.pose.covariance[7] = 0.04
        pose.pose.covariance[35] = math.radians(15.0) ** 2
        publisher.publish(pose)
        return True

    def close(self):
        for node in self._nodes:
            node.destroy_node()
        self._nodes.clear()
        for context in self._contexts:
            if context.ok():
                rclpy.shutdown(context=context)
        self._contexts.clear()
