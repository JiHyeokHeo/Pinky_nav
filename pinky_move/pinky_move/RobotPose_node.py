import rclpy

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class RobotPosePublisher(Node):

    def __init__(self):
        super().__init__('robot_pose_publisher')

        self.publisher = self.create_publisher(
            PoseStamped,
            '/robot_pose',
            10
        )

        # 0.5초마다 publish
        self.timer = self.create_timer(
            0.5,
            self.publish_pose
        )

        self.x = 0.0
        self.y = 0.0


    def publish_pose(self):

        pose = PoseStamped()

        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.position.z = 0.0

        pose.pose.orientation.w = 1.0

        self.publisher.publish(pose)

        self.get_logger().info(
            f'위치 전송: x={self.x:.2f}, y={self.y:.2f}'
        )

        # 테스트용 가짜 이동
        self.x += 0.1


def main(args=None):
    rclpy.init(args=args)

    node = RobotPosePublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()