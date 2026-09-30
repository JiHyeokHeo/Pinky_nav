import math
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints


class FactoryPatrol(Node):

    def __init__(self):
        super().__init__('factory_patrol')

        # Nav2의 FollowWaypoints Action Server와 연결
        self.client = ActionClient(
            self,
            FollowWaypoints,
            'follow_waypoints'
        )

        # 여기에 네가 구한 실제 좌표를 넣어
        self.point_1 = (0.0, 0.0, 0.0)
        self.point_2 = (-1.45, 0.052, 0.0)
        self.point_3 = (-3.22, -2.88, 0.0)

        self.start_patrol()


    def make_pose(self, x, y, yaw):
        pose = PoseStamped()

        # Nav2 좌표계
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        # 위치
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        # yaw → quaternion
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)


        return pose


    def start_patrol(self):

        self.get_logger().info(
            'Nav2 FollowWaypoints 서버를 기다리는 중...'
        )

        self.client.wait_for_server()

        self.get_logger().info(
            'FollowWaypoints 서버 연결 완료'
        )

        # 1 → 2 → 3 → 1
        waypoints = [
            self.make_pose(*self.point_2),
            self.make_pose(*self.point_3),
            self.make_pose(*self.point_1),
        ]

        goal = FollowWaypoints.Goal()
        goal.poses = waypoints

        self.get_logger().info(
            '2 → 3 → 1 주행 시작'
        )

        future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(
            self.goal_response_callback
        )


    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                'Nav2가 목표를 거절했습니다.'
            )
            return

        self.get_logger().info(
            'Nav2가 목표를 수락했습니다.'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.result_callback
        )


    def feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self.get_logger().info(
            f'현재 waypoint 번호: {feedback.current_waypoint}'
        )


    def result_callback(self, future):

        self.get_logger().info(
            '모든 waypoint 주행 완료'
        )


def main(args=None):

    rclpy.init(args=args)

    node = FactoryPatrol()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()