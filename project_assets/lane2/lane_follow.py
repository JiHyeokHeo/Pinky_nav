#!/usr/bin/env python3
"""차선을 따라 주행하는 노드 (PC 에서 실행).

    python3 lane_follow.py                    # 학습 전: 영상처리로 차선 검출
    python3 lane_follow.py --model best.pt    # 학습 후: YOLO 로 검출

로봇에서 cam_stream.py 와 라이다(bringup)가 돌고 있어야 한다.
YOLO 는 라즈베리파이가 감당하기 어려워 PC 에서 돌리고 cmd_vel 만 보낸다.

과제 요구사항 대응:
  차선 추종/중앙 정렬  lane_core 의 추적 + PD 제어
  횡단보도 일시 정지    crosswalk 검출 -> 정지 -> 재출발
  장애물 감지 및 대기    라이다 전방 거리 (카메라보다 훨씬 정확하다)
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_core as core

IMAGE_TOPIC = '/pinky/camera/image_raw/compressed'
SCAN_TOPIC = '/scan'
CMD_TOPIC = '/cmd_vel'

FLOOR_TOP = 200          # 이 위는 벽이라 보지 않는다
LOOKAHEAD_ROW = 0.45     # 바닥 영역에서 목표를 읽을 높이 (0=먼 곳, 1=바로 앞)
MIN_BLOB_RATIO = 0.008

CONTROL_HZ = 15.0
CROSSWALK_STOP_SEC = 3.0
CROSSWALK_IGNORE_SEC = 6.0   # 정지 후 이 시간 동안은 같은 횡단보도를 무시


# --------------------------------------------------------------------------
# 차선 검출기 - 학습 전후로 갈아끼운다
# --------------------------------------------------------------------------

class ThresholdDetector:
    """영상처리만으로 흰 테이프를 찾는다. 모델 없이 지금 바로 쓸 수 있다."""

    name = '영상처리'

    def __init__(self, args):
        self.kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame):
        floor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[FLOOR_TOP:, :]
        _, mask = cv2.threshold(floor, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        row = int(mask.shape[0] * LOOKAHEAD_ROW)
        xs = runs_at_row(mask, row)

        # 가로로 긴 흰 줄이 여러 개면 횡단보도로 본다
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        wide = sum(1 for i in range(1, count)
                   if stats[i, 4] > mask.size * MIN_BLOB_RATIO
                   and stats[i, 2] > mask.shape[1] * 0.35)
        return xs, wide >= 3, mask


class YoloDetector:
    """학습한 yolo11 세그멘테이션 모델로 차선과 횡단보도를 찾는다."""

    name = 'YOLO'

    def __init__(self, args):
        from ultralytics import YOLO
        self.model = YOLO(args.model)
        self.conf = args.conf
        self.imgsz = args.imgsz
        self.names = self.model.names
        print(f'모델 클래스: {self.names}', flush=True)

    def detect(self, frame):
        floor = frame[FLOOR_TOP:, :]
        result = self.model.predict(floor, conf=self.conf, imgsz=self.imgsz,
                                    verbose=False)[0]

        height, width = floor.shape[:2]
        lane_mask = np.zeros((height, width), np.uint8)
        crosswalk = False

        if result.masks is not None:
            for poly, cls in zip(result.masks.xy, result.boxes.cls.tolist()):
                label = self.names[int(cls)]
                if label == 'lane':
                    cv2.fillPoly(lane_mask, [poly.astype(np.int32)], 255)
                elif label == 'Crossline':
                    crosswalk = True

        row = int(height * LOOKAHEAD_ROW)
        return runs_at_row(lane_mask, row), crosswalk, lane_mask


def runs_at_row(mask, row):
    """한 줄에서 흰 구간들의 중심 x 목록을 뽑는다."""
    if not 0 <= row < mask.shape[0]:
        return []
    line = mask[row] > 0
    edges = np.diff(line.astype(np.int8))
    starts = list(np.where(edges == 1)[0])
    ends = list(np.where(edges == -1)[0])
    if line[0]:
        starts.insert(0, 0)
    if line[-1]:
        ends.append(len(line) - 1)

    centers = []
    for s, e in zip(starts, ends):
        if e - s >= 6:                  # 너무 얇은 것은 잡음
            centers.append((s + e) / 2.0)
    return centers


# --------------------------------------------------------------------------

class LaneFollower(Node):
    def __init__(self, detector, args):
        super().__init__('lane_follow')
        self.detector = detector
        self.args = args

        self.frame = None
        self.front_range = float('inf')

        self.left = self.right = None
        self.prev_error = None
        self.last_angular = 0.0
        self.last_seen = time.time()
        self.crosswalk_until = 0.0
        self.crosswalk_ignore_until = 0.0
        self.stopped_reason = None

        self.create_subscription(CompressedImage, IMAGE_TOPIC,
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, SCAN_TOPIC,
                                 self.on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)

        self.create_timer(1.0 / CONTROL_HZ, self.control_loop)
        self.get_logger().info(
            f'{detector.name} 로 차선을 찾습니다. '
            f'속도 {args.speed} m/s, 차선폭 {args.lane_width}px')

    # ---------------------------------------------------------------- 수신
    def on_image(self, msg):
        buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is not None:
            self.frame = frame

    def on_scan(self, msg):
        """정면 좌우 15도 안에서 가장 가까운 거리."""
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        front = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) < np.radians(15)
        valid = front & np.isfinite(ranges) & (ranges > msg.range_min)
        self.front_range = float(ranges[valid].min()) if np.any(valid) else float('inf')

    # ---------------------------------------------------------------- 명령
    def publish(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self, reason):
        self.publish(0.0, 0.0)
        if self.stopped_reason != reason:
            self.get_logger().warn(f'정지 - {reason}')
            self.stopped_reason = reason

    # ---------------------------------------------------------------- 제어
    def control_loop(self):
        now = time.time()

        if self.frame is None:
            self.get_logger().warn('영상 대기 중...', throttle_duration_sec=5.0)
            return

        # 1) 장애물이 먼저다. 라이다가 카메라보다 훨씬 확실하다.
        if self.front_range < self.args.obstacle_dist:
            self.stop(f'전방 {self.front_range:.2f} m 에 장애물 - 치워주세요')
            return

        # 2) 횡단보도에서 멈춰 있는 중인가
        if now < self.crosswalk_until:
            self.stop('횡단보도 일시 정지')
            return

        xs, crosswalk, mask = self.detector.detect(self.frame)

        # 3) 횡단보도를 새로 만났으면 멈춘다
        if crosswalk and now > self.crosswalk_ignore_until:
            self.crosswalk_until = now + self.args.crosswalk_stop
            self.crosswalk_ignore_until = now + CROSSWALK_IGNORE_SEC
            self.stop('횡단보도 발견')
            return

        height, width = mask.shape[:2]
        center = width / 2.0

        # 4) 차선 추적
        if xs:
            self.left, self.right = core.assign_sides(
                xs, self.left, self.right, self.args.lane_width, center)
            self.last_seen = now
        else:
            self.left = self.right = None

        target = core.target_x(self.left, self.right, self.args.lane_width, center)

        if target is None:
            self.handle_lost(now)
            return

        self.stopped_reason = None
        angular, error = core.steering(
            target, center, width / 2.0, self.args.kp, self.args.kd,
            self.prev_error, 1.0 / CONTROL_HZ, self.args.max_angular)
        self.prev_error = error
        self.last_angular = angular

        linear = core.linear_speed(error, self.args.speed, self.args.min_speed)
        self.publish(linear, angular)

        sides = ('양쪽' if self.left is not None and self.right is not None
                 else '왼쪽만' if self.left is not None else '오른쪽만')
        self.get_logger().info(
            f'{sides} | 오차 {error:+.2f} | v={linear:.2f} w={angular:+.2f}',
            throttle_duration_sec=1.0)

        if self.args.view:
            self.show(mask, target, center)

    def handle_lost(self, now):
        """차선을 놓쳤을 때: 잠깐 버티다 -> 찾아보다 -> 정지."""
        lost = now - self.last_seen
        action = core.lost_action(lost, self.args.hold, self.args.search)

        if action == core.HOLD:
            self.publish(self.args.min_speed, self.last_angular)
        elif action == core.SEARCH:
            direction = core.search_direction(self.left, self.right, 320.0)
            self.get_logger().warn(f'차선 탐색 중 ({lost:.1f}초)',
                                   throttle_duration_sec=1.0)
            self.publish(0.0, direction * self.args.search_angular)
        else:
            self.stop(f'{lost:.0f}초간 차선을 찾지 못함')

    def show(self, mask, target, center):
        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        row = int(mask.shape[0] * LOOKAHEAD_ROW)
        cv2.line(view, (0, row), (mask.shape[1], row), (80, 80, 80), 1)
        cv2.circle(view, (int(target), row), 7, (0, 0, 255), -1)
        cv2.line(view, (int(center), 0), (int(center), mask.shape[0]), (0, 255, 0), 1)
        cv2.imshow('lane', view)
        cv2.waitKey(1)


# --------------------------------------------------------------------------

def parse_args(argv):
    parser = argparse.ArgumentParser(description='차선 추종 주행 (PC 에서 실행).')
    parser.add_argument('--model', default=None,
                        help='학습한 yolo11 세그멘테이션 가중치(.pt). '
                             '생략하면 영상처리로 검출한다.')
    parser.add_argument('--conf', type=float, default=0.35, help='YOLO 신뢰도 기준')
    parser.add_argument('--imgsz', type=int, default=320, help='YOLO 입력 크기')

    parser.add_argument('--speed', type=float, default=0.15, help='기본 속도 [m/s]')
    parser.add_argument('--min-speed', type=float, default=0.05, help='최저 속도 [m/s]')
    parser.add_argument('--max-angular', type=float, default=1.2, help='최대 각속도 [rad/s]')
    parser.add_argument('--kp', type=float, default=1.6, help='조향 비례 이득')
    parser.add_argument('--kd', type=float, default=0.25, help='조향 미분 이득')
    parser.add_argument('--lane-width', type=float, default=300.0,
                        help='화면에서의 차선 간격 [px]. 한쪽만 보일 때 쓴다.')

    parser.add_argument('--obstacle-dist', type=float, default=0.30,
                        help='이 거리 안에 장애물이 있으면 정지 [m]')
    parser.add_argument('--crosswalk-stop', type=float, default=CROSSWALK_STOP_SEC,
                        help='횡단보도에서 멈추는 시간 [s]')
    parser.add_argument('--hold', type=float, default=0.3,
                        help='차선을 놓쳐도 이 시간은 직전 조향 유지 [s]')
    parser.add_argument('--search', type=float, default=2.0,
                        help='그 후 이 시간 동안 탐색 회전 [s]')
    parser.add_argument('--search-angular', type=float, default=0.5,
                        help='탐색 회전 속도 [rad/s]')
    parser.add_argument('--view', action='store_true', help='검출 결과를 창에 띄운다')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    detector = YoloDetector(args) if args.model else ThresholdDetector(args)

    rclpy.init()
    node = LaneFollower(detector, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n중단 - 로봇을 정지시킵니다.')
    finally:
        node.publish(0.0, 0.0)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        if args.view:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
