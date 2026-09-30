#!/usr/bin/env python3
"""로봇(pinky) 카메라 영상을 계속 내보내는 노드.

    LD_PRELOAD=/usr/local/lib/aarch64-linux-gnu/libpisp.so.1.0.7 python3 cam_stream.py

매 프레임을 JPEG 으로 압축해 토픽으로 보낸다. 원본 그대로 보내면 WiFi 가 못 버틴다
(640x480 RGB 한 장이 900KB, JPEG 이면 30~60KB).

카메라 설정은 이 로봇에서 동작이 확인된 경로를 쓴다:
  create_preview_configuration + RGB888 + capture_array

이 로봇은 카메라가 거꾸로 달려 있어 기본으로 180도 돌려서 내보낸다.
여기서 바로잡아야 녹화/사진/rqt_image_view 까지 모두 제대로 나온다.
방향이 다르면 --rotate 와 --flip 으로 맞출 것.
"""

import argparse
import sys
import time

# 카메라 라이브러리를 ROS 보다 먼저 올린다
from picamera2 import Picamera2
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage

# image_transport 규약을 따르는 이름이라 rqt_image_view 로도 바로 볼 수 있다
IMAGE_TOPIC = '/pinky/camera/image_raw/compressed'

DEFAULT_SIZE = (640, 480)
DEFAULT_FPS = 15.0
DEFAULT_QUALITY = 80
DEFAULT_ROTATE = 180          # 이 로봇의 카메라는 거꾸로 달려 있다

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
FLIPS = {'none': None, 'h': 1, 'v': 0, 'both': -1}


def orient(frame, rotate, flip):
    """회전 -> 뒤집기 순서로 방향을 바로잡는다."""
    code = ROTATIONS[rotate]
    if code is not None:
        frame = cv2.rotate(frame, code)
    code = FLIPS[flip]
    if code is not None:
        frame = cv2.flip(frame, code)
    return frame


class CamStream(Node):
    def __init__(self, picam2, args):
        super().__init__('cam_stream')
        self.picam2 = picam2
        self.quality = args.quality
        self.rotate = args.rotate
        self.flip = args.flip

        # 영상은 최신 프레임이 중요하지 빠진 프레임을 다시 보낼 필요가 없다
        self.pub = self.create_publisher(
            CompressedImage, IMAGE_TOPIC, qos_profile_sensor_data)

        self.frames = 0
        self.bytes_sent = 0
        self.last_report = time.time()

        self.create_timer(1.0 / args.fps, self.tick)
        self.get_logger().info(
            f'{args.width}x{args.height} @ {args.fps:g} fps 로 '
            f'{IMAGE_TOPIC} 에 내보냅니다. '
            f'(회전 {args.rotate}도, 뒤집기 {args.flip})')

    def tick(self):
        frame = orient(self.picam2.capture_array(), self.rotate, self.flip)

        # picamera2 의 RGB888 은 numpy 에서 BGR 순서라 imencode 에 그대로 넣는다
        ok, encoded = cv2.imencode(
            '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            self.get_logger().warn('JPEG 인코딩 실패 - 이 프레임은 건너뜁니다.')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        self.pub.publish(msg)

        self.frames += 1
        self.bytes_sent += len(msg.data)

        now = time.time()
        elapsed = now - self.last_report
        if elapsed >= 5.0:
            self.get_logger().info(
                f'{self.frames / elapsed:.1f} fps, '
                f'{self.bytes_sent / elapsed / 1024:.0f} KB/s')
            self.frames = 0
            self.bytes_sent = 0
            self.last_report = now


def parse_args(argv):
    parser = argparse.ArgumentParser(description='pinky 카메라 영상 송출 (로봇에서 실행).')
    parser.add_argument('--width', type=int, default=DEFAULT_SIZE[0])
    parser.add_argument('--height', type=int, default=DEFAULT_SIZE[1])
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                        help=f'초당 프레임 수 (기본 {DEFAULT_FPS:g})')
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY,
                        help=f'JPEG 품질 1~100 (기본 {DEFAULT_QUALITY}). '
                             '낮출수록 대역폭이 줄어든다.')
    parser.add_argument('--rotate', type=int, default=DEFAULT_ROTATE,
                        choices=sorted(ROTATIONS), metavar='{0,90,180,270}',
                        help=f'시계 방향 회전 각도 (기본 {DEFAULT_ROTATE}). '
                             '이 로봇은 카메라가 거꾸로라 180 이 기본이다.')
    parser.add_argument('--flip', default='none', choices=sorted(FLIPS),
                        help='회전 뒤 추가로 뒤집기: h 좌우, v 상하, both 둘 다 (기본 none)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    print('카메라를 준비합니다...', flush=True)
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (args.width, args.height)}))
    picam2.start()
    print(f'카메라 준비 완료 ({args.width}x{args.height})', flush=True)

    rclpy.init()
    node = CamStream(picam2, args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n송출을 중단합니다.')
    finally:
        picam2.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
