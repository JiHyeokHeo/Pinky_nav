#!/usr/bin/env python3
"""Publish Raspberry Pi CSI frames from rpicam-vid as ROS images."""

import os
import select
import subprocess

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CsiCameraPublisher(Node):
    """Read rpicam-vid MJPEG output and publish sensor_msgs/Image."""

    def __init__(self):
        super().__init__('pinky_csi_camera')
        self.declare_parameter('camera', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('rotation_degrees', 0)
        self.declare_parameter('frame_id', 'front_camera_link')
        self.declare_parameter('image_topic', '/camera/image_raw')

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, str(self.get_parameter('image_topic').value),
            qos_profile_sensor_data)
        command = [
            'rpicam-vid', '--nopreview', '--timeout', '0',
            '--codec', 'mjpeg',
            '--mode', '640:480:10',
            '--camera', str(self.get_parameter('camera').value),
            '--width', str(self.get_parameter('width').value),
            '--height', str(self.get_parameter('height').value),
            '--framerate', str(self.get_parameter('framerate').value),
            '--output', '-',
        ]
        camera_env = os.environ.copy()
        camera_env['LD_LIBRARY_PATH'] = ':'.join(
            path for path in camera_env.get('LD_LIBRARY_PATH', '').split(':')
            if not path.startswith('/opt/ros/'))
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, env=camera_env,
            bufsize=0)
        self.buffer = bytearray()
        self.timer = self.create_timer(0.002, self._read_frame)
        self.get_logger().info(
            f'Publishing CSI camera frames to '
            f'{self.get_parameter("image_topic").value}')

    def _read_frame(self):
        """Drain ready bytes and publish complete JPEG frames."""
        if self.process.poll() is not None:
            self.get_logger().error('rpicam-vid exited; camera stream stopped')
            self.timer.cancel()
            return
        readable, _, _ = select.select([self.process.stdout], [], [], 0)
        if readable:
            self.buffer.extend(os.read(self.process.stdout.fileno(), 65536))
        start = self.buffer.find(b'\xff\xd8')
        end = self.buffer.find(b'\xff\xd9', start + 2)
        if start < 0 or end < 0:
            if len(self.buffer) > 4_000_000:
                self.buffer.clear()
            return
        jpeg = bytes(self.buffer[start:end + 2])
        del self.buffer[:end + 2]
        image = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        rotation_degrees = int(
            self.get_parameter('rotation_degrees').value)
        if rotation_degrees == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif rotation_degrees != 0:
            self.get_logger().error(
                'rotation_degrees must be 0 or 180; frame dropped',
                throttle_duration_sec=2.0)
            return
        message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter('frame_id').value)
        self.publisher.publish(message)

    def destroy_node(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CsiCameraPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
