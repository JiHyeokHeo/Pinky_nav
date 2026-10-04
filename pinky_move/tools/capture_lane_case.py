#!/usr/bin/env python3
"""Read-only ROS case capture: images, status and actual odometry, no commands.

Use: ROS_DOMAIN_ID=22 python3 tools/capture_lane_case.py --output reports/cases/CASE
Camera/debug frames are independently timestamped, not claimed synchronised.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=8.)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error('--seconds must be in (0,60]')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier case.
    started = time.monotonic()
    record = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
                  read_only=True, frames={}, samples=[])
    rclpy.init()
    node = rclpy.create_node('lane_case_recorder', enable_rosout=False,
                            start_parameter_services=False)
    record['domain_id'] = node.context.get_domain_id()

    def stamp(msg):
        return dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec,
                    frame_id=msg.header.frame_id)

    def sample(kind, values):
        record['samples'].append(dict(elapsed_s=time.monotonic()-started,
                                      kind=kind, **values))

    def raw(msg):
        if 'raw' in record['frames']:
            return
        data = bytes(msg.data)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        # Preserve received JPEG bytes, without overlays or re-encoding.
        (output/'camera_raw.jpg').write_bytes(data)
        record['frames']['raw'] = stamp(msg)

    def debug(msg):
        if 'debug' in record['frames'] or msg.encoding not in ('bgr8', 'rgb8'):
            return
        rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        frame = rows[:, :msg.width*3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if cv2.imwrite(str(output/'debug_image.jpg'), frame):
            record['frames']['debug'] = stamp(msg)

    def velocity(msg):
        sample('cmd_vel', dict(linear_x=msg.linear.x, angular_z=msg.angular.z))

    def odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        sample('odom', dict(stamp=stamp(msg), position=[p.x,p.y,p.z],
            orientation=[q.x,q.y,q.z,q.w], linear_x=msg.twist.twist.linear.x,
            angular_z=msg.twist.twist.angular.z))

    node.create_subscription(CompressedImage, '/pinky/camera/image_raw/compressed',
                             raw, qos_profile_sensor_data)
    node.create_subscription(Image, '/lane_autonomy/debug_image', debug, qos_profile_sensor_data)
    node.create_subscription(String, '/lane_autonomy/status',
        lambda msg: sample('status', dict(text=msg.data)), 10)
    node.create_subscription(Twist, '/cmd_vel', velocity, qos_profile_sensor_data)
    node.create_subscription(Odometry, '/odom', odom, qos_profile_sensor_data)
    try:
        while time.monotonic()-started < args.seconds:
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        (output/'capture.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        statuses = [p['text'] for p in record['samples'] if p['kind'] == 'status']
        (output/'status.log').write_text('\n'.join(statuses)+'\n', encoding='utf-8')
        print(json.dumps(dict(output=str(output), frames=record['frames'],
                              last_status=statuses[-1] if statuses else None)))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
