"""Supervised AUTOMATIC Gazebo-only lane trial and evidence recorder.

No hardware host. Requires separate domain + /lane_sim namespace. Commands
are exclusively the simulated controller's enable/disable service; this tool
never publishes velocities. A trial can FAIL, and records that explicitly.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import SetBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinky_move.metric_lane import nearest_on_chain, arc_stations
from pinky_move.video_export import make_browser_mp4

SIM_DDS = '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', type=int, default=172)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--course-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.domain in (20,22,52) or not 0 <= args.domain <= 232 or not 1 <= args.seconds <= 300:
        parser.error('isolated simulation domain and bounded duration required')
    deadline = time.monotonic()+20.
    while not (args.course_dir/'course.json').is_file() and time.monotonic()<deadline:
        time.sleep(.1)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.course_dir/'course.json').read_text())
    calibration = json.loads((args.course_dir/'simulation_calibration.json').read_text())
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), CYCLONEDDS_URI=SIM_DDS)
    rclpy.init(args=[])
    node = rclpy.create_node('gazebo_lane_trial', enable_rosout=False)
    observations = dict(raw_frames=0, debug_frames=0, status=[], cmd=[], odom=[], ground_truth=[])
    started = time.monotonic()
    client = node.create_client(SetBool, '/lane_sim/lane_autonomy/enable')
    camera_matrix = None
    last_frame = None
    videos = {}

    def image_callback(message, debug=False):
        nonlocal last_frame
        channels = 3
        data = np.frombuffer(message.data, np.uint8).reshape(message.height,message.step)
        pixels = data[:, :message.width*channels].reshape(message.height,message.width,channels)
        if message.encoding == 'rgb8':
            pixels = cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
        elif message.encoding != 'bgr8':
            return
        key = 'debug_frames' if debug else 'raw_frames'
        observations[key] += 1
        if key not in videos:
            videos[key] = cv2.VideoWriter(str(args.output/(key+'.mp4')),
                cv2.VideoWriter_fourcc(*'mp4v'), 15., (message.width,message.height))
            if not videos[key].isOpened():
                raise RuntimeError('video capture writer failed: '+key)
        videos[key].write(pixels)
        if observations[key] == 1:
            cv2.imwrite(str(args.output/('first_debug.jpg' if debug else 'first_camera.jpg')), pixels)
        if observations[key]%30 == 0:
            cv2.imwrite(str(args.output/('latest_debug.jpg' if debug else 'latest_camera.jpg')), pixels)
        if not debug:
            last_frame = pixels.copy()

    def camera_info(message):
        nonlocal camera_matrix
        camera_matrix = np.asarray(message.k).reshape(3,3)

    def odom(message, kind):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        observations[kind].append(dict(t=time.monotonic()-started,x=p.x,y=p.y,z=p.z,yaw=yaw,
            stamp=message.header.stamp.sec+message.header.stamp.nanosec/1e9,
            frame=message.header.frame_id,child=message.child_frame_id))

    node.create_subscription(Image,'/lane_sim/camera/image_raw',image_callback,qos_profile_sensor_data)
    node.create_subscription(Image,'/lane_sim/debug_image',lambda m:image_callback(m,True),qos_profile_sensor_data)
    node.create_subscription(CameraInfo,'/lane_sim/camera/camera_info',camera_info,qos_profile_sensor_data)
    node.create_subscription(Odometry,'/lane_sim/odom',lambda m:odom(m,'odom'),qos_profile_sensor_data)
    node.create_subscription(Odometry,'/lane_sim/ground_truth',lambda m:odom(m,'ground_truth'),qos_profile_sensor_data)
    node.create_subscription(Twist,'/lane_sim/cmd_vel',lambda m:observations['cmd'].append(
        dict(t=time.monotonic()-started,v=m.linear.x,w=m.angular.z)),10)
    node.create_subscription(String,'/lane_sim/lane_autonomy/status',lambda m:observations['status'].append(
        dict(t=time.monotonic()-started,text=m.data)),10)
    enabled = False
    error = None
    try:
        deadline = time.monotonic()+30.
        while time.monotonic()<deadline and (not observations['raw_frames'] or not observations['ground_truth'] or
                                             camera_matrix is None or not client.service_is_ready()):
            rclpy.spin_once(node,timeout_sec=.1)
        if not observations['raw_frames'] or not observations['ground_truth'] or not client.service_is_ready():
            raise RuntimeError('simulation readiness timeout: camera/ground truth/enable service missing')
        if camera_matrix is None or not np.allclose(camera_matrix,np.asarray(calibration['camera_matrix']),atol=1.):
            raise RuntimeError('Gazebo camera intrinsics not received or differ from simulation calibration')
        request = SetBool.Request(data=True)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=5.)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('simulated controller refused enable: '+str(future.result() if future.done() else 'timeout'))
        enabled = True
        trial_start = time.monotonic()
        while time.monotonic()-trial_start < args.seconds:
            rclpy.spin_once(node,timeout_sec=.05)
            if observations['ground_truth']:
                current = observations['ground_truth'][-1]
                if np.linalg.norm(np.array([current['x'], current['y']])-
                                  np.asarray(metadata['centre'][-1])) < .10:
                    break  # End at physical checkpoint, not at painted stripe end.
    except Exception as exc:
        error = str(exc)
    finally:
        if client.service_is_ready():
            future = client.call_async(SetBool.Request(data=False))
            rclpy.spin_until_future_complete(node,future,timeout_sec=3.)
        if last_frame is not None:
            cv2.imwrite(str(args.output/'final_camera.jpg'),last_frame)
        for video in videos.values():
            video.release()
        node.destroy_node()
        rclpy.shutdown()
    points = np.array([[row['x'],row['y']] for row in observations['ground_truth']])
    summary = dict(course=metadata['course'],physics_trial=True,yolo_enabled=enabled,
                   perception=metadata.get('perception', 'unspecified (see run notes)'),
                   lane_count=metadata.get('lane_count', 1),
                   stripe_count=metadata.get('stripe_count', 2),
                   error=error,raw_frames=observations['raw_frames'],debug_frames=observations['debug_frames'],
                   camera_intrinsics=None if camera_matrix is None else camera_matrix.tolist(),
                   nonzero_commands=sum(abs(r['v'])+abs(r['w'])>1e-6 for r in observations['cmd']),
                   status_counts=dict(Counter(r['text'].split(':')[0] for r in observations['status'])))
    if len(points)>1:
        centre = np.asarray(metadata['centre'])
        nearest, indices, fraction = nearest_on_chain(points,centre)
        stations = arc_stations(centre)
        progress = stations[indices]+fraction*(stations[indices+1]-stations[indices])
        travelled = float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        error_m = np.linalg.norm(points-nearest,axis=1)
        summary.update(travelled_m=travelled,net_displacement_m=float(np.linalg.norm(points[-1]-points[0])),
            max_cross_track_m=float(error_m.max()),final_progress_m=float(progress[-1]-progress[0]),
            final_pose=observations['ground_truth'][-1],
            reached_finish=bool(np.linalg.norm(points[-1]-centre[-1])<.10),
            lane_departure=bool(np.any(error_m>metadata['lane_width_m']/2)))
    summary['passed'] = bool(error is None and enabled and summary.get('reached_finish') and not summary.get('lane_departure'))
    exports = []
    for key in videos:
        try:
            exports.append(make_browser_mp4(args.output/(key+'.mp4')))
        except Exception as exc:
            exports.append(dict(path=key+'.mp4',error=str(exc)))
    summary['video_exports'] = exports
    (args.output/'trial.json').write_text(json.dumps(dict(summary=summary,observations=observations),indent=2))
    (args.output/'course.json').write_text(json.dumps(metadata,indent=2))
    (args.output/'simulation_calibration.json').write_text(json.dumps(calibration,indent=2))
    print(json.dumps(summary,indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
