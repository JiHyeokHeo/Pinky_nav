#!/usr/bin/env python3
"""YOLO segmentation lane follower for Pinky Pro."""

from concurrent.futures import Future
from array import array
from enum import Enum
import os
import json
import hashlib
import uuid
import time
from threading import Thread
from copy import deepcopy
from types import SimpleNamespace

import cv2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .lane_logic import (
    crossline_is_close,
    estimate_lane_center,
    lane_candidates_at,
    select_near_lane,
    steering_command,
    consistent_single_far_center,
    near_priority_command,
)
from .floor_projection import forward_cm
from .robot_projection import robot_floor_point, draw_metric_target, validate_calibration
from .metric_lane import (MetricLaneTracker, LaneImageIdentity, pursuit, near_pixel_side,
                          loss_speed_scale, floor_curves, curve_match_error,
                          supported_chain, arc_stations, connected_floor_curve)
from . import lane_wire
from .debug_worker import LatestDebugWorker, IsolatedImageTransport, DiagnosticPublisher
from .lane_history import LaneHistory, SpatialLaneArchive
from .local_map_view import render_local_map
from .lane_corner import CornerConfig, CornerPolicy, wrap, staged_command
from .white_lane import white_floor_masks, white_center_path, white_target


def draw_boundary_labels(frame, polygons, lane_class_id, labels, corner):
    """Annotate actual assigned roles, not a guess from screen left/right."""
    h, w = frame.shape[:2]
    index = 0
    occupied = []
    for class_id, polygon in polygons:
        if class_id != lane_class_id:
            continue
        side, source = labels.get(index, ('unknown', 'UNASSIGNED'))
        text = f'#{index+1} {side.upper()} [{source}]'
        mode = f"STATE: {corner.get('mode', 'UNKNOWN')} {corner.get('direction', '')}"
        point = np.asarray(polygon).mean(axis=0).astype(int)
        x = max(2, min(w-310, int(point[0])-120))
        y = max(175, min(h-45, int(point[1])-12))
        while any(abs(y-old)<40 for old in occupied) and y+40 < h-40:
            y += 40
        occupied.append(y)
        color = {'left': (60,255,60), 'right': (255,180,40)}.get(side, (0,200,255))
        cv2.line(frame, tuple(point), (x,y), color, 1)
        for offset, line in enumerate((text, mode)):
            origin = (x,y+offset*17)
            cv2.putText(frame,line,origin,cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),3)
            cv2.putText(frame,line,origin,cv2.FONT_HERSHEY_SIMPLEX,.43,color,1)
        index += 1


class DriveState(Enum):
    """Externally visible controller states."""

    DISABLED = 'DISABLED'
    WAITING_FOR_LANE = 'WAITING_FOR_LANE'
    FOLLOWING = 'FOLLOWING'
    REACQUIRING_LANE = 'REACQUIRING_LANE'
    CROSSLINE_STOP = 'CROSSLINE_STOP'
    SAFETY_STOP = 'SAFETY_STOP'


class LaneAutonomy(Node):
    """Follow segmentation masks and stop once for each crossline."""

    def __init__(self):
        super().__init__('lane_autonomy')
        self._declare_parameters()
        self.floor_calibration = None
        self.robot_calibration = None
        self.calibration_block_reason = None
        calibration_path = self._string_parameter('calibration_path')
        if calibration_path:
            with open(calibration_path, encoding='utf-8') as stream:
                self.floor_calibration = json.load(stream)
            if self.floor_calibration.get('method') == 'intrinsics_urdf_floor':
                self.robot_calibration = self.floor_calibration
                validate_calibration(self.robot_calibration)
                if self.robot_calibration.get('frame_id') != 'base_link':
                    raise ValueError('Metric controller requires an explicit base_link calibration')
                self.floor_calibration = None
                self.get_logger().warn('URDF projection uses nominal mounting; physical accuracy remains unverified.')
            if self.floor_calibration and self.floor_calibration.get('method') == 'two_apriltag_cubes':
                self.calibration_block_reason = (
                    'Cube calibration not ready for steering: '
                    + self.floor_calibration.get('status', 'invalid'))
                self.floor_calibration = None
                self.get_logger().error(self.calibration_block_reason)
            if self.floor_calibration:
                self.get_logger().warn(
                    'Floor calibration is approximate: diagnostic camera-forward distance only; '
                    'mount/orientation must match calibration. Not used for steering or stopping.')

        model_path = self._string_parameter('model_path')
        self.remote_inference = self._bool_parameter('remote_inference')
        self._validate_yolo_white_profile()
        if self._bool_parameter('remote_geometry') and not self.remote_inference:
            raise ValueError('remote_geometry requires remote_inference')
        self.remote_session = uuid.uuid4().hex
        self.remote_sequence = 0
        self.remote_pending = None
        self.remote_mailbox = None
        self.remote_server = None
        if self.remote_inference:
            # Compare model identity without importing ultralytics/torch on robot.
            with open(model_path, 'rb') as stream:
                self.remote_model_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.model = None
            self.lane_class_id, self.crossline_class_id = 1, 0
            self.remote_mailbox = lane_wire.PerceptionMailbox()
        else:
            self._load_model(model_path)
        self._initialize_control(model_path)

    def _validate_yolo_white_profile(self):
        """실험 모드가 원격 경로나 실제 장비에서 우회 활성화되지 않게 한다."""
        if (self._bool_parameter('simulation_semantic_path') and
                not self._bool_parameter('simulation_yolo_white')):
            raise ValueError('semantic path comparison requires YOLO/white simulation')
        if self._bool_parameter('simulation_yolo_white'):
            if (self.context.get_domain_id() in (20, 22, 52) or
                    not self._string_parameter('image_topic').startswith('/lane_sim/') or
                    not self.robot_calibration or
                    self.robot_calibration.get('status') != 'simulation_only_ideal_camera'):
                raise ValueError('YOLO/white experiment requires isolated simulation')
            if (self._bool_parameter('simulation_white_lane') or
                    self._bool_parameter('remote_inference') or self._bool_parameter('remote_geometry')):
                raise ValueError('YOLO/white experiment uses only local YOLO, not white-only/remote mode')

    def _load_model(self, model_path):
        self._validate_yolo_white_profile()
        if self._bool_parameter('simulation_yolo_white'):
            from .yolo_white import YoloWhiteSupplement
            self.yolo_white = YoloWhiteSupplement(support_interval=(
                .08,  # 차선 관측은 PP 목표 최소 거리보다 가까워도 보존한다.
                self._float_parameter('metric_path_max_m')))
        if self._bool_parameter('simulation_white_lane'):
            if (self.context.get_domain_id() in (20, 22, 52) or
                    not self._string_parameter('image_topic').startswith('/lane_sim/') or
                    not self.robot_calibration or
                    self.robot_calibration.get('status') != 'simulation_only_ideal_camera'):
                raise ValueError('OpenCV experiment requires isolated simulation domain/camera/calibration')
            self.model = None
            self.lane_class_id, self.crossline_class_id = 1, 0
            return
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'YOLO model not found: {model_path}')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                'Ultralytics is not installed. Run: pip3 install ultralytics'
            ) from exc

        self.model = YOLO(model_path)
        if getattr(self.model, 'task', None) != 'segment':
            raise RuntimeError(
                f'{model_path} is task={self.model.task!r}; '
                'a segmentation model is required')
        self.lane_class_id = self._class_id('lane')
        self.crossline_class_id = self._class_id('crossline')

    def _initialize_control(self, model_path):
        self.enabled = bool(self.get_parameter('enabled').value)
        self.state = DriveState.DISABLED
        self.latest_linear = 0.0
        self.latest_angular = 0.0
        self.filtered_angular = 0.0
        self.last_image_time = None
        self.last_inference_time = None
        self.last_lane_time = None
        self.last_scan_time = None
        self.front_clear = not self._bool_parameter('use_lidar_guard')
        self.near_center = None
        self.active_near_y_ratio = self._float_parameter('near_y_ratio')
        self.far_center = None
        self.near_lane_width = None
        self.metric_target = None
        self.metric_missing_since = None
        self.metric_last_good_target = None
        self.crossline_streak = 0
        self.crossline_latched = False
        self.crossline_clear_since = None
        self.crossline_stop_until = None
        self.last_status_text = None
        self.last_status_time = None
        self.safety_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.latest_image = None
        self.inference_future = None
        self.inference_generation = 0
        self.last_result_input_time = None
        self.inference_error = None
        self.inference_seconds = 0.0
        self.lane_instances = 0
        self.crossline_instances = 0
        self.boundary_count = 0
        self.near_candidate_count = 0

        # Initialize identity/recovery state even while disabled, so the preview
        # can safely consume external segmentation before the operator enables.
        self._reset_transient_state()

        image_topic = self._string_parameter('image_topic')
        scan_topic = self._string_parameter('scan_topic')
        cmd_vel_topic = self._string_parameter('cmd_vel_topic')
        self.cmd_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.status_publisher = self.create_publisher(String, '~/status', 10)
        self.debug_publisher = self.create_publisher(
            Image, self._string_parameter('debug_image_topic'),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.local_map_publisher = self.create_publisher(
            Image, '~/local_map', QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.enable_service = self.create_service(
            SetBool, '~/enable', self._enable_callback)
        if self.remote_inference:
            self.remote_server = lane_wire.LoopbackPerceptionServer(
                self.remote_mailbox, self._int_parameter('remote_port'))
            self.get_logger().info('Remote segmentation mode: loopback-only SSH transport; no local YOLO.')
        # The robot's established cam_stream.py publishes JPEG, while other
        # launches/simulations still provide raw Image. Select at launch time.
        image_type = (CompressedImage if self._bool_parameter('image_compressed')
                      else Image)
        self.create_subscription(
            image_type, image_topic, self._image_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, scan_topic, self._scan_callback,
            qos_profile_sensor_data)
        self.create_subscription(Odometry, self._string_parameter('corner_odom_topic'),
                                 self._corner_odom_callback, qos_profile_sensor_data)

        frequency = max(1.0, self._float_parameter('control_frequency'))
        self.timer = self.create_timer(
            1.0 / frequency, self._control_loop, clock=self.safety_clock)
        self.get_logger().info(
            f'Lane autonomy ready and {"enabled" if self.enabled else "disabled"}. '
            f'perception={"YOLO + white supplement (simulation only)" if self._bool_parameter("simulation_yolo_white") else "OpenCV white pixels (simulation only)" if self._bool_parameter("simulation_white_lane") else "YOLO"}, '
            f'model={"not loaded" if self._bool_parameter("simulation_white_lane") else model_path}, '
            f'image={image_topic}, output={cmd_vel_topic}')
        if not self.enabled:
            self.get_logger().info(
                'Enable after placing the robot safely: '
                'ros2 service call /lane_autonomy/enable '
                'std_srvs/srv/SetBool "{data: true}"')

    def _declare_parameters(self):
        default_model = (
            '/home/tory/Downloads/yolo_runs/segment/train/weights/best.pt')
        parameters = (
            ('enabled', False),
            ('simulation_white_lane', False),
            ('simulation_yolo_white', False),
            ('simulation_semantic_path', False),
            ('connected_geometry', False),
            ('remote_inference', False),
            ('remote_geometry', False),
            ('remote_port', 18765),
            ('max_enabled_seconds', 0.0),
            ('model_path', default_model),
            ('calibration_path', ''),
            ('metric_trial_enabled', False),  # Deprecated compatibility parameter; no enable gate.
            ('metric_lookahead_m', .22),
            ('metric_single_line_timeout', 0.0),
            ('lane_tracking_gap_seconds', 2.0),
            ('lane_width', .154),  # metres; estimate from this course, replace with measurement
            ('path_polynomial_degree', 3),
            ('projection_max_forward_m', 2.0),
            ('metric_path_min_m', .14),
            ('metric_path_max_m', .48),
            ('near_pair_max_m', .28),
            ('corner_enabled', True),
            ('corner_staged_turn', True),
            ('corner_entry_speed_mps', .03),
            ('corner_pivot_tolerance_m', .025),
            ('corner_brake_seconds', .3),
            ('corner_spin_timeout_s', 20.),
            ('corner_odom_topic', '/odom'),
            ('corner_odom_timeout', .3),
            ('corner_angle_deg', 50.),
            ('corner_window_m', .12),
            ('corner_segment_m', .05),
            ('corner_confirm_frames', 3),
            ('corner_exit_frames', 3),
            ('corner_max_gap_s', .6),
            ('corner_relaxed_tracking', True),
            ('corner_approach_m', .30),
            ('corner_speed_mps', .05),
            ('corner_max_angular_speed', .25),
            ('corner_min_speed_mps', .003),
            ('corner_clearance_m', .005),
            ('corner_robot_front_m', .060),
            ('corner_robot_rear_m', .090),
            ('corner_robot_half_width_m', .075),
            ('single_line_turn_speed', .08),
            ('single_line_turn_delay_seconds', 3.0),
            ('single_line_turn_requires_path_loss', True),
            ('single_line_turn_seconds', 20.0),
            ('single_line_visibility_timeout', .6),
            ('linear_velocity', -1.0),  # -1 keeps legacy linear_speed parameter
            ('lookahead_distance', -1.0),  # -1 keeps metric_lookahead_m
            ('maximum_linear_speed', .2),
            ('single_line_max_speed', .03),
            ('lane_loss_hold_seconds', .3),
            ('lane_loss_stop_seconds', .8),
            ('lane_loss_max_speed', .01),
            ('image_topic', '/camera/image_raw'),
            ('image_compressed', False),
            ('scan_topic', '/scan'),
            ('cmd_vel_topic', '/cmd_vel'),
            ('debug_image_topic', '/lane_autonomy/debug_image'),
            ('control_frequency', 20.0),
            ('inference_frequency', 10.0),
            ('imgsz', 640),
            ('confidence', 0.55),
            ('iou', 0.70),
            ('device', ''),
            ('near_y_ratio', 0.78),
            ('adaptive_near_min_y_ratio', 0.66),
            ('adaptive_near_step', 0.04),
            ('far_y_ratio', 0.58),
            ('sample_band_ratio', 0.04),
            ('initial_lane_width_ratio', 0.55),
            ('far_lane_width_scale', 0.55),
            ('lane_width_alpha', 0.15),
            ('minimum_lane_width_ratio', 0.18),
            ('maximum_lane_width_ratio', 0.95),
            ('lateral_gain', 0.85),
            ('heading_gain', 1.15),
            ('near_priority_steering', True),
            ('near_center_deadband', 0.04),
            ('curve_slowdown_gain', 0.75),
            ('steering_alpha', 0.35),
            ('linear_speed', 0.1),
            ('minimum_linear_speed', 0.01),
            ('single_line_speed_scale', 0.70),
            ('maximum_angular_speed', 0.15),
            ('image_timeout', 1.50),
            ('lane_lost_timeout', 1.50),
            ('result_timeout', 2.0),
            ('crossline_stop_seconds', 3.0),
            ('crossline_trigger_y_ratio', 0.68),
            ('crossline_minimum_area_ratio', 0.002),
            ('crossline_confirm_frames', 2),
            ('crossline_release_seconds', 0.8),
            ('use_lidar_guard', False),
            ('lidar_timeout', 0.7),
            ('front_stop_distance', 0.25),
            ('front_sector_degrees', 25.0),
            ('publish_debug_image', True),
            ('debug_image_frequency', 3.0),
        )
        for name, value in parameters:
            self.declare_parameter(name, value)

    def _class_id(self, requested_name):
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for class_id, name in items:
            if str(name).casefold() == requested_name.casefold():
                return int(class_id)
        raise RuntimeError(
            f'Model class {requested_name!r} not found in {names!r}')

    def _string_parameter(self, name):
        return str(self.get_parameter(name).value)

    def _float_parameter(self, name):
        return float(self.get_parameter(name).value)

    def _int_parameter(self, name):
        return int(self.get_parameter(name).value)

    def _bool_parameter(self, name):
        return bool(self.get_parameter(name).value)

    def _metric_speed(self):
        value = self._float_parameter('linear_velocity')
        return value if value >= 0 else self._float_parameter('linear_speed')

    def _metric_lookahead(self):
        value = self._float_parameter('lookahead_distance')
        return value if value >= 0 else self._float_parameter('metric_lookahead_m')

    def _enable_callback(self, request, response):
        reason = self._calibration_stop_reason()
        if request.data and reason:
            self.enabled = False
            self._publish_zero()
            response.success = False
            response.message = reason
            return response
        self.enabled = bool(request.data)
        if (self.enabled and getattr(self, 'robot_calibration', None)
                and not self._bool_parameter('use_lidar_guard')):
            self.get_logger().warn('Lidar obstacle stopping DISABLED; camera/line loss and command watchdog remain active.')
        limit = self._float_parameter('max_enabled_seconds')
        self.enable_deadline_ns = (
            self.safety_clock.now().nanoseconds + int(limit * 1e9)
            if self.enabled and limit > 0 else None)
        self.set_parameters([
            Parameter('enabled', Parameter.Type.BOOL, self.enabled),
        ])
        self._reset_transient_state()
        response.success = True
        response.message = (
            'lane autonomy enabled' if self.enabled
            else 'lane autonomy disabled; zero velocity published')
        self._publish_zero()
        return response

    def _reset_transient_state(self):
        # An in-flight result from before enable/disable cannot resume motion.
        self.inference_generation += 1
        self.remote_pending = None
        if getattr(self, 'remote_mailbox', None) is not None:
            self.remote_mailbox.clear()
        self.latest_image = None
        self.last_result_input_time = None
        self.inference_error = None
        self.boundary_count = 0
        self.state = (
            DriveState.WAITING_FOR_LANE if self.enabled
            else DriveState.DISABLED)
        self.latest_linear = 0.0
        self.latest_angular = 0.0
        self.filtered_angular = 0.0
        self.last_lane_time = None
        self.near_center = None
        self.active_near_y_ratio = self._float_parameter('near_y_ratio')
        self.far_center = None
        self.near_lane_width = None
        self.crossline_streak = 0
        self.metric_target = None
        self.metric_tracker = MetricLaneTracker(
            connected_geometry=self._bool_parameter('connected_geometry'),
            tracking_gap_s=self._float_parameter('lane_tracking_gap_seconds'),
            minimum_lane_width_m=2*(self._float_parameter('corner_robot_half_width_m')+
                                    self._float_parameter('corner_clearance_m')))
        self.corner_policy = CornerPolicy(CornerConfig(
            relaxed_tracking=self._bool_parameter('corner_relaxed_tracking'),
            staged_turn=self._bool_parameter('corner_staged_turn'),
            entry_speed_mps=self._float_parameter('corner_entry_speed_mps'),
            pivot_tolerance_m=self._float_parameter('corner_pivot_tolerance_m'),
            brake_seconds=self._float_parameter('corner_brake_seconds'),
            spin_timeout_s=self._float_parameter('corner_spin_timeout_s'),
            angle_deg=self._float_parameter('corner_angle_deg'),
            window_m=self._float_parameter('corner_window_m'),
            segment_m=self._float_parameter('corner_segment_m'),
            confirm_frames=self._int_parameter('corner_confirm_frames'),
            exit_frames=self._int_parameter('corner_exit_frames'),
            max_gap_s=self._float_parameter('corner_max_gap_s'),
            approach_m=self._float_parameter('corner_approach_m'),
            speed_mps=self._float_parameter('corner_speed_mps'),
            min_speed_mps=self._float_parameter('corner_min_speed_mps'),
            clearance_m=self._float_parameter('corner_clearance_m'),
            front_m=self._float_parameter('corner_robot_front_m'),
            rear_m=self._float_parameter('corner_robot_rear_m'),
            half_width_m=self._float_parameter('corner_robot_half_width_m')))
        self.corner_odom = None
        self.corner_odom_stamp = None
        self.corner_odom_frame = None
        self.corner_odom_history = []
        self.corner_capture_stamp = None
        self.hybrid_frame_pose = None
        self.hybrid_white_map_side = None
        self.hybrid_last_semantic_time = None
        self.image_identity = LaneImageIdentity(self._float_parameter('lane_tracking_gap_seconds'))
        self.lane_history = LaneHistory(max_frames=5,
                                       max_age_s=self._float_parameter('lane_tracking_gap_seconds'))
        self.local_map_history = LaneHistory(max_frames=600, max_age_s=10.)
        if not hasattr(self, 'local_map_archive'):
            self.local_map_archive = SpatialLaneArchive()
        self.local_map_pose = None
        self.image_side_hint = None
        self.image_match = None
        self.metric_missing_since = None
        self.metric_last_good_target = None
        self.turn_started_at = None
        self.turn_side = None
        self.turn_seen_at = None
        self.turn_exhausted = False
        self.near_pair_streak = 0
        self.single_resume_observation = None
        self.single_side_since = None
        self.single_side_last_seen = None
        self.single_side_candidate = None
        self.crossline_latched = False
        self.crossline_clear_since = None
        self.crossline_stop_until = None

    def _corner_odom_callback(self, message):
        """Keep fresh planar odometry for landmark confirmation, not blind motion."""
        p, q = message.pose.pose.position, message.pose.pose.orientation
        stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
        if (not np.isfinite([p.x,p.y,q.x,q.y,q.z,q.w]).all() or
                abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) > .05 or
                message.child_frame_id not in ('base_link','base_footprint') or
                not message.header.frame_id):
            self.corner_odom = None
            return
        if self.corner_odom_stamp is not None and stamp <= self.corner_odom_stamp:
            return  # Duplicate/out-of-order samples cannot refresh age.
        if hasattr(self, 'get_clock'):
            age = (self.get_clock().now().nanoseconds-stamp)/1e9
            if not -.05 <= age <= self._float_parameter('corner_odom_timeout'):
                self.corner_odom = None
                return
        now = self.safety_clock.now().nanoseconds/1e9
        yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        pose = np.array([p.x,p.y,yaw])
        prior = self.corner_odom
        if (self.corner_odom_frame not in (None,message.header.frame_id) or
                (prior is not None and (np.linalg.norm(pose[:2]-prior[1][:2]) > .20 or
                                        abs(wrap(yaw-prior[1][2])) > .6))):
            self.corner_policy.reset()
            self.corner_odom_history = []
            self.lane_history.clear()
            self.local_map_history.clear()
            self.local_map_archive.clear()
            # World-frame pivot targets are invalid after an odometry reset.
            # Do not let the control timer use one until another fresh result.
            self._invalidate_lane()
            if self._bool_parameter('remote_geometry'):
                self.inference_generation += 1
                self.remote_pending = None
                if getattr(self, 'remote_mailbox', None) is not None:
                    self.remote_mailbox.clear()
        self.corner_odom = (now, pose)
        self.corner_odom_stamp = stamp
        self.corner_odom_frame = message.header.frame_id
        self.corner_odom_history.append((stamp, pose))
        self.corner_odom_history = [item for item in self.corner_odom_history
                                    if stamp-item[0] <= 2_000_000_000][-200:]

    def _corner_pose_for_frame(self, now):
        """Use current steady time for freshness, CAMERA time for pose lookup.

        `now` may be the inference worker's earlier completion time. A later
        odometry callback is not a future sample: comparing it to completion
        incorrectly rejected healthy odometry with a negative age.
        """
        current = self.safety_clock.now().nanoseconds/1e9
        odom = self.corner_odom
        self.corner_pose_diagnostic = dict(reason='odometry missing',
            processing_lag_s=current-now)
        if odom is None:
            return None
        age = current-odom[0]
        self.corner_pose_diagnostic.update(odom_age_s=age)
        if not 0 <= age <= self._float_parameter('corner_odom_timeout'):
            self.corner_pose_diagnostic['reason'] = 'odometry stale or future reception'
            return None
        stamp = self.corner_capture_stamp
        history = self.corner_odom_history
        if stamp is None or not history:
            self.corner_pose_diagnostic['reason'] = 'camera stamp or odometry history missing'
            return None
        for (ta,a),(tb,b) in zip(history,history[1:]):
            if ta <= stamp <= tb and tb-ta <= 300_000_000:
                f = (stamp-ta)/(tb-ta)
                self.corner_pose_diagnostic.update(reason='interpolated',
                                                   bracket_gap_s=(tb-ta)/1e9)
                return np.array([*(a[:2]*(1-f)+b[:2]*f), a[2]+f*wrap(b[2]-a[2])])
        closest = min(history,key=lambda item:abs(item[0]-stamp))
        gap = abs(closest[0]-stamp)
        self.corner_pose_diagnostic.update(nearest_camera_gap_s=gap/1e9,
            reason='nearest sample' if gap <= 50_000_000 else 'camera time not covered by odometry')
        return closest[1].copy() if gap <= 50_000_000 else None

    def _scan_callback(self, message):
        self.last_scan_time = self.safety_clock.now()
        sector = np.deg2rad(self._float_parameter('front_sector_degrees'))
        stop_distance = self._float_parameter('front_stop_distance')
        nearest = None
        for index, value in enumerate(message.ranges):
            angle = message.angle_min + index * message.angle_increment
            angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
            if ((not getattr(self, 'robot_calibration', None) and abs(angle) > sector)
                    or not np.isfinite(value)):
                continue
            if message.range_min <= value <= message.range_max:
                nearest = value if nearest is None else min(nearest, value)
        self.front_clear = ((nearest is not None and nearest > stop_distance)
                            if getattr(self, 'robot_calibration', None)
                            else nearest is None or nearest > stop_distance)

    def _image_callback(self, message):
        # Keep only the newest frame; never run YOLO in a ROS callback.
        now = self.safety_clock.now()
        self.last_image_time = now
        self.latest_image = (message, now, self.inference_generation)

    def _start_inference(self, now):
        if getattr(self, 'remote_inference', False):
            self._start_remote_inference(now)
            return
        if self.inference_future is not None or self.latest_image is None:
            return
        inference_period = 1.0 / max(
            0.1, self._float_parameter('inference_frequency'))
        if self.last_inference_time is not None:
            age = (now - self.last_inference_time).nanoseconds / 1e9
            if age < inference_period:
                return
        self.last_inference_time = now
        message, received, generation = self.latest_image
        self.latest_image = None
        if (now - received).nanoseconds / 1e9 > self._float_parameter(
                'image_timeout'):
            return
        predict_arguments = {
            'imgsz': self._int_parameter('imgsz'),
            'conf': self._float_parameter('confidence'),
            'iou': self._float_parameter('iou'),
            'retina_masks': True,
            'verbose': False,
        }
        device = self._string_parameter('device').strip()
        if device:
            predict_arguments['device'] = device
        future = Future()
        self.inference_future = (future, received, generation)

        def predict():
            # Only the mailbox is shared. No state changes or ROS publishing
            # happen on this thread, including after node shutdown.
            try:
                frame = self._image_to_bgr(message).copy()
                result = (SimpleNamespace(masks=None, boxes=None) if self.model is None else
                          self.model.predict(source=frame, **predict_arguments)[0])
                if self._bool_parameter('simulation_yolo_white'):
                    from .lane_wire import ClassIds
                    ids = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
                    original = list(result.masks.xy) if result.masks is not None else []
                    binary = []
                    for class_id, polygon in zip(ids, original):
                        if class_id == self.lane_class_id:
                            m = np.zeros(frame.shape[:2], np.uint8)
                            cv2.fillPoly(m, [np.asarray(polygon, np.int32)], 1)
                            binary.append(m)
                    complete, supplement = self.yolo_white.update(frame, binary,
                        received.nanoseconds/1e9, self.robot_calibration)
                    polygons, classes = [], []
                    for m in complete:
                        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            polygons.append(max(contours, key=cv2.contourArea).reshape(-1, 2))
                            classes.append(self.lane_class_id)
                    for class_id, polygon in zip(ids, original):
                        if class_id != self.lane_class_id:
                            polygons.append(polygon)
                            classes.append(class_id)
                    result = SimpleNamespace(masks=SimpleNamespace(xy=polygons),
                        boxes=SimpleNamespace(cls=ClassIds(classes)), supplement=supplement)
                    if self._bool_parameter('simulation_semantic_path'):
                        from .white_lane import semantic_center_path
                        try:
                            result.white_path, result.white_selected, result.white_side = semantic_center_path(
                                complete, self.robot_calibration, self._float_parameter('lane_width'))
                        except ValueError:
                            result.white_path, result.white_selected, result.white_side = None, {}, None
                if self.model is None:
                    # CPU-heavy thinning belongs to the perception worker,
                    # never the executor thread receiving odometry/control.
                    result.white_candidates = []
                    for binary in white_floor_masks(frame, self.robot_calibration):
                        curves = floor_curves([binary], self.robot_calibration, 1.2,
                                              self._float_parameter('metric_path_min_m'),
                                              self._float_parameter('metric_path_max_m'))
                        if len(curves) == 1:
                            connected = connected_floor_curve(binary, self.robot_calibration, 1.2, pixel_step=4)
                            result.white_candidates.append((binary, curves[0], connected))
                    try:
                        result.white_path, result.white_selected, result.white_side = white_center_path(
                            result.white_candidates, self._float_parameter('lane_width'))
                    except ValueError:
                        result.white_path, result.white_selected, result.white_side = None, {}, None
                future.set_result((
                    frame, result, self.safety_clock.now(), message.header))
            except Exception as exc:
                future.set_exception(exc)

        self.inference_thread = Thread(target=predict, name='lane_inference', daemon=False)
        self.inference_thread.start()

    def _start_remote_inference(self, now):
        """Retain the original frame/time; export only JPEG over the SSH tunnel."""
        if self.remote_pending is not None:
            if (now-self.remote_pending[2]).nanoseconds/1e9 < self._float_parameter('result_timeout'):
                return
            self.remote_pending = None
            self.remote_mailbox.clear()
            self._invalidate_lane()
            self.inference_error = 'remote inference timeout'
        if self.latest_image is None:
            return
        if (self.last_inference_time is not None and
                (now-self.last_inference_time).nanoseconds/1e9 <
                1./max(.1, self._float_parameter('inference_frequency'))):
            return
        message, received, generation = self.latest_image
        self.latest_image = None
        try:
            # Camera is on this same robot/ROS clock. Convert its timestamp to
            # steady-clock age before transmission; never trust a PC timestamp.
            stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
            capture_age = (self.get_clock().now().nanoseconds-stamp)/1e9
            if not 0 <= capture_age < min(self._float_parameter('image_timeout'),
                                         self._float_parameter('result_timeout')):
                raise ValueError('camera capture timestamp stale or future')
            captured = now-Duration(seconds=capture_age)
            frame = self._image_to_bgr(message).copy()
            self.remote_sequence += 1
            token = f'{self.remote_session}:{self.remote_sequence}'
            payload = lane_wire.encode_request(frame, token, message.header,
                                               self.remote_model_sha256)
            if self._bool_parameter('remote_geometry'):
                from .lane_planning_remote import plain
                self.corner_capture_stamp = stamp
                context = plain(dict(generation=generation, enabled=self.enabled,
                    now_ns=now.nanoseconds, pose=self._corner_pose_for_frame(now.nanoseconds/1e9),
                    calibration=self.robot_calibration, staged=self.corner_policy.staged,
                    parameters={name: self.get_parameter(name).value for name in self._parameters}))
                data = json.loads(payload)
                data['planning'] = context
                payload = json.dumps(data, allow_nan=False).encode()
                if len(payload) > lane_wire.MAX_MESSAGE:
                    raise ValueError('planning request too large')
                self.remote_plan_context = context
            self.remote_pending = (token, generation, captured, frame, message.header)
            self.last_inference_time = now
            self.remote_mailbox.offer(token, payload)
        except Exception as exc:
            self._invalidate_lane()
            self.inference_error = str(exc)

    def _consume_remote_inference(self, now):
        """Only exact, fresh, single-use replies may enter the existing controller."""
        data = self.remote_mailbox.take_reply()
        if data is None or self.remote_pending is None:
            return
        token, generation, captured, frame, header = self.remote_pending
        if data['token'] != token or generation != self.inference_generation:
            return
        self.remote_pending = None
        processing_started = time.monotonic()
        age = (now-captured).nanoseconds/1e9
        try:
            if not 0 <= age < self._float_parameter('result_timeout'):
                raise ValueError('remote result expired')
            result = lane_wire.decode_result(data, frame.shape[1], frame.shape[0],
                                             self.remote_model_sha256)
            if self._bool_parameter('remote_geometry'):
                from .lane_planning_remote import validate_plan
                stamp = header.stamp.sec*1_000_000_000+header.stamp.nanosec
                result.remote_plan = validate_plan(data.get('plan'), token, generation, stamp)
            self.inference_seconds = age
            self.last_result_input_time = captured
            self.inference_error = None
            # Saved raw frame is paired with these exact pixel coordinates.
            self._process_result(frame, result, now, header)
        except Exception as exc:
            self._invalidate_lane()
            self.inference_error = str(exc)
            self.get_logger().warn('Remote perception rejected: '+str(exc), throttle_duration_sec=2.)
        finally:
            pc = data.get('pc_processing_seconds')
            pc_text = (f'{pc:.3f}s' if isinstance(pc, (int, float))
                       and np.isfinite(pc) and pc >= 0 else 'unavailable')
            self.get_logger().info(
                f'Perception timing: input_age_at_reply={age:.3f}s, '
                f'pc_processing={pc_text}, '
                f'planning={"PC" if self._bool_parameter("remote_geometry") else "robot"}, '
                f'robot_postprocess={time.monotonic()-processing_started:.3f}s, '
                f'geometry={getattr(self, "last_geometry_seconds", 0.):.3f}s, '
                f'debug_enqueue={getattr(self, "last_debug_seconds", 0.):.3f}s, '
                f'debug_worker={getattr(getattr(self, "debug_worker", None), "last_seconds", 0.):.3f}s, '
                f'debug_publish={getattr(getattr(self, "debug_transport", None), "last_seconds", 0.):.3f}s, '
                f'debug_dropped={getattr(getattr(self, "debug_transport", None), "dropped", 0)}',
                throttle_duration_sec=5.)

    def _consume_inference(self, now):
        if getattr(self, 'remote_inference', False):
            self._consume_remote_inference(now)
            return
        if self.inference_future is None:
            return
        future, received, generation = self.inference_future
        if not future.done():
            return
        self.inference_future = None
        if generation != self.inference_generation:
            return
        try:
            frame, result, completed, header = future.result()
            self.inference_seconds = (completed - received).nanoseconds / 1e9
            self.last_result_input_time = received
            if (now - received).nanoseconds / 1e9 > self._float_parameter(
                    'result_timeout'):
                self._invalidate_lane()
                self.crossline_streak = 0
                self.crossline_clear_since = None
                return
            self.inference_error = None
            # A successful detection is dated at completion, not at the start
            # of CPU inference. Input age is guarded separately.
            self._process_result(frame, result, completed, header)
        except Exception as exc:
            self._invalidate_lane()
            self.inference_error = str(exc)
            self.crossline_streak = 0
            self.crossline_clear_since = None
            self.get_logger().error(
                f'Lane inference failed: {exc}', throttle_duration_sec=2.0)

    def _invalidate_lane(self):
        self.latest_linear = 0.0
        self.latest_angular = 0.0
        self.filtered_angular = 0.0
        self.last_lane_time = None
        self.boundary_count = 0
        self.near_candidate_count = 0
        self.near_center = None
        self.far_center = None
        self.metric_target = None
        self.metric_missing_since = None
        self.metric_last_good_target = None

    def _stage_white_mask(self, frame, now):
        """최근 YOLO 경계/확정 코너에 대응하는 현재 흰 경계만 보충한다.

        한쪽 경계가 사라져 반대쪽 선을 모델이 놓쳐도, 고정 코너의 실제/
        폭 기반 경계와 실좌표로 유일하게 대응하면 같은 차선의 관측이다.
        코너 확정 전에는 마지막 모델 확인 10초 이내, 두 독립 실관측의
        지도 역할에 유일하게 대응해야 한다. 흰색만으로 초기화하지 않는다.
        """
        if (not self._bool_parameter('simulation_yolo_white') or
                not self.robot_calibration or
                self.robot_calibration.get('method') != 'intrinsics_urdf_floor'):
            return None
        self.hybrid_white_map_side = None
        stage = self.corner_policy.staged
        semantic_time = getattr(self, 'hybrid_last_semantic_time', None)
        if stage is None and (semantic_time is None or
                not 0 <= now.nanoseconds/1e9-semantic_time <= 10.):
            return None
        pose = self._geometry_frame_pose(now)
        if pose is None:
            return None
        from .white_lane import white_floor_masks
        masks = [m for m in white_floor_masks(frame, self.robot_calibration)
                 if 60 <= np.count_nonzero(m) <= .18*frame.shape[0]*frame.shape[1]]
        if stage is None:
            mapped = []
            for index, mask in enumerate(masks):
                curves = connected_floor_curve(mask, self.robot_calibration, 2.)
                if len(curves)==1:
                    side = self.corner_policy.local_lane_map.measured_side(
                        curves[0], now.nanoseconds/1e9, pose)
                    if side is not None:
                        mapped.append((index,side))
            if len(mapped)!=1:
                return None
            self.hybrid_white_map_side = mapped[0][1]
            return masks[mapped[0][0]]
        observation = self.corner_policy.staged_observation(masks, self.robot_calibration, pose)
        if observation is None:
            return None
        return masks[observation['source_mask_index']]

    def _hybrid_connected_observation(self, masks, observation, ordinary):
        """현재 배정된 경계의 연결된 굽힘을 축 피팅이 잘라버리지 않게 한다.

        좌우/폭은 기존 추적기의 실제 관측에서만 가져온다. 연결 골격이
        현재 배정된 선과 유일하게 대응하고 실제 두 다리/S 굽힘을 가진
        경우만 코너 관측을 보충한다. 모호한 후보는 기존 처리를 유지한다.
        """
        self.hybrid_connected_trace=dict(masks=len(masks),
            stage_enabled=bool(self.corner_policy.config.staged_turn),
            observed_side=None if observation is None else observation['side'],candidates=[])
        if observation is None or not self._bool_parameter('simulation_yolo_white'):
            return observation, ordinary
        from .metric_lane import connected_floor_curve, select_lookahead
        from .lane_corner import classify_boundary, match_corner_fragment, first_sharp_corner_prefix
        from .connected_path import connected_miter_center
        minimum_entry=(self.corner_policy.config.segment_m if
                       getattr(self.corner_policy,'s_route',None) is not None else .12)
        def aligned_entry(prefix):
            if prefix is None:
                return False
            feature=classify_boundary(prefix,self.corner_policy.config)
            # 단 두 픽셀의 계단 접선이 아니라 검증된 진입 다리의 TLS 접선.
            return (feature['kind']=='CORNER' and
                    abs(np.arctan2(feature['tin'][1],feature['tin'][0]))<=np.deg2rad(15))
        # 기존 추적기가 이미 전체 S를 실관측으로 반환한 경우에는 잘못된
        # 축 기준에 재대응시키기 전에 그 관측의 첫 직각을 분리한다.
        if (self.corner_policy.config.staged_turn and
                classify_boundary(observation['curve'],self.corner_policy.config)['kind']=='S_BEND'):
            prefix=first_sharp_corner_prefix(observation['curve'],self.corner_policy.config,minimum_entry)
            if aligned_entry(prefix):
                self.hybrid_connected_trace['selected']='direct sharp prefix'
                self.metric_tracker.preferred_observation_side=observation['side']
                return dict(observation,curve=prefix,sharp_s_prefix=True),ordinary
        references = [(observation['side'], observation['curve'])]
        if observation.get('other') is not None and observation.get('width_source') == 'measured':
            references.append(('right' if observation['side']=='left' else 'left', observation['other']))
        candidates = []
        for index, mask in enumerate(masks):
            curves = connected_floor_curve(mask, self.robot_calibration, 2.)
            if len(curves) != 1:
                continue
            curve = curves[0]
            feature = classify_boundary(curve, self.corner_policy.config)
            trace=dict(index=index,kind=feature['kind'])
            self.hybrid_connected_trace['candidates'].append(trace)
            # 축 피팅의 먼 다리가 잘못 보간돼도 두 실경계의 가까운
            # 10cm가 1cm 이내로 유일하게 대응하면 그 마스크 역할은 같다.
            # 화면 좌우나 가장 아래 픽셀만으로 역할을 새로 배정하지 않는다.
            from .metric_lane import arc_stations
            try:
                # 골격 픽셀의 계단 접선은 역할 대응 전에 관측 오차 8mm
                # 이내로 정리한다. 위치/형상 검사는 connected_miter가 맡는다.
                matched_curve,_=connected_miter_center(curve,0.)
                near=matched_curve[arc_stations(matched_curve)<=.10]
            except ValueError as exc:
                trace['near_error']=str(exc)
                near=np.empty((0,2))
            def assigned(reference):
                if match_corner_fragment(reference,curve)['fragment_ok']:
                    return True
                match=match_corner_fragment(near,reference)
                return (match['fragment_ok'] and match.get('fragment_strong') and
                        match['fragment_error_m']<=.01)
            sides = ([observation['side']] if len(masks)==1 and len(references)==1 else
                     [side for side, reference in references
                      if assigned(reference)])
            trace['roles']=sides
            if feature['kind'] in ('CORNER', 'S_BEND') and len(sides)==1:
                candidates.append((index, curve, feature['kind'], sides[0]))
        # 기존 쌍의 두 실경계 중 S 전체가 보이는 선이 하나면 그것을 쓴다.
        # 선의 역할은 쌍에서 확인한 그대로이며 화면 위치로 다시 정하지 않는다.
        s_candidates = [c for c in candidates if c[2]=='S_BEND']
        if len(s_candidates)==1:
            candidates = s_candidates
        elif not s_candidates:
            candidates = [c for c in candidates if c[3]==observation['side']]
        if len(candidates) != 1:
            self.hybrid_connected_trace['selected']='non-unique or missing connected candidate'
            return observation, ordinary
        index, curve, kind, side = candidates[0]
        sharp_prefix = None
        if (self.corner_policy.config.staged_turn and
                (kind == 'S_BEND' or (kind == 'CORNER' and
                    getattr(self.corner_policy,'s_route',None) is not None))):
            sharp_prefix=first_sharp_corner_prefix(curve,self.corner_policy.config,minimum_entry)
            self.hybrid_connected_trace['rdp_vertices']=cv2.approxPolyDP(
                curve.astype(np.float32).reshape(-1,1,2),.008,False).reshape(-1,2).tolist()
            self.hybrid_connected_trace['prefix_exists']=sharp_prefix is not None
            if sharp_prefix is not None:
                pf=classify_boundary(sharp_prefix,self.corner_policy.config)
                self.hybrid_connected_trace['prefix_feature']=pf['kind']
                if pf['kind']=='CORNER':
                    self.hybrid_connected_trace['prefix_entry_deg']=float(np.rad2deg(np.arctan2(
                        pf['tin'][1],pf['tin'][0])))
            if aligned_entry(sharp_prefix):
                curve,kind=sharp_prefix,'CORNER'
            else:
                sharp_prefix=None
        measured = dict(observation, curve=curve, side=side, source_mask_index=index)
        if sharp_prefix is not None:
            measured['sharp_s_prefix']=True
            self.metric_tracker.preferred_observation_side=side
        self.hybrid_connected_trace.update(selected=kind,sharp_prefix=sharp_prefix is not None,
                                          side=side)
        measured.pop('other', None)  # 다른 경계의 축 피팅을 연결 S와 임의로 짝짓지 않는다.
        if (kind == 'S_BEND' or getattr(self.corner_policy, 's_route', None) is not None or
                (kind == 'CORNER' and self.corner_policy.config.staged_turn and ordinary is None)):
            try:
                _, center = connected_miter_center(curve, observation['width']*
                    (-.5 if side=='left' else .5))
                point, adaptive = select_lookahead(center, self._metric_lookahead())
                if point[0] <= .02:
                    raise ValueError('connected S target is not forward')
            except ValueError:
                return observation, ordinary
            ordinary = dict(x_m=float(point[0]), y_m=float(point[1]), center_path=center.tolist(),
                actual_curve=curve.tolist(), inferred=True, boundary_count=1,
                visible_side=side, source_mask_index=index,
                width_m=observation['width'], normal_width_m=observation['width'],
                width_source=observation.get('width_source', 'configured'), adaptive=adaptive,
                path_note='current assigned connected boundary')
            # 다음 실제 쌍에서도 S를 시작한 경계를 주 관측으로 선택한다.
            # 선의 좌우 역할을 바꾸는 것이 아니라 이미 확인된 선택을 유지한다.
            self.metric_tracker.preferred_observation_side = side
        return measured, ordinary

    def _geometry_frame_pose(self, now):
        """영상 처리 시작 때 검증한 촬영 시각 pose를 같은 영상에 재사용한다.

        처리 중 odom callback이 지연돼도 정확한 촬영 pose를 없애지 않는다.
        실제 모터 명령에는 별도의 현재 odom freshness 검사가 계속 적용된다.
        """
        cached = getattr(self, 'hybrid_frame_pose', None)
        if (self._bool_parameter('simulation_yolo_white') and cached is not None and
                cached[0] == self.corner_capture_stamp):
            return cached[1]
        return self._corner_pose_for_frame(now.nanoseconds/1e9)

    def _process_result(self, frame, result, now, source_header):
        geometry_started = time.monotonic()
        self.corner_capture_stamp = (source_header.stamp.sec*1_000_000_000+
                                     source_header.stamp.nanosec)
        self.hybrid_frame_pose = None
        if self._bool_parameter('simulation_yolo_white'):
            self.hybrid_frame_pose = (self.corner_capture_stamp,
                                     self._corner_pose_for_frame(now.nanoseconds/1e9))
        height, width = frame.shape[:2]
        self.perception_source = 'YOLO'
        lane_masks = []
        crossline_masks = []
        polygons_by_class = []
        masks = result.masks
        class_ids = (
            result.boxes.cls.int().cpu().tolist()
            if result.boxes is not None else [])
        self.lane_instances = class_ids.count(self.lane_class_id)
        self.crossline_instances = class_ids.count(self.crossline_class_id)
        if masks is not None and result.boxes is not None:
            for class_id, polygon in zip(class_ids, masks.xy):
                polygon = np.asarray(polygon, dtype=np.int32)
                if polygon.shape[0] < 3:
                    continue
                binary_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [polygon], 1)
                polygons_by_class.append((class_id, polygon))
                if class_id == self.lane_class_id:
                    lane_masks.append(binary_mask)
                elif class_id == self.crossline_class_id:
                    crossline_masks.append(binary_mask)

        if (self._bool_parameter('simulation_white_lane') or
                self._bool_parameter('simulation_semantic_path')):
            if not self._string_parameter('image_topic').startswith('/lane_sim/'):
                raise ValueError('white-pixel experiment requires isolated /lane_sim camera')
            # Camera-only centreline pursuit intentionally bypasses the
            # hardware staged-corner identity gates in this isolated profile.
            lane_masks = [item[1] for item in result.white_selected.values()]
            semantic = self._bool_parameter('simulation_semantic_path')
            self.perception_source = 'YOLO_SEMANTIC_PATH_SIM' if semantic else 'OPENCV_WHITE_SIM'
            if semantic:
                self.supplement_debug = dict(result.supplement)
            if not semantic:
                self.lane_instances = len(lane_masks)
            self.corner_policy.debug = {'mode': 'WHITE_PATH', 'reason': 'connected normal-offset centre'}
            self.corner_policy.staged = None
            self.sim_white_path = result.white_path
            self.sim_white_side = result.white_side
            count = self._update_lane_command(lane_masks, width, now)
            self.boundary_count = count
            self.debug_boundary_labels = {i: (side, 'USED') for i, side in enumerate(result.white_selected)}
            polygons_by_class = []
            for binary in lane_masks:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons_by_class.append((self.lane_class_id, max(contours, key=cv2.contourArea).reshape(-1, 2)))
            self.last_geometry_seconds = time.monotonic()-geometry_started
            if self._bool_parameter('publish_debug_image'):
                self._queue_debug(frame, polygons_by_class, count, source_header)
            return

        if self._bool_parameter('simulation_yolo_white'):
            self.perception_source = 'YOLO_WHITE_SIM'
            self.supplement_debug = dict(result.supplement)
            if self.supplement_debug.get('yolo',0)>0:
                self.hybrid_last_semantic_time = now.nanoseconds/1e9
            self.hybrid_white_map_side = None
            measured = (self._stage_white_mask(frame, now) if
                        self.corner_policy.staged is not None or not lane_masks else None)
            if measured is not None:
                # 제어는 동일한 MetricLaneTracker/CornerPolicy를 거친다.
                # YOLO 통계는 보존하고 실제 사용 픽셀을 별도로 표시한다.
                lane_masks = [measured]
                self.lane_instances = 1
                self.supplement_debug['white_map_geometry' if self.hybrid_white_map_side else
                                      'white_stage_geometry'] = 1
                polygons_by_class = [item for item in polygons_by_class if item[0] != self.lane_class_id]
                contours, _ = cv2.findContours(measured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons_by_class.append((self.lane_class_id, max(contours, key=cv2.contourArea).reshape(-1,2)))

        if self.enabled:
            self._update_crossline(crossline_masks, self.safety_clock.now())
        if self._bool_parameter('remote_geometry') and getattr(self, 'remote_inference', False):
            self._process_planned_result(result.remote_plan, now)
            self.local_map_pose = self._corner_pose_for_frame(now.nanoseconds/1e9)
            self.local_map_history.remember(self.corner_capture_stamp, self.local_map_pose,
                                           self.metric_tracker.last_observation)
            self.local_map_archive.remember(self.corner_capture_stamp, self.local_map_pose,
                                           self.metric_tracker.last_observation)
            self.last_geometry_seconds = time.monotonic()-geometry_started
            if self._bool_parameter('publish_debug_image'):
                self._queue_debug(frame, polygons_by_class, self.boundary_count, source_header)
            return
        image_match = None
        if self.enabled and getattr(self, 'robot_calibration', None):
            image_match = self.image_identity.match(frame, lane_masks, now.nanoseconds/1e9)
        self.image_side_hint = (image_match[1] if image_match and len(lane_masks) == 1
                                else None)
        self.image_match = image_match
        # If optical-flow association is unavailable, use odometry-aligned
        # measured history as a role hint BEFORE the single tracker update.
        # Never process the same frame twice: that would double confirmation
        # counters and violate CornerPolicy's strictly increasing timestamps.
        self.history_match = None
        if (self.enabled and getattr(self, 'robot_calibration', None) and
                lane_masks and image_match is None):
            pose = self._corner_pose_for_frame(now.nanoseconds/1e9)
            match = self.lane_history.match(
                lane_masks, self.robot_calibration, self.corner_capture_stamp, pose,
                self._float_parameter('metric_path_min_m'),
                self._float_parameter('metric_path_max_m'),
                self._float_parameter('projection_max_forward_m'),
                self._int_parameter('path_polynomial_degree'))
            if match:
                self.history_match = match
                self.image_match = match
                self.image_side_hint = match[1] if len(lane_masks)==1 else None
                image_match = match
        boundary_count = self._update_lane_command(lane_masks, width, now)
        self.boundary_count = boundary_count
        self.debug_boundary_labels = {}
        if self.enabled and getattr(self, 'robot_calibration', None):
            if boundary_count:
                self.lane_history.remember(self.corner_capture_stamp,
                    self._corner_pose_for_frame(now.nanoseconds/1e9),
                    getattr(self.metric_tracker, 'last_observation', None))
            roles = self._image_roles_from_target(lane_masks, self.metric_target)
            self.debug_boundary_labels = {i: (side, 'USED') for i, side in roles.items()}
            if not roles and image_match:
                roles = {image_match[0]: image_match[1]}
                self.debug_boundary_labels = {i: (side, 'HINT') for i, side in roles.items()}
            obs = getattr(self.metric_tracker, 'last_observation', None)
            if not self.debug_boundary_labels and len(lane_masks) == 1 and obs:
                self.debug_boundary_labels = {0: (obs['side'], 'OBSERVED; NO TARGET')}
            self.image_identity.commit(now.nanoseconds/1e9, roles)
            for index in getattr(self.metric_tracker, 'context_only_indices', []):
                self.debug_boundary_labels[index] = ('context', 'FAR; NOT STEERING')
            self._update_turn_observation(lane_masks, now)
        # Independent display history. It never supplies planner observations,
        # enables motors or alters the tracking/recovery state.
        self.local_map_pose = self._corner_pose_for_frame(now.nanoseconds/1e9)
        self.local_map_history.remember(self.corner_capture_stamp, self.local_map_pose,
                                       getattr(getattr(self, 'metric_tracker', None), 'last_observation', None))
        self.local_map_archive.remember(self.corner_capture_stamp, self.local_map_pose,
                                       getattr(getattr(self, 'metric_tracker', None), 'last_observation', None))
        self.last_geometry_seconds = time.monotonic()-geometry_started
        self.last_debug_seconds = 0.
        if self._bool_parameter('publish_debug_image'):
            debug_started = time.monotonic()
            self._queue_debug(
                frame, polygons_by_class, boundary_count, source_header)
            self.last_debug_seconds = time.monotonic()-debug_started

    def _image_roles_from_target(self, masks, target):
        """Label current masks only from confirmed metric geometry.

        The side history can then survive a frame whose centre path is invalid.
        Nearest mask assignment is unique and limited to 4 cm lateral error.
        """
        if not target:
            return {}
        if target.get('inferred'):
            side = target.get('visible_side')
            index = target.get('source_mask_index', 0 if len(masks) == 1 else -1)
            return {index: side} if side in ('left', 'right') and 0 <= index < len(masks) else {}
        if (target.get('boundary_count') != 2 or
                getattr(self.metric_tracker, 'streak', 0) < 2):
            return {}
        scores = []
        for index, mask in enumerate(masks):
            curves = floor_curves([mask], self.robot_calibration,
                                  self._float_parameter('projection_max_forward_m'),
                                  self._float_parameter('metric_path_min_m'),
                                  self._float_parameter('metric_path_max_m'),
                                  self._int_parameter('path_polynomial_degree'))
            if len(curves) != 1:
                continue
            curve = curves[0]
            for side in ('left', 'right'):
                reference = np.asarray(target.get(side+'_curve', []), float)
                if reference.ndim != 2 or len(reference) < 2:
                    continue
                error = curve_match_error(curve, reference)
                scores.append((float(error), index, side))
        roles = {}
        for error, index, side in sorted(scores):
            if error <= .04 and index not in roles and side not in roles.values():
                roles[index] = side
        return roles

    def _near_curve_visible(self, curve):
        """Require at least 4 cm of observed support inside the near floor band."""
        points = np.asarray(curve, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            return False
        near = supported_chain(points, self._float_parameter('metric_path_min_m'),
                               self._float_parameter('near_pair_max_m'))
        return len(near) >= 2 and arc_stations(near)[-1] >= .04

    def _near_pair_visible(self, target):
        """A measured pair must overlap near the robot, not merely in the distance."""
        if not target or target.get('boundary_count') != 2 or target.get('inferred'):
            return False
        left = np.asarray(target.get('left_curve', []), float)
        right = np.asarray(target.get('right_curve', []), float)
        if not self._near_curve_visible(left) or not self._near_curve_visible(right):
            return False
        lo = max(np.min(left[:, 0]), np.min(right[:, 0]), self._float_parameter('metric_path_min_m'))
        hi = min(np.max(left[:, 0]), np.max(right[:, 0]), self._float_parameter('near_pair_max_m'))
        return hi-lo >= .04

    def _update_turn_observation(self, lane_masks, now):
        """Use each fresh inference once to track the visible side and near pair.

        The turn timer is never renewed by inference frames. Two consecutive
        measured pairs in the near band release the turn; far pairs do not.
        """
        target = self.metric_target
        if self._bool_parameter('corner_enabled') and self.corner_policy.block_recovery:
            # A confirmed/pending corner must not fall through to blind search spin.
            self.turn_started_at = None
            self.turn_seen_at = None
            self.single_side_since = None
            return
        # A recovered, freshly validated single-side centre path can resume the
        # ordinary slow follower after three consistent observations. A mere
        # visible line (or held target) cannot do this. Keep the turn budget
        # exhausted so repeated single-side recoveries cannot cause endless spins.
        resume = (self.turn_started_at is not None and
                  self._bool_parameter('single_line_turn_requires_path_loss') and
                  target and target.get('inferred') and not target.get('held') and
                  target.get('visible_side') == self.turn_side and
                  self._near_curve_visible(target.get('actual_curve', [])))
        if resume:
            prior = self.single_resume_observation
            curve = np.asarray(target['actual_curve'], float)
            count = 1
            if (prior is not None and prior[1] == self.turn_side and
                    0 < (now-prior[0]).nanoseconds/1e9 <=
                    self._float_parameter('single_line_visibility_timeout') and
                    self.metric_tracker._same_curve(curve, prior[2])):
                count = prior[3]+1
            self.single_resume_observation = (now, self.turn_side, curve.copy(), count)
            if count >= 3:
                self.turn_started_at = None
                self.turn_side = None
                self.turn_seen_at = None
                self.turn_exhausted = True
                self.single_resume_observation = None
        else:
            self.single_resume_observation = None
        if self._near_pair_visible(target):
            self.near_pair_streak += 1
            self.single_side_since = None
            self.single_side_last_seen = None
            self.single_side_candidate = None
            if self.near_pair_streak >= 2:
                self.turn_started_at = None
                self.turn_side = None
                self.turn_seen_at = None
                self.turn_exhausted = False
            return
        self.near_pair_streak = 0
        # A valid one-sided centre path is already enough to drive. Recovery
        # must not interrupt it just because the second boundary is offscreen.
        if (target and not target.get('held') and self.turn_started_at is None and
                self._bool_parameter('single_line_turn_requires_path_loss')):
            self.single_side_since = None
            self.single_side_last_seen = None
            self.single_side_candidate = None
            return
        if len(lane_masks) != 1:
            self.single_side_since = None
            self.single_side_last_seen = None
            self.single_side_candidate = None
            return
        side = None
        if target and target.get('inferred') and self._near_curve_visible(target.get('actual_curve', [])):
            side = target.get('visible_side')
        else:
            tracker = self.metric_tracker
            old = getattr(tracker, 'observed_curve', None)
            old_side = getattr(tracker, 'observed_side', None)
            curves = floor_curves(lane_masks, self.robot_calibration,
                                  self._float_parameter('projection_max_forward_m'),
                                  self._float_parameter('metric_path_min_m'),
                                  self._float_parameter('metric_path_max_m'),
                                  self._int_parameter('path_polynomial_degree'))
            if (old_side in ('left', 'right') and old is not None and len(curves) == 1 and
                    self._near_curve_visible(curves[0]) and
                    (tracker._same_curve(curves[0], old) or
                     getattr(self, 'image_side_hint', None) == old_side)):
                side = old_side
        if side not in ('left', 'right') or (self.turn_side and side != self.turn_side):
            self.single_side_since = None
            self.single_side_last_seen = None
            self.single_side_candidate = None
            if self.turn_started_at is not None:
                self.turn_seen_at = None
            return
        gap = ((now-self.single_side_last_seen).nanoseconds/1e9
               if self.single_side_last_seen is not None else float('inf'))
        if (self.single_side_candidate != side or gap < 0 or
                gap > self._float_parameter('single_line_visibility_timeout')):
            self.single_side_since = now
            self.single_side_candidate = side
        self.single_side_last_seen = now
        if (self.turn_started_at is None and not self.turn_exhausted and
                (now-self.single_side_since).nanoseconds/1e9 >=
                self._float_parameter('single_line_turn_delay_seconds')):
            self.turn_started_at = now
            self.turn_side = side
        if self.turn_started_at is not None:
            self.turn_seen_at = now

    def _metric_command(self, target):
        """Lightweight local pursuit and speed limits; no geometry fitting."""
        if target.get('white_path') and target['x_m'] < .04:
            limit = self._float_parameter('maximum_angular_speed')
            return 0., float(np.clip(np.arctan2(target['y_m'], target['x_m']), -limit, limit))
        speed = min(self._float_parameter('maximum_linear_speed'), max(0., self._metric_speed()))
        distance = np.hypot(target['x_m'], target['y_m'])
        speed *= max(.5, min(1., distance/self._metric_lookahead()))
        speed *= max(.3, 1.-.75*abs(target['y_m'])/max(distance, .01))
        corner_active = 'corner_speed_cap' in target
        angular_limit = self._float_parameter(
            'corner_max_angular_speed' if corner_active else 'maximum_angular_speed')
        if target.get('inferred'):
            cap = target['corner_speed_cap'] if corner_active else self._float_parameter('single_line_max_speed')
            speed = min(cap, speed*self._float_parameter('single_line_speed_scale'))
        if corner_active:
            speed = min(speed, target['corner_speed_cap'])
        k = abs(2*target['y_m']/max(target['x_m']**2+target['y_m']**2, 1e-8))
        speed = min(speed, angular_limit/max(k, 1e-6))
        if target.get('corner_stationary'):
            speed, angular = 0., 0.
        else:
            angular = pursuit(target, speed, angular_limit)
        return speed, angular

    def _apply_metric_target(self, target, now):
        speed, angular = self._metric_command(target)
        self.metric_target = target
        self.metric_error = target.get('center_path_warning') or target.get('virtual_boundary_warning')
        if self.metric_error:
            self.get_logger().warn('Lane geometry: '+self.metric_error, throttle_duration_sec=2.)
        self.latest_linear, self.latest_angular, self.filtered_angular = speed, angular, angular
        self.last_lane_time, self.metric_missing_since = now, None
        self.metric_last_good_target = target.copy()
        self.metric_last_good_linear, self.metric_last_good_angular = speed, angular
        self.near_center = self.far_center = None
        return target['boundary_count']

    def _process_planned_result(self, plan, now):
        """Accept a fresh PC plan; never run projection, fitting or history here."""
        from .lane_planning_remote import restore_stage, TIMES, VALUES
        from rclpy.time import Time
        for key in TIMES:
            value = plan['times'][key]
            if value is not None and (type(value) is not int or value > self.remote_plan_context['now_ns']):
                raise ValueError('PC attempted to renew robot timestamp')
        incoming = restore_stage(plan['staged'])
        current = self.corner_policy.staged
        if current is not None and incoming is not None:
            if not np.allclose(current['pivot'], incoming['pivot'], atol=1e-9):
                raise ValueError('PC attempted to move committed corner pivot')
            if any(incoming[k] != current[k] for k in ('started', 'entry_yaw', 'exit_yaw', 'side')):
                raise ValueError('PC attempted to change committed corner identity')
            if current['brake_at'] is not None:
                incoming['brake_at'] = current['brake_at']
            if current.get('heading_reached'):
                incoming['heading_reached'] = True
            if current.get('fault'):
                incoming['fault'] = current['fault']
            if incoming.get('blind_entry') is not None and current.get('blind_entry') is not None:
                incoming['blind_entry'] = current['blind_entry']
        if current is not None and incoming is None:
            if (self.corner_odom is None or
                    not 0 <= now.nanoseconds/1e9-self.corner_odom[0] <= self._float_parameter('corner_odom_timeout') or
                    abs(wrap(current['exit_yaw']-self.corner_odom[1][2])) > np.deg2rad(12)):
                raise ValueError('PC exit before local heading arrival')
        target = deepcopy(plan['target'])
        if incoming is not None and incoming.get('fault'):
            target = None
        if target is not None and target.get('corner_staged'):
            cfg = self.corner_policy.config
            target['corner_speed_cap'] = min(target['corner_speed_cap'], cfg.entry_speed_mps)
            target['corner_entry_deadline'] = incoming['started']+20.
            if target['boundary_count'] == 0:
                target['corner_speed_cap'] = min(target['corner_speed_cap'], .015)
                target['corner_angular_cap'] = .15
                if not target.get('corner_stationary'):
                    target['corner_blind_deadline'] = incoming['matched_at']+2.
            if target.get('corner_stationary'):
                if incoming['brake_at'] is None:
                    raise ValueError('stationary target without arrival')
                target['corner_spin_yaw'] = incoming['exit_yaw']
                target['corner_brake_until'] = incoming['brake_at']+cfg.brake_seconds
                target['corner_spin_deadline'] = target['corner_brake_until']+cfg.spin_timeout_s
                target['corner_heading_reached'] = bool(incoming.get('heading_reached'))
            if target.get('corner_blind_entry') is not None:
                target['corner_blind_entry'] = incoming['blind_entry']
        if target is not None and not target.get('corner_staged'):
            pose = self.remote_plan_context['pose']
            if pose is None:
                raise ValueError('camera-time odometry unavailable for PC target')
            if not target.get('held'):
                c, s = np.cos(pose[2]), np.sin(pose[2])
                target['remote_world_target'] = (np.array([[c, -s], [s, c]])@
                    np.array([target['x_m'], target['y_m']])+pose[:2]).tolist()
            if np.asarray(target.get('remote_world_target')).shape != (2,):
                raise ValueError('held PC target lacks fixed odom position')
        self.corner_policy.staged = incoming
        self.corner_policy.debug = plan['debug']
        self.corner_policy.block_recovery = plan['block_recovery']
        self.metric_tracker.last_observation = plan.get('observation')
        self.debug_boundary_labels = {int(k): tuple(v) for k, v in plan.get('labels', {}).items()}
        if target is None:
            self._invalidate_lane()
            self.metric_error = plan.get('error')
        else:
            self.boundary_count = self._apply_metric_target(target, now)
        for key in TIMES:
            value = plan['times'][key]
            setattr(self, key, None if value is None else Time(nanoseconds=value, clock_type=ClockType.STEADY_TIME))
        for key in VALUES:
            setattr(self, key, plan['values'][key])

    def _update_lane_command(self, lane_masks, image_width, now):
        if (self._bool_parameter('simulation_white_lane') or
                self._bool_parameter('simulation_semantic_path')):
            try:
                if self.sim_white_path is None:
                    raise ValueError('white centre path unavailable')
                target = white_target(self.sim_white_path, self._metric_lookahead(),
                                      len(lane_masks), self.sim_white_side)
                target['width_m'] = self._float_parameter('lane_width')
                return self._apply_metric_target(target, now)
            except ValueError as exc:
                self.metric_error = str(exc)
                self.metric_target = None
                self._invalidate_lane()
                return 0
        if getattr(self, 'robot_calibration', None):
            try:
                stage = getattr(getattr(self, 'corner_policy', None), 'staged', None)
                if stage is not None and stage.get('heading_reached') and not stage.get('exit_tracking_reset'):
                    # Reacquire in the NEW heading, not against the pre-turn
                    # left/right lock. Preserve measured width, reset once.
                    estimator = getattr(self.metric_tracker, 'width_estimator', None)
                    self.metric_tracker = MetricLaneTracker(
                        connected_geometry=self._bool_parameter('connected_geometry'),
                        tracking_gap_s=self._float_parameter('lane_tracking_gap_seconds'),
                        minimum_lane_width_m=2*(self._float_parameter('corner_robot_half_width_m')+
                                               self._float_parameter('corner_clearance_m')))
                    if estimator is not None:
                        self.metric_tracker.width_estimator = estimator
                    self.image_identity = LaneImageIdentity(self._float_parameter('lane_tracking_gap_seconds'))
                    self.lane_history.clear()
                    self.image_side_hint = self.image_match = self.history_match = None
                    self.metric_last_good_target = None
                    stage['exit_tracking_reset'] = True
                if not hasattr(self, 'metric_tracker'):
                    self.metric_tracker = MetricLaneTracker(
                        connected_geometry=self._bool_parameter('connected_geometry'),
                        tracking_gap_s=self._float_parameter('lane_tracking_gap_seconds'),
                        minimum_lane_width_m=2*(self._float_parameter('corner_robot_half_width_m')+
                                                self._float_parameter('corner_clearance_m')))
                ordinary_error = None
                map_side = getattr(self, 'hybrid_white_map_side', None) if self._bool_parameter('simulation_yolo_white') else None
                if (self._bool_parameter('simulation_yolo_white') and len(lane_masks) == 1
                        and self.robot_calibration.get('method') == 'intrinsics_urdf_floor'
                        and self.corner_policy.local_lane_map.frames):
                    from .metric_lane import floor_curves
                    pose = self._geometry_frame_pose(now)
                    curves = floor_curves(lane_masks, self.robot_calibration,
                        self._float_parameter('projection_max_forward_m'),
                        self._float_parameter('metric_path_min_m'),
                        self._float_parameter('metric_path_max_m'),
                        self._int_parameter('path_polynomial_degree'),
                        connected_geometry=self._bool_parameter('connected_geometry'))
                    if len(curves) == 1:
                        map_side = map_side or self.corner_policy.local_lane_map.measured_side(
                            curves[0], now.nanoseconds/1e9, pose)
                try:
                    target = self.metric_tracker.update(lane_masks, self.robot_calibration,
                                       now.nanoseconds/1e9,
                                       self._metric_lookahead(),
                                       self._float_parameter('metric_single_line_timeout'),
                                       lane_width=self._float_parameter('lane_width'),
                                       degree=self._int_parameter('path_polynomial_degree'),
                                       max_forward_m=self._float_parameter('projection_max_forward_m'),
                                       path_min_m=self._float_parameter('metric_path_min_m'),
                                       path_max_m=self._float_parameter('metric_path_max_m'),
                                       image_side_hint=map_side or getattr(self, 'image_side_hint', None),
                                       image_match=getattr(self, 'image_match', None),
                                       recovery_side_hint=(map_side or near_pixel_side(lane_masks[0])
                                                           if len(lane_masks) == 1 else None))
                except ValueError as exc:
                    target = None
                    ordinary_error = exc
                active_mask_count = len(lane_masks)-len(
                    getattr(self.metric_tracker, 'context_only_indices', []))
                if self._bool_parameter('corner_enabled'):
                    observation = getattr(self.metric_tracker, 'last_observation', None)
                    # A confirmed staged turn uses its fixed odom pivot, not
                    # the ordinary inferred centre target. That target's jump
                    # must not discard the freshly measured boundary. The
                    # staged policy still checks side, fragment match, odom,
                    # drift and deadlines. Other geometry/identity failures
                    # and unconfirmed candidates keep the original rejection.
                    staged_target_jump = (
                        getattr(self.corner_policy, 'staged', None) is not None
                        and str(ordinary_error) in ('inferred target discontinuity',
                                                   'no forward centre path'))
                    s_route_target_jump = (
                        getattr(self.corner_policy, 's_route', None) is not None
                        and str(ordinary_error) == 'inferred target discontinuity')
                    if ordinary_error and (active_mask_count != 1 or
                            ('center_path curve folds back' not in str(ordinary_error)
                             and not (self._bool_parameter('simulation_yolo_white') and
                                      str(ordinary_error) == 'no forward centre path')
                             and not staged_target_jump and not s_route_target_jump)):
                        observation = None
                    pose = (self._geometry_frame_pose(now) if
                            self._bool_parameter('simulation_yolo_white') else
                            self._corner_pose_for_frame(now.nanoseconds/1e9))
                    if (pose is not None and
                            self._bool_parameter('simulation_yolo_white') and
                            self.corner_policy.staged is not None):
                        recovered = self.corner_policy.staged_observation(
                            lane_masks, self.robot_calibration, pose)
                        if recovered is not None:
                            observation = recovered
                    if (self._bool_parameter('simulation_yolo_white') and
                            getattr(self.corner_policy, 'staged', None) is None):
                        observation, target = self._hybrid_connected_observation(
                            lane_masks, observation, target)
                        if (self.corner_policy.s_route is not None and observation is not None and
                                target is not None and not target.get('held')):
                            target = dict(target, preserve_pending_s=True)
                    self.corner_policy.floor_calibration = self.robot_calibration
                    target = self.corner_policy.update(observation, target, now.nanoseconds/1e9,
                              pose, self._metric_lookahead(), self._float_parameter('corner_max_angular_speed'),
                              no_boundaries=len(lane_masks) == 0)
                    self.corner_policy.debug['pose_timing'] = getattr(self, 'corner_pose_diagnostic', {})
                    if self._bool_parameter('simulation_yolo_white'):
                        self.corner_policy.debug['connected_trace']=getattr(self,'hybrid_connected_trace',{})
                    if map_side:
                        self.corner_policy.debug['measured_map_side_hint'] = map_side
                    if ordinary_error:
                        self.corner_policy.debug['ordinary_error'] = str(ordinary_error)
                    if staged_target_jump:
                        self.corner_policy.debug['ordinary_target_ignored'] = True
                    if s_route_target_jump:
                        self.corner_policy.debug['s_route_revalidated_target_jump'] = True
                if target is None:
                    if self._bool_parameter('corner_enabled') and self.corner_policy.block_recovery:
                        info = self.corner_policy.debug
                        raise ValueError(f"corner {info.get('mode')}: {info.get('reason')}")
                    if ordinary_error:
                        raise ordinary_error
                    raise ValueError('corner: '+self.corner_policy.debug.get('reason', 'awaiting confirmation'))
                return self._apply_metric_target(target, now)
            except ValueError as exc:
                self.metric_error = str(exc)
                # Only genuinely EMPTY detections qualify for short dead reckoning.
                # Ambiguous/nonphysical geometry and inference failures stop directly.
                saved = getattr(self, 'metric_last_good_target', None)
                if (not lane_masks and saved is not None and self.last_lane_time is not None
                        and not (self._bool_parameter('corner_enabled') and self.corner_policy.block_recovery)
                        and 0 <= (now-self.last_lane_time).nanoseconds/1e9 <
                        self._float_parameter('lane_loss_stop_seconds')):
                    self.metric_missing_since = self.last_lane_time
                    self.metric_target = dict(saved, held=True, boundary_count=0)
                    return 0
                self.metric_target = None
                self._invalidate_lane()
                self.get_logger().warn('Metric lane unavailable: '+str(exc), throttle_duration_sec=2.)
                return 0
        band = self._float_parameter('sample_band_ratio')
        near_ratio = self._float_parameter('near_y_ratio')
        minimum_near_ratio = self._float_parameter(
            'adaptive_near_min_y_ratio')
        near_step = max(0.01, self._float_parameter('adaptive_near_step'))
        far_candidates = lane_candidates_at(
            lane_masks, self._float_parameter('far_y_ratio'), band)

        image_center = image_width / 2.0
        expected_near = (
            self.near_center if self.near_center is not None else image_center)
        expected_width = self.near_lane_width
        if expected_width is None:
            expected_width = (
                image_width
                * self._float_parameter('initial_lane_width_ratio'))
        minimum_ratio = self._float_parameter('minimum_lane_width_ratio')
        maximum_ratio = self._float_parameter('maximum_lane_width_ratio')
        near, selected_ratio, candidates = select_near_lane(
            lane_masks, image_width, expected_near, expected_width,
            near_ratio, minimum_near_ratio, near_step, band,
            minimum_ratio, maximum_ratio)
        self.active_near_y_ratio = selected_ratio
        self.near_candidate_count = candidates
        if near is None:
            self._invalidate_lane()
            return 0

        if near.boundary_count >= 2:
            alpha = self._float_parameter('lane_width_alpha')
            self.near_lane_width = (
                (1.0 - alpha) * expected_width + alpha * near.lane_width)
        else:
            self.near_lane_width = expected_width
        self.near_center = near.center_x

        far_scale = self._float_parameter('far_lane_width_scale')
        far_width = self.near_lane_width * far_scale
        expected_far = (
            self.far_center if self.far_center is not None else near.center_x)
        far = estimate_lane_center(
            far_candidates, image_width, expected_far, far_width,
            minimum_ratio * far_scale, maximum_ratio)
        self.far_center = far.center_x if far is not None else near.center_x
        if far is not None and far.boundary_count == 1:
            consistent = consistent_single_far_center(
                lane_masks, near.center_x, selected_ratio,
                self._float_parameter('far_y_ratio'), far_width, band)
            self.far_center = (near.center_x if consistent is None else
                               min(image_width - 1.0, max(0.0, consistent)))

        target_angular = steering_command(
            self.near_center, self.far_center, image_width,
            self._float_parameter('lateral_gain'),
            self._float_parameter('heading_gain'),
            self._float_parameter('maximum_angular_speed'))
        preview_severity = 0.0
        if self._bool_parameter('near_priority_steering'):
            target_angular, preview_severity = near_priority_command(
                self.near_center, self.far_center, image_width,
                self._float_parameter('lateral_gain'),
                self._float_parameter('maximum_angular_speed'),
                self._float_parameter('near_center_deadband'))
        alpha = self._float_parameter('steering_alpha')
        self.filtered_angular = (
            (1.0 - alpha) * self.filtered_angular + alpha * target_angular)
        self.latest_angular = self.filtered_angular

        max_angular = self._float_parameter('maximum_angular_speed')
        turn_ratio = min(
            1.0, abs(self.latest_angular) / max(0.01, max_angular))
        turn_ratio = max(turn_ratio, min(1.0, max(0.0,
            self._float_parameter('curve_slowdown_gain')) * preview_severity))
        maximum_speed = self._float_parameter('linear_speed')
        minimum_speed = self._float_parameter('minimum_linear_speed')
        speed = maximum_speed - turn_ratio * (maximum_speed - minimum_speed)
        if near.boundary_count == 1:
            speed *= self._float_parameter('single_line_speed_scale')
        self.latest_linear = max(0.0, speed)
        self.last_lane_time = now
        return near.boundary_count

    def _update_crossline(self, masks, now):
        close = any(
            crossline_is_close(
                mask,
                self._float_parameter('crossline_trigger_y_ratio'),
                self._float_parameter('crossline_minimum_area_ratio'))
            for mask in masks
        )
        if close:
            self.crossline_clear_since = None
            if not self.crossline_latched:
                self.crossline_streak += 1
                if self.crossline_streak >= self._int_parameter(
                        'crossline_confirm_frames'):
                    self.crossline_latched = True
                    seconds = self._float_parameter('crossline_stop_seconds')
                    self.crossline_stop_until = now + Duration(seconds=seconds)
                    self.get_logger().info(
                        f'Crossline reached: stopping for {seconds:.1f} seconds')
        else:
            self.crossline_streak = 0
            if self.crossline_latched and not self._crossline_stop_active(now):
                if self.crossline_clear_since is None:
                    self.crossline_clear_since = now
                clear_age = (
                    now - self.crossline_clear_since).nanoseconds / 1e9
                if clear_age >= self._float_parameter(
                        'crossline_release_seconds'):
                    self.crossline_latched = False
                    self.crossline_stop_until = None
                    self.crossline_clear_since = None

    def _crossline_stop_active(self, now):
        return (
            self.crossline_stop_until is not None
            and now < self.crossline_stop_until
        )

    def _control_loop(self):
        now = self.safety_clock.now()
        deadline = getattr(self, 'enable_deadline_ns', None)
        if self.enabled and deadline is not None and now.nanoseconds >= deadline:
            self._enable_callback(SetBool.Request(data=False), SetBool.Response())
            self.get_logger().warn('Configured duration expired: lane autonomy disabled.')
        # Record arrival independently of image availability or metric_target.
        # Do this BEFORE consuming an in-flight empty PC result, so its older
        # entry snapshot cannot erase a locally observed arrival. This does NOT
        # publish motion: all safety checks below still run before commands.
        odom = getattr(self, 'corner_odom', None)
        if (self.enabled and odom is not None and
                0 <= now.nanoseconds/1e9-odom[0] <= self._float_parameter('corner_odom_timeout')):
            self.corner_policy.observe_arrival(odom[1], now.nanoseconds/1e9)
        self._consume_inference(now)
        now = self.safety_clock.now()
        self._start_inference(now)
        if not self.enabled:
            self.state = DriveState.DISABLED
            self._publish_zero()
            self._publish_status('DISABLED')
            return
        if self._crossline_stop_active(now):
            remaining = (self.crossline_stop_until - now).nanoseconds / 1e9
            self.state = DriveState.CROSSLINE_STOP
            self._publish_zero()
            self._publish_status(f'CROSSLINE_STOP: {remaining:.1f}s remaining')
            return

        safety_reason = self._safety_stop_reason(now)
        if safety_reason:
            self.crossline_streak = 0
            self.crossline_clear_since = None
            self.state = DriveState.SAFETY_STOP
            self._publish_zero()
            self._publish_status(f'SAFETY_STOP: {safety_reason}')
            return
        if self.metric_target and self.metric_target.get('corner_staged'):
            self.corner_policy.observe_arrival(
                self.corner_odom[1] if self.corner_odom else None, now.nanoseconds/1e9)
            stage = self.corner_policy.staged
            if (stage is not None and self.metric_target.get('corner_stationary') and
                    self.corner_odom is not None and abs(wrap(stage['exit_yaw']-self.corner_odom[1][2])) <= np.deg2rad(12)):
                stage['heading_reached'] = True
            if stage is not None and stage.get('heading_reached'):
                self.metric_target['corner_heading_reached'] = True
            v, w, detail = staged_command(self.metric_target,
                self.corner_odom[1] if self.corner_odom else None,
                now.nanoseconds/1e9, self._float_parameter('corner_max_angular_speed'),
                self._float_parameter('corner_pivot_tolerance_m'))
            command = Twist()
            command.linear.x, command.angular.z = v, w
            self.cmd_publisher.publish(command)
            self.state = DriveState.FOLLOWING if v or w else DriveState.WAITING_FOR_LANE
            self._publish_status(f'CORNER_STAGED: {detail}; v={v:.3f}, w={w:.3f}')
            return
        if self.metric_target and 'remote_world_target' in self.metric_target:
            pose = self.corner_odom[1]  # Freshness enforced by _safety_stop_reason.
            c, s = np.cos(pose[2]), np.sin(pose[2])
            point = np.array([[c, s], [-s, c]])@(np.asarray(self.metric_target['remote_world_target'])-pose[:2])
            if point[0] <= .01:
                self._publish_zero()
                self._publish_status('WAITING_FOR_LANE: PC target reached or behind robot')
                return
            corrected = dict(self.metric_target, x_m=float(point[0]), y_m=float(point[1]))
            self.latest_linear, self.latest_angular = self._metric_command(corrected)
            self.filtered_angular = self.latest_angular
        if self.turn_started_at is not None:
            elapsed = (now-self.turn_started_at).nanoseconds/1e9
            if elapsed >= self._float_parameter('single_line_turn_seconds'):
                self.turn_exhausted = True
            seen_age = ((now-self.turn_seen_at).nanoseconds/1e9
                        if self.turn_seen_at is not None else float('inf'))
            if (not self.turn_exhausted and 0 <= seen_age <=
                    self._float_parameter('single_line_visibility_timeout')):
                command = Twist()
                # Left boundary means the lane interior lies to the right.
                command.angular.z = (-1. if self.turn_side == 'left' else 1.) * \
                    self._float_parameter('single_line_turn_speed')
                self.state = DriveState.REACQUIRING_LANE
                self.cmd_publisher.publish(command)
                self._publish_status(
                    f'REACQUIRING_LANE: side={self.turn_side}, t={elapsed:.1f}s, '
                    f'near_pair={self.near_pair_streak}/2, w={command.angular.z:.3f}')
                return
            self.state = DriveState.WAITING_FOR_LANE
            self._publish_zero()
            reason = ('turn limit reached' if self.turn_exhausted else 'near boundary not visible')
            self._publish_status(f'WAITING_FOR_LANE: {reason}; '+self._detection_summary())
            return
        if getattr(self, 'metric_missing_since', None) is not None:
            # Run this clock-based ramp at control_frequency even if no new
            # inference arrives. Never renew it from subsequent empty masks.
            age = (now-self.metric_missing_since).nanoseconds/1e9
            scale = loss_speed_scale(age, self._float_parameter('lane_loss_hold_seconds'),
                                     self._float_parameter('lane_loss_stop_seconds'))
            if scale <= 0:
                self._invalidate_lane()
                self.state = DriveState.WAITING_FOR_LANE
                self._publish_zero()
                self._publish_status('WAITING_FOR_LANE: centre-path hold expired')
                return
            speed = min(self.metric_last_good_linear, self._float_parameter('lane_loss_max_speed'))*scale
            command = Twist()
            command.linear.x = float(speed)
            command.angular.z = pursuit(self.metric_target, speed, self._float_parameter('maximum_angular_speed'))
            self.cmd_publisher.publish(command)
            self._publish_status(f'LANE_LOSS_HOLD: age={age:.2f}s, v={speed:.3f}; stale target, not a new detection')
            return
        if self.last_lane_time is None:
            self.state = DriveState.WAITING_FOR_LANE
            self._publish_zero()
            self._publish_status(
                'WAITING_FOR_LANE: no usable boundary; '
                + self._detection_summary())
            return
        lane_age = (now - self.last_lane_time).nanoseconds / 1e9
        if lane_age > self._float_parameter('lane_lost_timeout'):
            self.state = DriveState.WAITING_FOR_LANE
            self._publish_zero()
            self._publish_status(
                f'WAITING_FOR_LANE: result expired ({lane_age:.2f}s); '
                + self._detection_summary())
            return

        self.state = DriveState.FOLLOWING
        command = Twist()
        command.linear.x = float(self.latest_linear)
        command.angular.z = float(self.latest_angular)
        self.cmd_publisher.publish(command)
        self._publish_status(
            f'FOLLOWING: v={command.linear.x:.3f}, w={command.angular.z:.3f}; '
            + self._detection_summary())

    def _calibration_stop_reason(self):
        if self._bool_parameter('corner_enabled'):
            entry = self._float_parameter('corner_entry_speed_mps')
            tolerance = self._float_parameter('corner_pivot_tolerance_m')
            brake = self._float_parameter('corner_brake_seconds')
            spin_timeout = self._float_parameter('corner_spin_timeout_s')
            if (not np.isfinite([entry, tolerance, brake, spin_timeout]).all() or
                    not .005 <= entry <= min(.03, self._float_parameter('maximum_linear_speed')) or
                    not .01 <= tolerance <= .04 or not .2 <= brake <= 2. or
                    not 1 <= spin_timeout <= 20.):
                return 'Invalid staged corner parameters'
            timeout = self._float_parameter('corner_odom_timeout')
            if not np.isfinite(timeout) or not .05 <= timeout <= 1.:
                return 'Invalid corner odometry timeout'
            corner_speed = self._float_parameter('corner_speed_mps')
            corner_angular = self._float_parameter('corner_max_angular_speed')
            if (not np.isfinite([corner_speed, corner_angular]).all() or
                    not 0 < corner_speed <= self._float_parameter('maximum_linear_speed') or
                    not 0 < corner_angular <= 3.):
                return 'Invalid corner speed limits'
        reason = getattr(self, 'calibration_block_reason', None)
        if not getattr(self, 'robot_calibration', None):
            return reason
        # Zero duration is continuous operation, enabled explicitly by service.
        # This does not declare the nominal optical mounting physically validated.
        duration = self._float_parameter('max_enabled_seconds')
        values = [self._metric_speed(), self._metric_lookahead(),
                  *[self._float_parameter(name) for name in
                    ('maximum_linear_speed', 'maximum_angular_speed', 'lane_width',
                     'single_line_max_speed', 'single_line_speed_scale', 'projection_max_forward_m',
                     'metric_path_min_m', 'metric_path_max_m',
                     'near_pair_max_m', 'single_line_turn_speed',
                     'single_line_turn_delay_seconds', 'single_line_turn_seconds',
                     'single_line_visibility_timeout',
                     'lane_loss_hold_seconds', 'lane_loss_stop_seconds', 'lane_loss_max_speed',
                     'linear_velocity', 'lookahead_distance')]]
        if not np.isfinite(values).all():
            return 'Metric control parameters must be finite'
        if (not np.isfinite(duration) or duration < 0 or
                not 0 <= self._metric_speed() <= self._float_parameter('maximum_linear_speed') <= .5 or
                not 0 < self._float_parameter('maximum_angular_speed') <= 3. or
                not .05 <= self._metric_lookahead() <= self._float_parameter('projection_max_forward_m') <= 5. or
                not .05 <= self._float_parameter('metric_path_min_m') < self._metric_lookahead() <=
                    self._float_parameter('metric_path_max_m') <= self._float_parameter('projection_max_forward_m') or
                not self._float_parameter('metric_path_min_m')+.04 <=
                    self._float_parameter('near_pair_max_m') <= self._float_parameter('metric_path_max_m') or
                not 0 < self._float_parameter('single_line_turn_speed') <=
                    self._float_parameter('maximum_angular_speed') or
                not 0 < self._float_parameter('single_line_turn_delay_seconds') <= 20. or
                not 0 < self._float_parameter('single_line_turn_seconds') <= 20. or
                not .1 <= self._float_parameter('single_line_visibility_timeout') <= 1. or
                not .04 <= self._float_parameter('lane_width') <= 1.5 or
                not 0 < self._float_parameter('single_line_max_speed') <= self._float_parameter('maximum_linear_speed') or
                not 0 < self._float_parameter('single_line_speed_scale') <= 1 or
                not 0 <= self._float_parameter('lane_loss_hold_seconds') < self._float_parameter('lane_loss_stop_seconds') <= 1.5 or
                not 0 <= self._float_parameter('lane_loss_max_speed') <= self._float_parameter('maximum_linear_speed') or
                self._int_parameter('path_polynomial_degree') not in (1, 2, 3) or
                self._float_parameter('linear_velocity') < -1 or self._float_parameter('lookahead_distance') < -1 or
                not np.isfinite(self._float_parameter('metric_single_line_timeout')) or
                self._float_parameter('metric_single_line_timeout') < 0 or
                not np.isfinite(self._float_parameter('lane_tracking_gap_seconds')) or
                not 0 < self._float_parameter('lane_tracking_gap_seconds') <= 2.):
            return 'Metric control parameters invalid'
        return None

    def _safety_stop_reason(self, now):
        reason = self._calibration_stop_reason()
        if reason:
            return reason
        if self.last_image_time is None:
            return 'no camera frame'
        image_age = (now - self.last_image_time).nanoseconds / 1e9
        if image_age > self._float_parameter('image_timeout'):
            return f'camera timeout ({image_age:.2f}s)'
        if self.inference_error is not None:
            return 'inference failed'
        if (self.metric_target and (self._bool_parameter('remote_geometry') or
                (self._bool_parameter('corner_enabled') and
                 self.metric_target.get('corner_speed_cap') is not None))):
            odom = self.corner_odom
            if odom is None or not 0 <= now.nanoseconds/1e9-odom[0] <= self._float_parameter('corner_odom_timeout'):
                return 'corner odometry timeout'
        if self.last_result_input_time is not None:
            input_age = (now - self.last_result_input_time).nanoseconds / 1e9
            if input_age > self._float_parameter('result_timeout'):
                return f'inference input expired ({input_age:.2f}s)'
        if self._bool_parameter('use_lidar_guard'):
            if self.last_scan_time is None:
                return 'no lidar scan'
            scan_age = (now - self.last_scan_time).nanoseconds / 1e9
            if scan_age > self._float_parameter('lidar_timeout'):
                return f'lidar timeout ({scan_age:.2f}s)'
            if not self.front_clear:
                return 'obstacle ahead'
        return None

    @staticmethod
    def _image_to_bgr(message):
        """Decode the existing JPEG camera stream or a raw ROS image."""
        if isinstance(message, CompressedImage):
            frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid compressed camera frame')
            return frame
        encoding = message.encoding.casefold()
        channels_by_encoding = {
            'bgr8': 3, 'rgb8': 3, '8uc3': 3,
            'bgra8': 4, 'rgba8': 4,
            'mono8': 1, '8uc1': 1,
            'yuyv': 2, 'yuv422_yuy2': 2,
        }
        if encoding not in channels_by_encoding:
            raise ValueError(f'Unsupported camera encoding: {message.encoding}')
        channels = channels_by_encoding[encoding]
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step)
        pixels = rows[:, :message.width * channels].reshape(
            message.height, message.width, channels)
        if encoding == 'rgb8':
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        elif encoding == 'rgba8':
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
        elif encoding == 'bgra8':
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
        elif encoding in ('mono8', '8uc1'):
            pixels = cv2.cvtColor(pixels[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif encoding in ('yuyv', 'yuv422_yuy2'):
            pixels = cv2.cvtColor(pixels, cv2.COLOR_YUV2BGR_YUY2)
        return np.ascontiguousarray(pixels)

    @staticmethod
    def _bgr_to_message(frame, header):
        """Build a bgr8 ROS image without cv_bridge."""
        message = Image()
        message.header = header
        message.height, message.width = frame.shape[:2]
        message.encoding = 'bgr8'
        message.is_bigendian = 0
        message.step = message.width * 3
        data = array('B')
        data.frombytes(np.ascontiguousarray(frame).tobytes())
        message.data = data
        return message

    def _queue_debug(self, frame, polygons, boundary_count, header):
        """진단 표시의 오류는 유효한 차선 추론/주행 상태를 변경하지 않는다."""
        try:
            self._queue_debug_snapshot(frame, polygons, boundary_count, header)
        except Exception as exc:
            self.get_logger().warn(f'Diagnostic image skipped: {exc}', throttle_duration_sec=2.)

    def _queue_debug_snapshot(self, frame, polygons, boundary_count, header):
        """Copy display state only; worker never reads changing controller state."""
        now = time.monotonic()
        debug_due = self.debug_publisher.get_subscription_count() > 0
        map_publisher = getattr(self, 'local_map_publisher', None)
        map_due = (map_publisher is not None and map_publisher.get_subscription_count() > 0
                   and now-getattr(self, 'last_map_enqueued', 0.) >= .5)
        if not debug_due and not map_due:
            return  # No diagnostic subscribers: no copies, rendering or DDS writes.
        frequency = max(.1, self._float_parameter('debug_image_frequency'))
        if now-getattr(self, 'last_debug_enqueued', 0.) < 1./frequency:
            return
        self.last_debug_enqueued = now
        transport = getattr(self, 'debug_transport', None)
        if transport is None:
            transport = self.debug_transport = IsolatedImageTransport(
                self.context.get_domain_id(), dict(debug=self.debug_publisher.topic_name,
                                                   map=map_publisher.topic_name))
        worker = getattr(self, 'debug_worker', None)
        if worker is None:
            worker = self.debug_worker = LatestDebugWorker(
                lambda job: LaneAutonomy._publish_debug(*job))
        fields = ('lane_class_id', 'lane_instances', 'crossline_instances',
                  'perception_source',
                  'floor_calibration', 'robot_calibration', 'metric_target', 'metric_error',
                  'active_near_y_ratio', 'near_center', 'far_center', 'near_candidate_count',
                  'inference_seconds', 'turn_exhausted', 'turn_side',
                  'near_pair_streak')
        fields += ('debug_boundary_labels',)
        snapshot = SimpleNamespace(**{name: deepcopy(getattr(self, name, None)) for name in fields})
        # rclpy.Time는 C 핸들을 보유해 deepcopy/pickle 할 수 없다.
        # 표시 작업에는 값으로 만든 독립 시간 객체만 전달한다.
        from rclpy.time import Time
        turn = self.turn_started_at
        snapshot.turn_started_at = (None if turn is None else
                                    Time(nanoseconds=turn.nanoseconds, clock_type=turn.clock_type))
        params = {name: self.get_parameter(name).value for name in
                  ('far_y_ratio', 'single_line_turn_seconds', 'corner_enabled')}
        snapshot._float_parameter = lambda name: float(params[name])
        snapshot._bool_parameter = lambda name: bool(params[name])
        snapshot.corner_policy = SimpleNamespace(debug=deepcopy(self.corner_policy.debug))
        captured_now = self.safety_clock.now()
        snapshot.safety_clock = SimpleNamespace(now=lambda: captured_now)
        snapshot._bgr_to_message = self._bgr_to_message
        snapshot.debug_publisher = (
            DiagnosticPublisher(self.debug_publisher, transport, 'debug') if debug_due else None)
        snapshot.local_map_publisher = (
            DiagnosticPublisher(map_publisher, transport, 'map') if map_due else None)
        if snapshot.local_map_publisher is not None:
            self.last_map_enqueued = now
        snapshot.local_map_frames = deepcopy(list(self.local_map_history.frames)) if map_due else []
        snapshot.local_map_archive = self.local_map_archive.snapshot() if map_due else None
        snapshot.local_map_pose = deepcopy(self.local_map_pose)
        snapshot.local_map_stamp = self.corner_capture_stamp
        worker.submit((snapshot, frame.copy(), [(c,p.copy()) for c,p in polygons],
                       boundary_count, deepcopy(header)))

    def _publish_debug(self, frame, polygons, boundary_count, header):
        overlay = frame.copy()
        for class_id, polygon in polygons:
            color = (
                (255, 180, 0) if class_id == self.lane_class_id
                else (0, 0, 255))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(frame, [polygon], True, color, 2)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0.0, frame)
        height, width = frame.shape[:2]
        calibration = getattr(self, 'floor_calibration', None)
        robot_calibration = getattr(self, 'robot_calibration', None)
        if robot_calibration:
            for class_id, polygon in polygons:
                point = max(polygon, key=lambda p: p[1])
                label = 'Lane' if class_id == self.lane_class_id else 'Crossline'
                try:
                    x_m, y_m = robot_floor_point(*map(float, point), robot_calibration, (width, height))
                    text = f'{label} base X~{x_m*100:.1f} Y~{y_m*100:.1f}cm'
                except ValueError:
                    text = f'{label} base distance N/A'
                cv2.putText(frame, text, (10, max(90, min(height-10, int(point[1])))),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.putText(frame, 'URDF projection: mounting UNVERIFIED; X forward Y left',
                        (10, 69), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 220, 255), 1)
            target = getattr(self, 'metric_target', None)
            if target:
                cv2.putText(frame, f"Target X={target['x_m']*100:.1f} Y={target['y_m']*100:.1f}cm",
                            (10, 89), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
        if calibration:
            for class_id, polygon in polygons:
                # Bottom-most detected vertex; never clamp/extrapolate a
                # near out-of-range line into the calibrated band.
                point = max(polygon, key=lambda p: p[1])
                label = 'Lane' if class_id == self.lane_class_id else 'Crossline'
                try:
                    distance = forward_cm(float(point[0]), float(point[1]),
                                          calibration, (width, height))
                    text = f'{label} cam-forward~{distance:.1f}cm'
                except ValueError:
                    text = f'{label} distance N/A (calibration range)'
                x = max(5, min(int(point[0]), width-330))
                y = max(85, min(int(point[1])-8, height-10))
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 255, 255), 1)
            cv2.putText(frame, 'APPROX camera-forward only; mount unverified',
                        (10, 69), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 220, 255), 1)
        samples = () if robot_calibration else (
            (self.active_near_y_ratio,
             self.near_center, (0, 255, 0)),
            (self._float_parameter('far_y_ratio'),
             self.far_center, (0, 255, 255)),
        )
        for ratio, center, color in samples:
            y_coordinate = int(height * ratio)
            cv2.line(
                frame, (0, y_coordinate), (width - 1, y_coordinate), color, 1)
            if center is not None:
                cv2.circle(frame, (int(center), y_coordinate), 6, color, -1)
        cv2.line(
            frame, (width // 2, 0), (width // 2, height - 1),
            (255, 255, 255), 1)
        cv2.putText(
            frame,
            f'{getattr(self, "perception_source", "YOLO")} Lane={self.lane_instances} Crossline={self.crossline_instances} '
            f'used_boundaries={boundary_count}',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        summary = (f'METRIC target={getattr(self, "metric_target", None)}'
                   if robot_calibration else
                   f'near_y={self.active_near_y_ratio:.2f} candidates={self.near_candidate_count}')
        if robot_calibration:
            t = getattr(self, 'metric_target', None)
            summary = (("HELD " if t.get('held') else "") + f"METRIC X={t['x_m']*100:.0f} Y={t['y_m']*100:.1f}cm" if t else 'METRIC no valid target')
        cv2.putText(
            frame, summary + f' processing={self.inference_seconds:.2f}s',
            (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if robot_calibration:
            draw_metric_target(frame, getattr(self, 'metric_target', None), robot_calibration)
            reason = getattr(self, 'metric_error', None)
            if not reason and t:
                reason = t.get('center_path_warning')
            if reason:
                cv2.putText(frame, reason[:90], (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (0, 0, 255), 1)
            if self.turn_started_at is not None:
                elapsed = (self.safety_clock.now()-self.turn_started_at).nanoseconds/1e9
                limit = self._float_parameter('single_line_turn_seconds')
                label = (f'{"TURN LIMIT" if self.turn_exhausted else "TURN"} {self.turn_side}: {min(elapsed, limit):.1f}/'
                         f'{self._float_parameter("single_line_turn_seconds"):.0f}s '
                         f'near pair {self.near_pair_streak}/2')
                cv2.putText(frame, label, (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 165, 255), 1)
            if self._bool_parameter('corner_enabled'):
                info = self.corner_policy.debug
                label = (f"{info.get('mode')} {info.get('direction','')} "
                         f"angle={info.get('angle_deg',0):.0f} "
                         f"distance={info.get('corner_distance_m',0)*100:.0f}cm "
                         f"confirm={info.get('confirmations',0)}")
                cv2.putText(frame, label, (10, 152), cv2.FONT_HERSHEY_SIMPLEX,
                            .4, (0, 200, 255), 1)
                if 'remaining_forward_m' in info:
                    progress = (f"ENTRY left={info['remaining_forward_m']*100:.1f}cm "
                                f"side={info['lateral_error_m']*100:.1f}cm "
                                f"pivot(odom)={info.get('pivot_world')}")
                    cv2.putText(frame, progress[:110], (10, 170), cv2.FONT_HERSHEY_SIMPLEX,
                                .38, (0, 220, 255), 1)
        draw_boundary_labels(frame, polygons, self.lane_class_id,
                             getattr(self, 'debug_boundary_labels', None) or {},
                             self.corner_policy.debug)
        debug_message = self._bgr_to_message(frame, header)
        if self.debug_publisher is not None:
            self.debug_publisher.publish(debug_message)
        if getattr(self, 'local_map_publisher', None) is not None:
            map_image = render_local_map(self.local_map_frames, self.local_map_stamp,
                                         self.local_map_pose, getattr(self, 'metric_target', None),
                                         self.corner_policy.debug, self.local_map_archive)
            map_header = deepcopy(header)
            map_header.frame_id = 'base_link'
            self.local_map_publisher.publish(self._bgr_to_message(map_image, map_header))

    def _detection_summary(self):
        if getattr(self, 'robot_calibration', None):
            target = getattr(self, 'metric_target', None)
            brief = None if target is None else {k: target[k] for k in
                ('x_m', 'y_m', 'width_m', 'boundary_count', 'inferred')}
            return (f'Lane={self.lane_instances}, used_boundaries={self.boundary_count}, '
                    f'perception={getattr(self, "perception_source", "YOLO")}, '
                    f'supplement={getattr(self, "supplement_debug", None)}, '
                    f'metric_target={brief}, corner={self.corner_policy.debug}')
        return (
            f'Lane={self.lane_instances}, Crossline={self.crossline_instances}, '
            f'used_boundaries={self.boundary_count}, '
            f'candidates={self.near_candidate_count}, '
            f'near_y={self.active_near_y_ratio:.2f}')

    def _publish_zero(self):
        self.cmd_publisher.publish(Twist())

    def _publish_status(self, text):
        now = self.safety_clock.now()
        unchanged = text == self.last_status_text
        if (unchanged and self.last_status_time is not None
                and (now - self.last_status_time).nanoseconds < 1_000_000_000):
            return
        message = String()
        message.data = text
        self.status_publisher.publish(message)
        self.last_status_text = text
        self.last_status_time = now
        if not unchanged:
            self.get_logger().info(text)

    def destroy_node(self):
        """Stop safely while the ROS context is still valid."""
        if rclpy.ok(context=self.context):
            self._publish_zero()
        debug_worker = getattr(self, 'debug_worker', None)
        if debug_worker is not None:
            debug_worker.close()
        debug_transport = getattr(self, 'debug_transport', None)
        if debug_transport is not None:
            debug_transport.close()
        remote_server = getattr(self, 'remote_server', None)
        if remote_server is not None:
            remote_server.close()
        worker = getattr(self, 'inference_thread', None)
        if worker is not None:
            worker.join(timeout=3.)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LaneAutonomy()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
