"""Exercise production control methods without ROS transport or motors.

The harness uses synthetic masks, a controllable monotonic clock and in-memory
publishers. No model download, robot connection or velocity publication occurs.
"""

from concurrent.futures import Future
from threading import Event
from types import MethodType, SimpleNamespace

import numpy as np
import cv2
from pinky_move.lane_autonomy import DriveState, LaneAutonomy
from pinky_move.metric_lane import MetricLaneTracker
import pytest
from rclpy.clock import ClockType
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from std_srvs.srv import SetBool


class FakeClock:
    """Advance time without sleeping or depending on ROS wall time."""

    def __init__(self):
        self.seconds = 10.0

    def now(self):
        return Time(
            nanoseconds=int(self.seconds * 1e9),
            clock_type=ClockType.STEADY_TIME)


class ClassIds(list):
    """Provide the small tensor API used for result class IDs."""

    def int(self):  # noqa: A003 - match torch.Tensor.int() in the test double.
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


def rectangle(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def detection(second_bottom=99, crossline=False, empty=False):
    classes = ClassIds([] if empty else [1, 1])
    polygons = [] if empty else [
        rectangle(48, 50, 52, 99), rectangle(148, 50, 152, second_bottom)]
    if crossline:
        classes.append(0)
        polygons.append(rectangle(20, 70, 180, 78))
    return SimpleNamespace(
        boxes=SimpleNamespace(cls=classes),
        masks=SimpleNamespace(xy=polygons))


@pytest.fixture
def controller():
    node = SimpleNamespace()
    # Bind the actual implementation to a transport-free state container.
    for name, member in vars(LaneAutonomy).items():
        if name.startswith('__'):
            continue
        if isinstance(member, staticmethod):
            setattr(node, name, member.__func__)
        elif callable(member):
            setattr(node, name, MethodType(member, node))
    parameters = {}
    node.declare_parameter = lambda name, value: parameters.update({name: value})
    node._declare_parameters()
    parameters.update(use_lidar_guard=False, publish_debug_image=False)
    node.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    node.set_parameters = lambda values: parameters.update(
        {p.name: p.value for p in values})
    node.parameters = parameters
    node.safety_clock = FakeClock()
    node.enabled = True
    node.inference_generation = 0
    node.inference_future = None
    node.last_image_time = node.safety_clock.now()
    node.last_inference_time = None
    node.last_scan_time = None
    node.front_clear = True
    node.last_status_text = None
    node.last_status_time = None
    node.inference_seconds = 0.0
    node.lane_instances = 0
    node.crossline_instances = 0
    node.near_candidate_count = 0
    node.lane_class_id = 1
    node.crossline_class_id = 0
    node.commands = []
    node.statuses = []
    node.debug_images = []
    node.cmd_publisher = SimpleNamespace(publish=node.commands.append)
    node.status_publisher = SimpleNamespace(publish=node.statuses.append)
    node.debug_publisher = SimpleNamespace(publish=node.debug_images.append,
                                          get_subscription_count=lambda: 1)
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *a, **k: None, error=lambda *a, **k: None,
        warn=lambda *a, **k: None)
    node._reset_transient_state()
    return node


def deliver(node, result, latency=0.9, refresh_camera=True):
    received = node.safety_clock.now()
    node.safety_clock.seconds += latency
    if refresh_camera:
        node.last_image_time = node.safety_clock.now()
    future = Future()
    future.set_result((
        np.zeros((100, 200, 3), np.uint8), result,
        node.safety_clock.now(), Header()))
    node.inference_future = (future, received, node.inference_generation)
    node._control_loop()


def test_existing_jpeg_camera_frame_decodes_to_bgr(controller):
    frame = np.full((24, 32, 3), (20, 80, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', frame)
    assert ok
    message = CompressedImage(format='jpeg', data=encoded.tobytes())
    decoded = controller._image_to_bgr(message)
    assert decoded.shape == frame.shape
    assert np.max(np.abs(decoded.astype(int) - frame.astype(int))) < 5


def test_invalid_jpeg_camera_frame_is_rejected(controller):
    with pytest.raises(ValueError, match='Invalid compressed camera frame'):
        controller._image_to_bgr(CompressedImage(format='jpeg', data=b'bad'))


def test_trial_disables_at_deadline(controller):
    controller.parameters['max_enabled_seconds'] = 3.0
    controller._enable_callback(SetBool.Request(data=True), SetBool.Response())
    deliver(controller, detection(), latency=.3)
    assert controller.commands[-1].linear.x > 0
    controller.safety_clock.seconds = 13.0
    controller._control_loop()
    assert not controller.enabled
    assert not controller.parameters['enabled']
    assert controller.commands[-1].linear.x == 0
    assert controller.commands[-1].angular.z == 0


def test_metric_continuous_default_and_optional_timer(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.calibration_block_reason = 'mount unverified'
    assert controller.parameters['metric_trial_enabled'] is False
    assert controller._enable_callback(SetBool.Request(data=True), SetBool.Response()).success
    assert controller.enable_deadline_ns is None
    assert controller._calibration_stop_reason() is None
    controller.parameters['max_enabled_seconds'] = 15.
    assert controller._enable_callback(SetBool.Request(data=True), SetBool.Response()).success
    assert controller.enable_deadline_ns == controller.safety_clock.now().nanoseconds + 15_000_000_000
    controller.parameters['max_enabled_seconds'] = -1.
    assert controller._calibration_stop_reason()


def test_metric_invalid_duration_rejected(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    for duration in [float('nan'), float('inf'), -1.]:
        controller.parameters['max_enabled_seconds'] = duration
        response = controller._enable_callback(SetBool.Request(data=True), SetBool.Response())
        assert not response.success
        assert not controller.enabled


def test_no_lidar_keeps_speed_and_duration_limits(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.calibration_block_reason = 'mount unverified'
    controller.parameters.update(metric_trial_enabled=True, max_enabled_seconds=10.,
                                 linear_speed=.02, maximum_angular_speed=.15,
                                 use_lidar_guard=False)
    assert controller._calibration_stop_reason() is None
    controller.parameters['linear_speed'] = .21
    assert controller._calibration_stop_reason()


def test_inferred_metric_command_is_slow(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.parameters['linear_speed'] = .03
    target = dict(x_m=.28, y_m=0., inferred=True, boundary_count=1)
    controller.metric_tracker = SimpleNamespace(update=lambda *args, **kwargs: target)
    assert controller._update_lane_command([], 640, controller.safety_clock.now()) == 1
    assert 0 < controller.latest_linear <= .03
    controller._reset_transient_state()
    assert controller.metric_tracker.confirmed is None
    assert controller.metric_target is None


def test_metric_empty_detection_hold_ramps_and_expires(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    target = dict(x_m=.28, y_m=.01, inferred=False, boundary_count=2)
    def update(masks, *args, **kwargs):
        if not masks:
            raise ValueError('no boundaries detected')
        return target
    controller.metric_tracker = SimpleNamespace(update=update)
    controller._update_lane_command([object()], 640, controller.safety_clock.now())
    for age in (.1, .5, .81):
        controller.safety_clock.seconds = 10.0 + age
        controller.last_image_time = controller.safety_clock.now()
        controller._update_lane_command([], 640, controller.safety_clock.now())
        controller._control_loop()
        speed = controller.commands[-1].linear.x
        if age == .1:
            assert speed == pytest.approx(.01)
        elif age == .5:
            assert 0 < speed < .01
            assert controller.metric_missing_since.nanoseconds == 10_000_000_000
        else:
            assert speed == 0
            assert controller.commands[-1].angular.z == 0


def test_metric_hold_does_not_override_inference_failure(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.metric_missing_since = controller.safety_clock.now()
    controller.inference_error = 'worker failed'
    controller._control_loop()
    assert controller.commands[-1].linear.x == 0
    assert controller.state == DriveState.SAFETY_STOP


def test_requested_metric_parameter_examples(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.parameters.update(linear_velocity=.2, lookahead_distance=.7,
                                 lane_width=.6, metric_path_max_m=.9)
    assert controller._calibration_stop_reason() is None
    assert controller._metric_speed() == .2
    assert controller._metric_lookahead() == .7


def test_single_near_line_turns_until_two_near_boundaries(controller):
    controller.parameters['single_line_turn_requires_path_loss'] = False
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    one = dict(inferred=True, boundary_count=1, visible_side='left',
               actual_curve=[[.15, .07], [.24, .07]])
    far = dict(inferred=False, boundary_count=2,
               left_curve=[[.35, .07], [.45, .07]],
               right_curve=[[.35, -.07], [.45, -.07]])
    near = dict(inferred=False, boundary_count=2,
                left_curve=[[.15, .07], [.27, .07]],
                right_curve=[[.15, -.07], [.27, -.07]])
    controller.metric_target = one
    for step in range(11):
        controller.safety_clock.seconds = 10. + .3*step
        controller._update_turn_observation([object()], controller.safety_clock.now())
        if step < 10:
            assert controller.turn_started_at is None
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.state == DriveState.REACQUIRING_LANE
    assert controller.commands[-1].linear.x == 0
    assert controller.commands[-1].angular.z == pytest.approx(-.08)
    controller.safety_clock.seconds += .2
    controller.last_image_time = controller.safety_clock.now()
    controller.metric_target = far
    controller._update_turn_observation([object(), object()], controller.safety_clock.now())
    controller._control_loop()
    assert controller.state == DriveState.REACQUIRING_LANE
    controller.metric_target = near
    controller._update_turn_observation([object(), object()], controller.safety_clock.now())
    assert controller.turn_started_at is not None
    controller.safety_clock.seconds += .2
    controller.metric_target = near
    controller._update_turn_observation([object(), object()], controller.safety_clock.now())
    assert controller.turn_started_at is None


def test_turn_stops_at_twenty_seconds_or_when_line_disappears(controller):
    controller.parameters['single_line_turn_requires_path_loss'] = False
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.metric_target = dict(x_m=.2, y_m=0., width_m=.16,
                                    inferred=True, boundary_count=1,
                                    visible_side='right',
                                    actual_curve=[[.15, -.07], [.24, -.07]])
    for step in range(11):
        controller.safety_clock.seconds = 10. + .3*step
        controller._update_turn_observation([object()], controller.safety_clock.now())
    assert controller.turn_started_at is not None
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.commands[-1].angular.z == pytest.approx(.08)
    controller.safety_clock.seconds += .7
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.commands[-1].angular.z == 0  # No fresh visible boundary.
    controller.safety_clock.seconds = 33.1
    controller.last_image_time = controller.safety_clock.now()
    controller._update_turn_observation([object()], controller.safety_clock.now())
    controller._control_loop()
    assert controller.turn_exhausted
    assert controller.commands[-1].angular.z == 0


def test_single_line_flicker_does_not_start_turn(controller):
    controller.parameters['single_line_turn_requires_path_loss'] = False
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    one = dict(inferred=True, boundary_count=1, visible_side='left',
               actual_curve=[[.15, .07], [.24, .07]])
    pair = dict(inferred=False, boundary_count=2,
                left_curve=[[.15, .07], [.27, .07]],
                right_curve=[[.15, -.07], [.27, -.07]])
    for seconds in (10., 10.3, 10.6, 10.9):
        controller.safety_clock.seconds = seconds
        controller.metric_target = one
        controller._update_turn_observation([object()], controller.safety_clock.now())
    controller.safety_clock.seconds = 11.2
    controller.metric_target = pair
    controller._update_turn_observation([object(), object()], controller.safety_clock.now())
    for seconds in (11.5, 11.8, 12.1, 12.4, 12.7):
        controller.safety_clock.seconds = seconds
        controller.metric_target = one
        controller._update_turn_observation([object()], controller.safety_clock.now())
    assert controller.turn_started_at is None


def test_confirmed_pair_seeds_image_roles(controller):
    import json
    from pathlib import Path
    from test_metric_lane import masks_for_lines
    calibration = json.loads((Path(__file__).parents[1]/'config/robot_floor_calibration.json').read_text())
    controller.robot_calibration = calibration
    controller.metric_tracker = MetricLaneTracker()
    masks = masks_for_lines()
    target = controller.metric_tracker.update(masks, calibration, 0., lane_width=.16,
                                              path_min_m=.14, path_max_m=.48)
    target = controller.metric_tracker.update(masks, calibration, .3, lane_width=.16,
                                              path_min_m=.14, path_max_m=.48)
    assert controller._image_roles_from_target(masks, target) == {0: 'left', 1: 'right'}


def test_valid_single_path_never_starts_recovery_turn(controller):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.metric_target = dict(inferred=True, boundary_count=1, visible_side='left',
                                    actual_curve=[[.15, .07], [.24, .07]])
    for seconds in np.arange(10., 18., .3):
        controller.safety_clock.seconds = float(seconds)
        controller._update_turn_observation([object()], controller.safety_clock.now())
    assert controller.turn_started_at is None
    assert controller.single_side_since is None


def test_failed_path_with_tracked_boundary_starts_turn_after_delay(controller, monkeypatch):
    import pinky_move.lane_autonomy as module
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    curve = np.array([[.15, .07], [.2, .07], [.26, .07]])
    controller.metric_tracker.observed_side = 'left'
    controller.metric_tracker.observed_curve = curve
    monkeypatch.setattr(module, 'floor_curves', lambda *args: [curve])
    for step in range(11):
        controller.safety_clock.seconds = 10.+.3*step
        controller._update_turn_observation([object()], controller.safety_clock.now())
        if step < 10:
            assert controller.turn_started_at is None
    assert controller.turn_started_at is not None


def test_folded_center_retains_identity_for_delayed_turn(controller, monkeypatch):
    import pinky_move.metric_lane as geometry
    from test_metric_lane import calibration, masks_for_lines
    controller.robot_calibration = calibration()
    masks = masks_for_lines()[:1]
    tracker = controller.metric_tracker
    tracker.update(masks, controller.robot_calibration, 10., lane_width=.16)
    original = geometry.normal_offset
    def fail_center(curve, distance, role='offset'):
        if role == 'center_path':
            raise ValueError('center_path curve folds back')
        return original(curve, distance, role)
    monkeypatch.setattr(geometry, 'normal_offset', fail_center)
    controller.metric_target = None
    for step in range(12):
        now = 10.3 + .3*step
        controller.safety_clock.seconds = now
        with pytest.raises(ValueError, match='center_path curve folds back'):
            tracker.update(masks, controller.robot_calibration, now, lane_width=.16)
        controller._update_turn_observation(masks, controller.safety_clock.now())
        if step < 10:
            assert controller.turn_started_at is None
    assert controller.turn_started_at is not None
    assert controller.turn_side == 'left'
    assert controller.metric_target is None


@pytest.mark.parametrize('exhausted', [False, True])
def test_valid_near_single_path_releases_turn_without_resetting_budget(controller, exhausted):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.turn_started_at = controller.safety_clock.now()
    controller.turn_side = 'left'
    controller.turn_exhausted = exhausted
    controller.metric_target = dict(inferred=True, visible_side='left',
                                    actual_curve=[[.15, .07], [.24, .07]])
    for step in range(3):
        controller.safety_clock.seconds += .3
        controller._update_turn_observation([object()], controller.safety_clock.now())
        if step < 2:
            assert controller.turn_started_at is not None
    assert controller.turn_started_at is None
    assert controller.turn_exhausted  # No new 20-second spin from a single line.


@pytest.mark.parametrize('kind', ['held', 'far', 'opposite', 'gap', 'flicker'])
def test_single_resume_rejects_untrusted_or_noncontinuous_path(controller, kind):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.turn_started_at = controller.safety_clock.now()
    controller.turn_side = 'left'
    for step in range(4):
        controller.safety_clock.seconds += .9 if kind == 'gap' else .3
        target = dict(inferred=True, visible_side='left',
                      actual_curve=[[.15, .07], [.24, .07]])
        if kind == 'held': target['held'] = True
        if kind == 'far': target['actual_curve'] = [[.35,.07],[.45,.07]]
        if kind == 'opposite': target['visible_side'] = 'right'
        if kind == 'flicker' and step % 2: target = None
        controller.metric_target = target
        # Geometry for the recovery turn itself is deliberately not available.
        controller._update_turn_observation([], controller.safety_clock.now())
    assert controller.turn_started_at is not None


def test_inferred_multi_mask_target_seeds_only_its_actual_source(controller):
    target = dict(inferred=True, visible_side='left', source_mask_index=1)
    assert controller._image_roles_from_target([object(), object()], target) == {1: 'left'}
    target['source_mask_index'] = 3
    assert controller._image_roles_from_target([object(), object()], target) == {}


def test_stopped_corner_replay_exits_exhausted_turn(controller):
    from test_metric_lane import calibration, stopped_corner_masks
    from pinky_move.metric_lane import floor_curves
    controller.robot_calibration = calibration()
    masks = stopped_corner_masks()
    tracker = controller.metric_tracker
    tracker.observed_side = 'left'
    tracker.observed_curve = floor_curves([masks[1]], controller.robot_calibration)[0]
    tracker.observed_at = 10.
    tracker.requires_pair = True
    controller.turn_started_at = controller.safety_clock.now()
    controller.turn_side = 'left'
    controller.turn_exhausted = True
    controller.image_match = (1, 'left')
    for step in range(5):
        controller.safety_clock.seconds = 10.3+.3*step
        now = controller.safety_clock.now()
        count = controller._update_lane_command(masks, 640, now)
        controller._update_turn_observation(masks, now)
        if step < 2:
            assert count == 0 and controller.metric_target is None
        if step < 4:
            assert controller.turn_started_at is not None
    assert controller.turn_started_at is None and controller.turn_exhausted
    assert controller.metric_target['visible_side'] == 'left'
    assert controller.metric_target['source_mask_index'] == 1
    assert controller.latest_linear <= .03


def test_measured_right_corner_follows_without_recovery_spin(controller):
    from test_parametric_lane import corner_mask
    from test_metric_lane import calibration
    controller.robot_calibration = calibration()
    for step in range(18):  # More than the 3-second recovery delay.
        controller.safety_clock.seconds = 10.+.2*step
        now = controller.safety_clock.now()
        masks = [corner_mask()]
        assert controller._update_lane_command(masks, 640, now) == 1
        controller._update_turn_observation(masks, now)
        assert controller.metric_target['visible_side'] == 'right'
        assert 0 < controller.latest_linear <= .03
        assert 0 < controller.latest_angular <= controller.parameters['maximum_angular_speed']
        assert controller.turn_started_at is None
    # A missing result still expires; valid corner geometry cannot mask timeout.
    controller.last_image_time = controller.safety_clock.now()
    controller.last_inference_time = controller.safety_clock.now()
    controller.safety_clock.seconds += 2.
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.commands[-1].linear.x == 0
    assert controller.commands[-1].angular.z == 0


def test_slow_valid_detection_can_follow(controller):
    deliver(controller, detection(), latency=0.9)
    assert controller.state == DriveState.FOLLOWING
    assert controller.commands[-1].linear.x > 0
    assert controller.last_lane_time == controller.safety_clock.now()
    controller.safety_clock.seconds += 0.6
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.state == DriveState.FOLLOWING


def test_counts_distinguish_model_instances_and_used_boundaries(controller):
    deliver(controller, detection(second_bottom=59, crossline=True))
    assert controller.lane_instances == 2
    assert controller.crossline_instances == 1
    assert controller.boundary_count == 1
    assert 'Lane=2, Crossline=1, used_boundaries=1' in (
        controller.statuses[-1].data)


def test_pair_search_uses_both_detected_lanes(controller):
    deliver(controller, detection(second_bottom=73))
    assert controller.boundary_count == 2
    assert controller.active_near_y_ratio == pytest.approx(0.74)


def test_no_lane_result_clears_previous_motion(controller):
    deliver(controller, detection())
    deliver(controller, detection(empty=True))
    assert controller.state == DriveState.WAITING_FOR_LANE
    assert controller.commands[-1].linear.x == 0
    assert controller.commands[-1].angular.z == 0
    assert controller.near_center is None


def test_stalled_inference_expires_even_while_camera_is_alive(controller):
    deliver(controller, detection())
    controller.safety_clock.seconds += 1.2
    controller.last_image_time = controller.safety_clock.now()
    controller.inference_future = (
        Future(), controller.last_image_time, controller.inference_generation)
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert 'inference input expired' in controller.statuses[-1].data
    assert controller.commands[-1].linear.x == 0


def test_late_result_is_rejected(controller):
    deliver(controller, detection(), latency=2.1)
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.last_lane_time is None
    assert controller.commands[-1].linear.x == 0


def test_camera_timeout_is_not_refreshed_by_inference_completion(controller):
    deliver(controller, detection(), latency=1.6, refresh_camera=False)
    assert controller.state == DriveState.SAFETY_STOP
    assert 'camera timeout' in controller.statuses[-1].data
    assert controller.commands[-1].linear.x == 0


def test_disable_during_inference_discards_old_generation(controller):
    old_generation = controller.inference_generation
    received = controller.safety_clock.now()
    controller._enable_callback(
        SetBool.Request(data=False), SetBool.Response())
    controller._enable_callback(
        SetBool.Request(data=True), SetBool.Response())
    future = Future()
    future.set_result((
        np.zeros((100, 200, 3), np.uint8), detection(), received, Header()))
    controller.inference_future = (future, received, old_generation)
    controller._control_loop()
    assert controller.state == DriveState.WAITING_FOR_LANE
    assert controller.commands[-1].linear.x == 0


def test_crossline_holds_full_three_seconds_then_does_not_retrigger(controller):
    deliver(controller, detection(crossline=True))
    assert controller.state == DriveState.FOLLOWING
    deliver(controller, detection(crossline=True))
    assert controller.state == DriveState.CROSSLINE_STOP
    for _ in range(3):
        deliver(controller, detection(crossline=True), latency=0.9)
        assert controller.state == DriveState.CROSSLINE_STOP
        assert controller.commands[-1].linear.x == 0
    deliver(controller, detection(crossline=True), latency=0.4)
    assert controller.state == DriveState.FOLLOWING


def test_error_result_stops_and_next_success_recovers(controller):
    deliver(controller, detection())
    future = Future()
    future.set_exception(RuntimeError('test inference failure'))
    controller.inference_future = (
        future, controller.safety_clock.now(), controller.inference_generation)
    controller._control_loop()
    assert controller.state == DriveState.SAFETY_STOP
    assert controller.commands[-1].linear.x == 0
    deliver(controller, detection())
    assert controller.state == DriveState.FOLLOWING


def test_blocked_worker_does_not_block_camera_stop_or_disable(controller):
    started = Event()
    release = Event()

    def predict(**kwargs):
        started.set()
        assert release.wait(3.0)
        return [detection()]

    controller.model = SimpleNamespace(predict=predict)
    frame = np.zeros((100, 200, 3), np.uint8)
    message = controller._bgr_to_message(frame, Header())
    controller._image_callback(message)
    controller._control_loop()
    future = controller.inference_future[0]
    try:
        assert started.wait(1.0)
        controller.safety_clock.seconds += 1.6
        controller._control_loop()
        assert controller.state == DriveState.SAFETY_STOP
        # New camera messages replace a single slot while inference is busy.
        controller._image_callback(message)
        newest = controller._bgr_to_message(frame + 1, Header())
        controller._image_callback(newest)
        assert controller.latest_image[0] is newest
        assert controller.inference_future[0] is future
        controller._enable_callback(
            SetBool.Request(data=False), SetBool.Response())
        assert controller.commands[-1].linear.x == 0
        assert not controller.enabled
    finally:
        release.set()
        future.result(timeout=2.0)
    controller._control_loop()
    assert controller.state == DriveState.DISABLED


def test_debug_image_contains_separate_counts(controller):
    controller.parameters['publish_debug_image'] = True
    rendered = Event()
    def publish(message):
        controller.debug_images.append(message)
        rendered.set()
    controller.debug_publisher.publish = publish
    controller.debug_transport = SimpleNamespace(submit=lambda key, msg: publish(msg))
    try:
        deliver(controller, detection(second_bottom=59, crossline=True))
        assert rendered.wait(2.)
    finally:
        controller.debug_worker.close()
    assert controller.debug_images[-1].encoding == 'bgr8'
    assert controller.lane_instances == 2
    assert controller.boundary_count == 1


def test_detection_expiry_is_distinct_from_no_usable_lane(controller):
    deliver(controller, detection(), latency=0.1)
    controller.safety_clock.seconds += 1.6
    controller.last_image_time = controller.safety_clock.now()
    controller._control_loop()
    assert controller.state == DriveState.WAITING_FOR_LANE
    assert 'result expired' in controller.statuses[-1].data
    assert controller.commands[-1].linear.x == 0


def test_status_heartbeat_repeats_without_state_change(controller):
    controller._publish_status('DISABLED')
    controller._publish_status('DISABLED')
    assert len(controller.statuses) == 1
    controller.safety_clock.seconds += 1.1
    controller._publish_status('DISABLED')
    assert len(controller.statuses) == 2


def test_lidar_guard_still_stops_when_explicitly_enabled(controller):
    controller.parameters['use_lidar_guard'] = True
    controller.front_clear = False
    controller.last_scan_time = controller.safety_clock.now()
    deliver(controller, detection(), latency=0.1)
    assert controller.state == DriveState.SAFETY_STOP
    assert 'obstacle ahead' in controller.statuses[-1].data
    assert controller.commands[-1].linear.x == 0


def test_corner_policy_can_replace_fold_only_after_confirmation(controller):
    from test_lane_corner import observation
    from test_metric_lane import calibration
    controller.robot_calibration = calibration()
    obs = observation()
    def folded(*args, **kwargs):
        raise ValueError('center_path curve folds back')
    controller.metric_tracker = SimpleNamespace(last_observation=obs,update=folded)
    controller._corner_pose_for_frame = lambda now: np.zeros(3)
    for step in range(3):
        controller.safety_clock.seconds = 10.+.2*step
        now = controller.safety_clock.now()
        count = controller._update_lane_command([object()],640,now)
        controller._update_turn_observation([object()],now)
        if step<2:
            assert count==0 and controller.latest_linear==0
        else:
            assert count==1 and 0<controller.latest_linear<=controller._float_parameter('corner_speed_mps')
            assert controller.metric_target['corner_staged']
            assert not controller.metric_target.get('corner_stationary')
        assert controller.turn_started_at is None


@pytest.mark.parametrize('corner', [False, True])
def test_corner_speed_limits_do_not_change_ordinary_single_line_limits(controller, corner):
    controller.robot_calibration = {'method': 'intrinsics_urdf_floor'}
    controller.parameters.update(corner_enabled=corner, corner_speed_mps=.05,
                                 corner_max_angular_speed=.25,
                                 single_line_max_speed=.03, maximum_angular_speed=.15)
    target = dict(x_m=.15, y_m=.10, inferred=True, boundary_count=1)
    if corner:
        target.update(corner_path=True, corner_speed_cap=.05)
    controller.metric_tracker = SimpleNamespace(
        update=lambda *a, **k: target, last_observation=None)
    controller.corner_policy = SimpleNamespace(update=lambda *a, **k: target)
    controller._corner_pose_for_frame = lambda now: None
    assert controller._update_lane_command([object()], 640, controller.safety_clock.now()) == 1
    if corner:
        assert .03 < controller.latest_linear <= .05
        assert .15 < controller.latest_angular <= .25
    else:
        assert controller.latest_linear <= .03
        assert abs(controller.latest_angular) <= .15


def test_invalid_corner_angular_limit_blocks_enable(controller):
    controller.parameters['corner_max_angular_speed'] = float('nan')
    assert controller._calibration_stop_reason() == 'Invalid corner speed limits'


def test_corner_cannot_bypass_ambiguous_identity_or_empty_hold(controller):
    from test_lane_corner import observation
    from test_metric_lane import calibration
    controller.robot_calibration = calibration()
    def ambiguous(*args, **kwargs):
        raise ValueError('visible boundary identity ambiguous')
    controller.metric_tracker=SimpleNamespace(last_observation=observation(),update=ambiguous)
    controller._corner_pose_for_frame=lambda now:np.zeros(3)
    for step in range(4):
        controller.safety_clock.seconds=10.+step*.2
        assert controller._update_lane_command([object()],640,controller.safety_clock.now())==0
        assert controller.latest_linear==0
    controller.corner_policy.block_recovery=True
    controller.last_lane_time=controller.safety_clock.now()
    controller.metric_last_good_target=dict(x_m=.2,y_m=.04,corner_path=True)
    controller._update_lane_command([],640,controller.safety_clock.now())
    assert controller.metric_target is None


def test_corner_odometry_interpolation_duplicates_reset_and_expiry(controller):
    from nav_msgs.msg import Odometry
    msg=Odometry()
    msg.header.frame_id='odom';msg.child_frame_id='base_footprint'
    msg.pose.pose.orientation.w=1.
    msg.header.stamp.sec=1
    controller._corner_odom_callback(msg)
    controller.safety_clock.seconds+=.2
    msg.header.stamp.nanosec=200_000_000
    msg.pose.pose.position.x=.02
    controller._corner_odom_callback(msg)
    controller.corner_capture_stamp=1_100_000_000
    np.testing.assert_allclose(controller._corner_pose_for_frame(10.2),[.01,0.,0.])
    received=controller.corner_odom[0]
    controller.safety_clock.seconds+=.2
    controller._corner_odom_callback(msg)
    assert controller.corner_odom[0]==received
    assert controller._corner_pose_for_frame(10.6) is None
    controller.corner_policy.count=3
    msg.header.stamp.nanosec=400_000_000
    msg.pose.pose.position.x=2.
    controller._corner_odom_callback(msg)
    assert controller.corner_policy.count==0 and len(controller.corner_odom_history)==1
    msg.child_frame_id='unknown'
    controller._corner_odom_callback(msg)
    assert controller.corner_odom is None
