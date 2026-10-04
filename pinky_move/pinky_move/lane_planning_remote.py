"""PC planning, with no ROS node, publisher, service or motor endpoint.

Reuse the tested lane implementation in a transport-free state container.
Robot monotonic timestamps are opaque: the PC never supplies its own clock.
"""
from copy import deepcopy
from types import MethodType, SimpleNamespace
import json
import numpy as np
from rclpy.clock import ClockType
from rclpy.time import Time


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def restore_stage(value):
    value = deepcopy(value)
    if value is not None:
        for key in ('pivot', 'curve', 'matched_pose'):
            value[key] = np.asarray(value[key], float)
        if value.get('blind_entry') is not None:
            value['blind_entry']['last_pose'] = np.asarray(value['blind_entry']['last_pose'], float)
    return value


TIMES = ('last_lane_time', 'metric_missing_since', 'turn_started_at', 'turn_seen_at',
         'single_side_since', 'single_side_last_seen')
VALUES = ('turn_side', 'turn_exhausted', 'near_pair_streak', 'single_side_candidate')


class PCPlanner:
    def __init__(self):
        self.key = None
        self.sequence = -1

    def _new(self, params):
        from .lane_autonomy import LaneAutonomy
        node = SimpleNamespace()
        # Explicit allowlist: no constructor, executor, control loop, services
        # or command publisher is installed in this container.
        methods = ('_declare_parameters', '_reset_transient_state', '_process_result',
                   '_update_lane_command', '_apply_metric_target', '_metric_command', '_invalidate_lane',
                   '_image_roles_from_target', '_near_curve_visible', '_near_pair_visible',
                   '_update_turn_observation', '_metric_speed', '_metric_lookahead',
                   '_string_parameter', '_float_parameter', '_int_parameter', '_bool_parameter')
        for name in methods:
            setattr(node, name, MethodType(getattr(LaneAutonomy, name), node))
        defaults = {}
        node.declare_parameter = lambda k, v: defaults.update({k: v})
        node._declare_parameters()
        defaults.update(params)
        defaults.update(remote_geometry=False, publish_debug_image=False)
        node.get_parameter = lambda k: SimpleNamespace(value=defaults[k])
        node.get_logger = lambda: SimpleNamespace(warn=lambda *a, **k: None)
        node.inference_generation = 0
        node.enabled = False
        node._reset_transient_state()
        node.lane_class_id, node.crossline_class_id = 1, 0
        node._update_crossline = lambda *args: None  # Robot retains stop-line authority.
        self.node = node

    def process(self, request, result, frame):
        ctx = request['planning']
        session, sequence = request['token'].rsplit(':', 1)
        key = (session, ctx['generation'], json.dumps(ctx['parameters'], sort_keys=True),
               json.dumps(ctx['calibration'], sort_keys=True))
        if key != self.key:
            self._new(ctx['parameters'])
            self.key, self.sequence = key, -1
        if int(sequence) <= self.sequence:
            raise ValueError('out-of-order planning input')
        self.sequence = int(sequence)
        node = self.node
        node.enabled = ctx['enabled']
        node.robot_calibration = ctx['calibration']
        now = Time(nanoseconds=ctx['now_ns'], clock_type=ClockType.STEADY_TIME)
        node.safety_clock = SimpleNamespace(now=lambda: now)
        pose = np.asarray(ctx['pose'], float) if ctx['pose'] is not None else None
        node._corner_pose_for_frame = lambda _: pose
        # Robot owns arrival latch, accumulated blind travel, and deadlines.
        # Echo its latest state before each plan; never invent a new session.
        node.corner_policy.staged = restore_stage(ctx['staged'])
        header = SimpleNamespace(stamp=SimpleNamespace(sec=request['capture_sec'],
                    nanosec=request['capture_nanosec']), frame_id=request['frame_id'])
        node._process_result(frame, result, now, header)
        target = node.metric_target
        if target is not None and not target.get('corner_staged') and not target.get('held') and pose is not None:
            c, s = np.cos(pose[2]), np.sin(pose[2])
            target['remote_world_target'] = (np.array([[c, -s], [s, c]])@
                np.array([target['x_m'], target['y_m']])+pose[:2]).tolist()
            node.metric_last_good_target = target.copy()
        return plain(dict(version=1, token=request['token'], generation=ctx['generation'],
            capture_ns=request['capture_sec']*1_000_000_000+request['capture_nanosec'],
            target=node.metric_target, error=getattr(node, 'metric_error', None),
            staged=node.corner_policy.staged, debug=node.corner_policy.debug,
            block_recovery=node.corner_policy.block_recovery,
            observation=getattr(node.metric_tracker, 'last_observation', None),
            labels=node.debug_boundary_labels,
            times={k: getattr(node, k).nanoseconds if getattr(node, k, None) is not None else None for k in TIMES},
            values={k: getattr(node, k, None) for k in VALUES}))


def validate_plan(plan, token, generation, capture_ns):
    """Reject nonfinite/malformed/replayed plans before touching robot state."""
    if (not isinstance(plan, dict) or plan.get('version') != 1 or
            plan.get('token') != token or plan.get('generation') != generation or
            plan.get('capture_ns') != capture_ns):
        raise ValueError('planning identity mismatch')
    json.dumps(plan, allow_nan=False)  # Includes nested paths, odom and deadlines.
    target = plan.get('target')
    if target is not None:
        if not isinstance(target, dict) or target.get('boundary_count') not in (0, 1, 2):
            raise ValueError('invalid planned target')
        for key in ('x_m', 'y_m'):
            if type(target.get(key)) not in (float, int) or abs(target[key]) > 5.:
                raise ValueError('invalid metric target coordinate')
        if target.get('corner_staged'):
            stage = plan.get('staged')
            if stage is None or np.asarray(stage.get('pivot')).shape != (2,):
                raise ValueError('missing staged pivot')
            if not np.allclose(target.get('corner_pivot_world'), stage['pivot'], atol=1e-9):
                raise ValueError('inconsistent staged pivot')
        for key in ('corner_speed_cap', 'corner_angular_cap'):
            if key in target and (type(target[key]) not in (int, float) or not 0 <= target[key] <= 1.):
                raise ValueError('invalid planned speed cap')
    if not isinstance(plan.get('debug'), dict) or type(plan.get('block_recovery')) is not bool:
        raise ValueError('invalid planning status')
    if set(plan.get('times', {})) != set(TIMES) or set(plan.get('values', {})) != set(VALUES):
        raise ValueError('incomplete planning state')
    return plan
