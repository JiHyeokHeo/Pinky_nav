"""Observed-boundary corner classification and bounded local turn planning.

No ROS transport or motor output. Classification is independent of offset-path
success. All geometry is metres in base_link; history landmarks live in odom.
Missing evidence is UNKNOWN, never an instruction to turn blindly.
"""
from dataclasses import dataclass
import numpy as np
from .metric_lane import (arc_stations, resample_chain, nearest_on_chain,
                          first_self_intersection, select_lookahead, curve_match_error)


def wrap(angle):
    return (angle+np.pi) % (2*np.pi)-np.pi


def world_point(point, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])@point+np.array([x, y])


def staged_command(target, pose, now, max_angular, tolerance):
    """Re-evaluate staged motion at control rate with CURRENT odometry.

    Caller must enforce fresh camera, inference and odometry first. This never
    advances the state machine: reaching the pivot stops until a fresh frame
    confirms BRAKE/TURN. It also stops at the yaw goal between inference frames.
    """
    if pose is None or not np.isfinite(pose).all():
        return 0., 0., 'odometry unavailable'
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pivot = np.array([[c, s], [-s, c]])@(np.asarray(target['corner_pivot_world'])-pose[:2])
    distance = float(np.linalg.norm(pivot))
    if target.get('corner_stationary'):
        if distance > tolerance+.015:
            return 0., 0., 'pivot drift; stop'
        if now >= target['corner_spin_deadline']:
            return 0., 0., 'spin deadline reached'
        if now < target['corner_brake_until']:
            return 0., 0., 'braking'
        error = wrap(target['corner_spin_yaw']-pose[2])
        if abs(error) <= np.deg2rad(12):
            return 0., 0., 'heading reached; waiting for near exit path'
        return 0., float(np.clip(error, -max_angular, max_angular)), 'stationary turn'
    if now >= target['corner_entry_deadline']:
        return 0., 0., 'entry deadline reached'
    if distance <= tolerance or pivot[0] <= 0:
        return 0., 0., 'pivot reached; awaiting fresh confirmation'
    speed = min(target['corner_speed_cap'], .5*distance)
    curvature = 2*pivot[1]/max(distance**2, 1e-8)
    speed = min(speed, max_angular/max(abs(curvature), 1e-6))
    return float(speed), float(speed*curvature), 'observed entry approach'


@dataclass
class CornerConfig:
    staged_turn: bool = False
    entry_speed_mps: float = .03
    pivot_tolerance_m: float = .025
    brake_seconds: float = .3
    spin_timeout_s: float = 20.
    angle_deg: float = 50.
    window_m: float = .12
    segment_m: float = .05
    confirm_frames: int = 3
    exit_frames: int = 3
    max_gap_s: float = .6
    landmark_gate_m: float = .05
    heading_gate_deg: float = 20.
    approach_m: float = .30
    speed_mps: float = .01
    min_speed_mps: float = .003
    clearance_m: float = .005
    # Existing Pinky navigation footprint, NOT a guessed smaller robot.
    front_m: float = .060
    rear_m: float = .090
    half_width_m: float = .075

    def validate(self):
        values = [getattr(self, k) for k in self.__dataclass_fields__ if k != 'staged_turn']
        if (not np.isfinite(values).all() or self.clearance_m < 0 or
                any(getattr(self, k) <= 0 for k in self.__dataclass_fields__
                    if k not in ('clearance_m', 'staged_turn'))):
            raise ValueError('invalid corner configuration')
        if (not 20 <= self.angle_deg < 150 or self.window_m < self.segment_m or
                self.confirm_frames < 2 or self.exit_frames < 2 or
                self.confirm_frames != int(self.confirm_frames) or
                self.exit_frames != int(self.exit_frames) or
                self.min_speed_mps > self.speed_mps or
                not .005 <= self.entry_speed_mps <= .03 or
                not .01 <= self.pivot_tolerance_m <= .04 or
                not .2 <= self.brake_seconds <= 2. or
                not 1 <= self.spin_timeout_s <= 20.):
            raise ValueError('invalid corner thresholds')


def _line(points):
    """TLS line oriented in sample order, not by its x component."""
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points-centre, full_matrices=False)
    tangent = vt[0]
    if tangent@(points[-1]-points[0]) < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]])
    return centre, tangent, float(np.sqrt(np.mean(((points-centre)@normal)**2)))


def classify_boundary(curve, config):
    """Recognize S sign reversals or a localized heading change with an exit.

    Chords span ~5 cm to suppress pixel-scale curvature spikes. A corner must
    have two supported, low-residual legs; a fit failure alone is not evidence.
    """
    p = np.asarray(curve, float)
    unknown = dict(kind='UNKNOWN', angle_deg=0., reason='insufficient observed legs')
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 5 or not np.isfinite(p).all():
        return unknown
    length = arc_stations(p)[-1]
    if length < config.segment_m or first_self_intersection(p) is not None:
        return unknown
    p = resample_chain(p, min(160, max(30, int(length/.003))))
    s = arc_stations(p)
    span = max(2, int(config.segment_m/(s[-1]/(len(p)-1))))
    chords = p[span:]-p[:-span]
    theta = np.unwrap(np.arctan2(chords[:, 1], chords[:, 0]))
    change = np.diff(theta)
    positive = float(np.sum(change[change > 0]))
    negative = float(-np.sum(change[change < 0]))
    if min(positive, negative) > np.deg2rad(20):
        return dict(kind='S_BEND', angle_deg=float(np.rad2deg(theta[-1]-theta[0])),
                    reason='opposite sustained heading changes')
    interval = max(1, int(config.window_m/(s[-1]/(len(p)-1))))
    deltas = [theta[min(i+interval, len(theta)-1)]-theta[i] for i in range(len(theta)-1)]
    strongest = max(deltas, key=abs, default=0.)
    result = dict(kind='NORMAL', angle_deg=float(np.rad2deg(strongest)), reason='smooth boundary')
    if abs(strongest) < np.deg2rad(config.angle_deg):
        return result
    options = []
    for split in range(3, len(p)-3):
        if s[split] < config.segment_m or s[-1]-s[split] < config.segment_m:
            continue
        a, tin, ea = _line(p[:split+1])
        b, tout, eb = _line(p[split:])
        angle = wrap(np.arctan2(tout[1], tout[0])-np.arctan2(tin[1], tin[0]))
        if not np.deg2rad(config.angle_deg) <= abs(angle) <= np.deg2rad(130):
            continue
        if max(ea, eb) > .008 or angle*strongest <= 0:
            continue
        matrix = np.column_stack((tin, -tout))
        if abs(np.linalg.det(matrix)) < .1:
            continue
        distances = np.linalg.solve(matrix, b-a)
        corner = a+distances[0]*tin
        # The intersection must be ahead of the entry and before the exit.
        if (corner-p[0])@tin < config.segment_m or (p[-1]-corner)@tout < config.segment_m:
            continue
        if np.min(np.linalg.norm(p-corner, axis=1)) > .04:
            continue
        options.append((ea+eb, dict(kind='CORNER', angle_deg=float(np.rad2deg(angle)),
                       corner=corner, entry=p[0], exit=p[-1], tin=tin, tout=tout,
                       reason='supported entry and exit')))
    if options:
        return min(options, key=lambda item:item[0])[1]
    return dict(kind='UNKNOWN', angle_deg=float(np.rad2deg(strongest)),
                reason='sharp change without reliable entry/exit')


def plan_corner(observation, feature, config, max_angular):
    """Search tangent-connected cubic Beziers and validate swept footprint.

    One boundary implies a width-prior corridor, NOT measured free space. With
    two boundaries every footprint sample must also be inside the other side.
    No plan is accepted outside observed longitudinal support. This is lane
    containment only, not camera-based obstacle detection.
    """
    width = observation['width']
    if width < 2*(config.half_width_m+config.clearance_m):
        raise ValueError('corner: lane width insufficient for footprint clearance')
    side = observation['side']
    inward_sign = -1 if side == 'left' else 1
    normal = lambda t: np.array([-t[1], t[0]])
    # Place the path where the physical rear/front still have observed line
    # support. The later swept-footprint test already checks every path pose;
    # do not add an extra 1 cm pad plus 2 cm residual here as a second gate.
    # Keep 5 mm to avoid placing a swept corner exactly at a sampled endpoint.
    endpoint_pad = .005
    entry_length = float((feature['corner']-feature['entry'])@feature['tin'])
    exit_length = float((feature['exit']-feature['corner'])@feature['tout'])
    if min(entry_length, exit_length) < config.segment_m:
        raise ValueError('corner: observed legs too short to establish turn geometry')
    entry_trim = config.rear_m+config.clearance_m+endpoint_pad
    exit_trim = config.front_m+config.clearance_m+endpoint_pad
    # A straight-leg endpoint trim is only a candidate construction heuristic,
    # not proof of containment. If it consumes a short leg, try its midpoint
    # instead. The unchanged full rectangle sweep below must still pass: do not
    # shrink the robot, extrapolate the boundary, or waive endpoint coverage.
    if entry_length-entry_trim <= .005:
        entry_trim = entry_length*.5
    if exit_length-exit_trim <= .005:
        exit_trim = exit_length*.5
    entry = feature['entry']+entry_trim*feature['tin']
    exit_point = feature['exit']-exit_trim*feature['tout']
    entry_support = float((feature['corner']-entry)@feature['tin'])
    exit_support = float((exit_point-feature['corner'])@feature['tout'])
    if entry_support <= .005 or exit_support <= .005:
        raise ValueError(
            'corner: observed legs too short for swept footprint '
            f'(remaining entry={entry_support:.3f}m, exit={exit_support:.3f}m)')
    boundary = resample_chain(observation['curve'], 100)
    # Sharp corners need not keep the centre at exactly half the lane width.
    # Search from near the observed boundary to the opposite footprint limit.
    # No candidate may put the actual swept robot outside the observed corridor.
    minimum_offset = config.half_width_m+config.clearance_m
    offsets = np.unique([minimum_offset, width*.5, width-minimum_offset])
    for offset in offsets:
        candidates = []
        start = entry+inward_sign*offset*normal(feature['tin'])
        end = exit_point+inward_sign*offset*normal(feature['tout'])
        length = np.linalg.norm(end-start)
        for fraction in (.20, .33, .50, .67):
            controls = np.array([start, start+fraction*length*feature['tin'],
                                 end-fraction*length*feature['tout'], end])
            t = np.linspace(0, 1, 70)[:, None]
            path = ((1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+
                    3*(1-t)*t*t*controls[2]+t**3*controls[3])
            if np.any(path[:, 0] <= .05) or first_self_intersection(path) is not None:
                continue
            ds = arc_stations(path)
            if np.any(np.diff(ds) < 1e-6):
                continue
            tangent = np.gradient(path, axis=0)
            heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
            curvature = np.gradient(heading, ds)
            maximum = float(np.max(abs(curvature)))
            if maximum*config.min_speed_mps > max_angular:
                continue
            # Rectangle vertices AND edge midpoints, enlarged by clearance.
            footprint = np.array([[x,y] for x in (-config.rear_m-config.clearance_m, 0.,
                                                   config.front_m+config.clearance_m)
                                  for y in (-config.half_width_m-config.clearance_m, 0.,
                                             config.half_width_m+config.clearance_m)])
            c, s = np.cos(heading), np.sin(heading)
            swept = np.stack((c[:,None]*footprint[:,0]-s[:,None]*footprint[:,1],
                              s[:,None]*footprint[:,0]+c[:,None]*footprint[:,1]), axis=2)+path[:,None]
            # Starts/ends with no boundary coverage cannot prove containment.
            query = swept.reshape(-1, 2)
            ok = True
            bounds = [(side, boundary)]
            other = observation.get('other')
            if other is not None:
                bounds.append(('right' if side == 'left' else 'left', resample_chain(other, 100)))
            for bound_side, bound in bounds:
                q, idx, f = nearest_on_chain(query, bound)
                d = np.diff(bound, axis=0)[idx]
                d /= np.linalg.norm(d, axis=1)[:, None]
                signed = d[:,0]*(query-q)[:,1]-d[:,1]*(query-q)[:,0]
                inward = signed*(-1 if bound_side == 'left' else 1)
                station = idx+f
                if (np.any(inward < -1e-5) or np.any(inward > width+1e-5) or
                        np.any(station <= .001) or np.any(station >= len(bound)-1.001)):
                    ok = False
                    break
            if ok:
                candidates.append((maximum, path))
        if candidates:
            maximum, path = min(candidates, key=lambda item:item[0])
            return path, min(config.speed_mps, max_angular/max(maximum, 1e-6))
    raise ValueError('corner: no supported footprint-safe turn path')


class CornerPolicy:
    """NORMAL -> CANDIDATE -> APPROACH -> TURN -> EXIT -> NORMAL.

    Every drive target is rebuilt from THIS observation. Odom history only
    confirms a landmark/exit direction; a stored path is never blindly driven.
    """
    def __init__(self, config=None):
        self.config = config or CornerConfig()
        self.config.validate()
        self.reset()

    def reset(self):
        self.state = 'NORMAL'
        self.previous = None
        self.last_time = None
        self.count = 0
        self.exit_count = 0
        self.confirmed_exit = None
        self.staged = None
        self.debug = dict(mode='NORMAL', reason='no corner observation')
        self.block_recovery = False

    def update(self, observation, ordinary, now, pose, lookahead, max_angular):
        cfg = self.config
        if self.last_time is not None and now <= self.last_time:
            self.debug.update(reason='duplicate or out-of-order observation')
            return None
        if self.last_time is not None and now-self.last_time > cfg.max_gap_s:
            self.previous = None
            self.count = 0
            self.exit_count = 0
        self.last_time = now
        if self.staged is not None:
            return self._staged_update(observation, ordinary, now, pose, lookahead)
        if observation is None:
            active = self.block_recovery or self.state not in ('NORMAL', 'S_BEND')
            self.previous = None
            self.count = 0
            self.exit_count = 0
            self.block_recovery = active
            self.debug = dict(mode='UNKNOWN', reason='no fresh trusted boundary')
            return None if active else ordinary
        feature = classify_boundary(observation['curve'], cfg)
        self.debug = dict(mode=feature['kind'], angle_deg=feature['angle_deg'], reason=feature['reason'])
        if feature['kind'] != 'CORNER':
            if self.confirmed_exit is not None:
                if pose is None or ordinary is None or feature['kind'] not in ('NORMAL', 'S_BEND'):
                    self.block_recovery = True
                    self.debug.update(mode='EXIT_WAIT', reason='fresh exit and odometry required')
                    return None
                p = resample_chain(observation['curve'])
                tangent = p[min(15,len(p)-1)]-p[0]
                yaw = np.arctan2(tangent[1],tangent[0])+pose[2]
                aligned = abs(wrap(yaw-self.confirmed_exit)) < np.deg2rad(cfg.heading_gate_deg)
                self.exit_count = self.exit_count+1 if aligned else 0
                self.state = 'EXIT'
                self.debug.update(mode='EXIT', confirmations=self.exit_count)
                self.block_recovery = True
                if not aligned:
                    return None
                ordinary = dict(ordinary, corner_speed_cap=cfg.speed_mps)
                if self.exit_count >= cfg.exit_frames:
                    self.reset()
                return ordinary
            self.previous = None
            self.count = 0
            self.state = feature['kind']
            self.block_recovery = feature['kind'] == 'UNKNOWN'
            return None if self.block_recovery else ordinary
        self.block_recovery = True
        corner_distance = float(np.linalg.norm(feature['corner']))
        # The ordinary centre path remains authoritative while it is valid
        # and the robot front is still beyond the controller's lookahead from
        # the corner. A candidate alone must not replace that path with STOP.
        centre_can_approach = (ordinary is not None and
                               corner_distance > lookahead+cfg.front_m and
                               self.confirmed_exit is None)
        self.debug.update(direction='LEFT' if feature['angle_deg'] > 0 else 'RIGHT',
                          corner_distance_m=corner_distance)
        if pose is None:
            self.previous = None
            self.count = 0
            self.debug.update(mode='CORNER_CANDIDATE', reason='fresh odometry required')
            return None
        landmark = world_point(feature['corner'], pose)
        exit_yaw = wrap(np.arctan2(feature['tout'][1],feature['tout'][0])+pose[2])
        prior = self.previous
        consistent = (prior is not None and prior['side'] == observation['side'] and
                      np.sign(prior['angle']) == np.sign(feature['angle_deg']) and
                      np.linalg.norm(landmark-prior['point']) <= cfg.landmark_gate_m and
                      abs(wrap(exit_yaw-prior['exit'])) <= np.deg2rad(cfg.heading_gate_deg))
        self.count = self.count+1 if consistent else 1
        self.previous = dict(point=landmark, side=observation['side'], angle=feature['angle_deg'], exit=exit_yaw)
        self.state = 'CORNER_CANDIDATE'
        self.debug.update(mode=self.state, confirmations=self.count)
        if self.count < cfg.confirm_frames:
            if centre_can_approach:
                self.block_recovery = False
                self.debug.update(reason='ordinary centre path remains valid')
                return dict(ordinary, corner_speed_cap=cfg.speed_mps)
            return None  # Do not enter a not-yet-confirmed sharp corner.
        self.state = 'APPROACH' if corner_distance > cfg.approach_m else 'TURN'
        self.debug.update(mode=self.state, confirmed=True)
        if cfg.staged_turn and 65 <= abs(feature['angle_deg']) <= 115:
            try:
                self._start_staged(observation, feature, pose, now)
            except ValueError as exc:
                self.debug.update(reason=str(exc))
                return None
            return self._staged_update(observation, ordinary, now, pose, lookahead)
        if centre_can_approach:
            self.block_recovery = False
            self.debug.update(reason='ordinary centre path remains valid')
            return dict(ordinary, corner_speed_cap=cfg.speed_mps)
        try:
            path, cap = plan_corner(observation, feature, cfg, max_angular)
            point, adaptive = select_lookahead(path, lookahead)
        except ValueError as exc:
            self.debug.update(reason=str(exc))
            return None
        target = dict(ordinary or {})
        self.debug.update(reason='validated local turn path')
        target.update(x_m=float(point[0]), y_m=float(point[1]), center_path=path.tolist(),
                      adaptive=adaptive, width_m=observation['width'], normal_width_m=observation['width'],
                      inferred=observation.get('other') is None,
                      boundary_count=1 if observation.get('other') is None else 2,
                      visible_side=observation['side'], actual_curve=observation['curve'].tolist(),
                      corner_speed_cap=cap, corner_path=True,
                      width_source=observation.get('width_source','configured'))
        # Commit only after both path validation and target selection succeed.
        # A rejected candidate must not arm EXIT and block ordinary following.
        # If a previous valid turn exists, a later failed replan retains it.
        self.confirmed_exit = exit_yaw
        return target

    def _start_staged(self, obs, feature, pose, now):
        """Intersect offset entry/exit lines to locate an axle-centre pivot.

        Only the measured entry span is used for approach. The stationary spin
        is NOT a footprint-safe Bezier plan or an obstacle-clearance guarantee.
        """
        width = obs['width']
        if width < 2*(self.config.half_width_m+self.config.clearance_m):
            raise ValueError('corner entry: lane narrower than robot')
        offset = width*.5*(-1 if obs['side'] == 'left' else 1)
        tin, tout = feature['tin'], feature['tout']
        normal = lambda t: np.array([-t[1], t[0]])
        a = feature['entry']+offset*normal(tin)
        b = feature['exit']+offset*normal(tout)
        travel, _ = np.linalg.solve(np.column_stack((tin, -tout)), b-a)
        support = float((feature['corner']-feature['entry'])@tin)
        if not .005 <= travel <= support:
            raise ValueError('corner entry: pivot outside observed entry support')
        pivot = a+travel*tin
        if pivot[0] <= 0 or abs(np.arctan2(tin[1], tin[0])) > np.deg2rad(30):
            raise ValueError('corner entry: robot not aligned with entry')
        self.staged = dict(pivot=world_point(pivot, pose),
            curve=np.array([world_point(p, pose) for p in obs['curve']]),
            side=obs['side'], width=width, started=now, brake_at=None,
            spin_at=None, exit_count=0,
            exit_yaw=wrap(np.arctan2(tout[1], tout[0])+pose[2]))

    def _staged_update(self, obs, ordinary, now, pose, lookahead):
        s, cfg = self.staged, self.config
        self.block_recovery = True  # Never fall through to blind lane-search spin.
        self.debug = dict(mode='ENTRY', reason='approaching observed pivot')
        if pose is None or obs is None or obs['side'] != s['side']:
            self.debug.update(mode='ENTRY_WAIT', reason='fresh same-side boundary and odometry required')
            return None
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        inverse = np.array([[c, sn], [-sn, c]])
        pivot = inverse@(s['pivot']-pose[:2])
        reference = (s['curve']-pose[:2])@inverse.T
        # A fresh line must agree with the odometry-transformed observation;
        # movement never continues merely because a historic pivot exists.
        if curve_match_error(np.asarray(obs['curve']), reference) > .035:
            self.debug.update(mode='ENTRY_WAIT', reason='boundary no longer matches stored corner')
            return None
        target = dict(x_m=float(pivot[0]), y_m=float(pivot[1]), inferred=True,
            boundary_count=1, visible_side=s['side'], actual_curve=obs['curve'].tolist(),
            width_m=s['width'], normal_width_m=s['width'], corner_speed_cap=cfg.entry_speed_mps,
            corner_staged=True, adaptive=True, corner_pivot_world=s['pivot'].tolist(),
            corner_entry_deadline=s['started']+20.)
        if s['brake_at'] is None:
            if now-s['started'] > 20.:
                self.debug.update(mode='ENTRY_TIMEOUT', reason='entry time limit; reset required')
                return None
            if np.linalg.norm(pivot) <= cfg.pivot_tolerance_m:
                s['brake_at'] = now
            elif pivot[0] <= 0:
                self.debug.update(mode='ENTRY_WAIT', reason='pivot passed; no reverse correction')
                return None
            else:
                self.debug.update(remaining_m=float(np.linalg.norm(pivot)))
                target['corner_speed_cap'] = min(cfg.entry_speed_mps, .5*float(np.linalg.norm(pivot)))
                return target
        self.debug.update(mode='BRAKE', reason='stop before stationary turn')
        target.update(corner_stationary=True, corner_spin_yaw=s['exit_yaw'],
                      corner_spin_deadline=s['brake_at']+cfg.brake_seconds+cfg.spin_timeout_s,
                      corner_brake_until=s['brake_at']+cfg.brake_seconds)
        if now < s['brake_at']+cfg.brake_seconds:
            return target
        if now >= target['corner_spin_deadline']:
            self.debug.update(mode='TURN_TIMEOUT', reason='20 second spin limit; reset required')
            return None
        error = wrap(s['exit_yaw']-pose[2])
        self.debug.update(mode='PIVOT_TURN', reason='stationary turn to exit heading',
                          heading_error_deg=float(np.rad2deg(error)))
        near_exit = (ordinary is not None and not ordinary.get('held') and
                     .05 < ordinary['x_m'] <= .30 and
                     abs(np.arctan2(ordinary['y_m'], ordinary['x_m'])) <= np.deg2rad(20))
        aligned = abs(error) <= np.deg2rad(12)
        s['exit_count'] = s['exit_count']+1 if aligned and near_exit else 0
        if s['exit_count'] >= cfg.exit_frames:
            result = dict(ordinary, corner_speed_cap=cfg.entry_speed_mps)
            self.reset()
            return result
        return target
