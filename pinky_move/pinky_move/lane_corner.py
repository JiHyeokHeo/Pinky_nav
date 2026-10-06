"""Observed-boundary corner classification and bounded local turn planning.

No ROS transport or motor output. Classification is independent of offset-path
success. All geometry is metres in base_link; history landmarks live in odom.
Missing evidence is UNKNOWN, never an instruction to turn blindly.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np
from .lane_history import LocalLaneMap
from .metric_lane import (arc_stations, resample_chain, nearest_on_chain,
                          first_self_intersection, select_lookahead, curve_match_error,
                          center_path_from_boundary, center_from_pair)


def wrap(angle):
    return (angle+np.pi) % (2*np.pi)-np.pi


def world_point(point, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])@point+np.array([x, y])


def match_corner_fragment(current, reference):
    """Match a visible fragment to a fixed odom-corrected corner polyline.

    Unlike ordinary forward-lane association, a turning robot may see the
    exit laterally or sample it in reverse order. Use segment projection and
    arc-length coverage without clipping by robot x. Never extend endpoints.
    Return diagnostic numbers even on rejection; do not mutate the landmark.
    """
    info = dict(fragment_ok=False, fragment_reason='invalid geometry')
    a, b = np.asarray(current, float), np.asarray(reference, float)
    if any(p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all()
           for p in (a, b)):
        return info
    if min(arc_stations(a)[-1], arc_stations(b)[-1]) < .03:
        return dict(info, fragment_reason='less than 3cm measured support')
    a, b = resample_chain(a, 80), resample_chain(b, 120)
    q, index, fraction = nearest_on_chain(a, b)
    distance = np.linalg.norm(a-q, axis=1)
    ta = np.gradient(a, axis=0)
    tb = np.diff(b, axis=0)[index]
    cosine = np.sum(ta*tb, axis=1)/np.maximum(
        np.linalg.norm(ta, axis=1)*np.linalg.norm(tb, axis=1), 1e-12)
    supported = (distance <= .035) & (np.abs(cosine) >= .82)
    runs = np.split(np.flatnonzero(supported),
                    np.flatnonzero(np.diff(np.flatnonzero(supported)) > 1)+1)
    run = max(runs, key=len)
    info.update(fragment_fraction=float(len(run)/len(a)),
                fragment_error_m=float(np.median(distance)))
    if len(run) < .4*len(a):
        return dict(info, fragment_reason='insufficient contiguous matching fraction')
    stations = arc_stations(b)
    along = stations[index]+fraction*np.diff(stations)[index]
    span = float(np.ptp(along[run]))
    direction = 1. if np.median(cosine[run]) >= 0 else -1.
    progress = direction*np.diff(along[run])
    info.update(fragment_span_m=span, fragment_reversed=direction < 0)
    partial = len(run)/len(a) < .7
    # Cropped corners can match only 40% of the chain. Require a precise,
    # long, direction-consistent segment rather than accepting random overlap.
    if partial and (np.quantile(distance[run], .9) > .01 or
                    np.quantile(np.abs(cosine[run]), .1) < .95):
        return dict(info, fragment_reason='partial overlap lacks precise correspondence')
    # Do not accept a translated L merely because one long leg still overlaps:
    # another substantial parallel leg displaced from the reference contradicts
    # the stored landmark. Cropped/turning continuations are not this case.
    if partial and np.mean((distance > .04) & (np.abs(cosine) >= .95)) > .2:
        return dict(info, fragment_reason='displaced parallel segment contradicts corner')
    # Short fragments need stronger evidence, not a blanket relaxed gate.
    # Live failure: 39.3 mm support, 100% correspondence, ~5.2 mm error.
    strong = (len(run)/len(a) >= .9 and np.quantile(distance[run], .9) <= .015
              and np.quantile(np.abs(cosine[run]), .1) >= .95)
    minimum = .05 if partial else (.03 if strong else .04)
    info.update(fragment_min_support_m=minimum, fragment_strong=bool(strong))
    if span < minimum or arc_stations(a[run])[-1] < minimum:
        return dict(info, fragment_reason='endpoint-only or short overlap')
    if np.any(progress < -.005) or np.sum(np.maximum(-progress, 0.)) > .01:
        return dict(info, fragment_reason='inconsistent segment order')
    return dict(info, fragment_ok=True, fragment_reason='matched visible corner segment')


def trim_corner_endpoint_extensions(current, reference):
    """기록 끝점을 지나 길어진 현재 선만 잘라 관측 지지의 길이를 맞춘다.

    새로 보이는 긴 출구를 분모에 넣으면 같은 차선의 대응률이 감소한다.
    끝점 바깥, 끝 선분과 15mm/10도 이내로 일직선인 앞/뒤 연결 부분만
    제외한다. 내부 불일치·평행한 다른 차선·분기는 제거하지 않는다.
    """
    a, b = np.asarray(current, float), np.asarray(reference, float)
    if len(a) < 5 or len(b) < 5:
        return a
    q, index, fraction = nearest_on_chain(a, b)
    tangent = np.gradient(a, axis=0)
    extension = np.zeros(len(a), bool)
    for point, segment, endpoint, outer in (
            (b[0], b[1]-b[0], 0, -1),
            (b[-1], b[-1]-b[-2], len(b)-2, 1)):
        segment = segment/max(np.linalg.norm(segment), 1e-12)
        delta = a-point
        normal = np.array([-segment[1], segment[0]])
        cosine = np.abs(tangent@segment)/np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)
        at_endpoint = (index == endpoint) & ((fraction <= .001) if outer < 0 else (fraction >= .999))
        extension |= (at_endpoint & (outer*(delta@segment) > 0.) &
                      (np.abs(delta@normal) <= .015) &
                      (cosine >= np.cos(np.deg2rad(10))))
    lo, hi = 0, len(a)
    while lo < hi and extension[lo]:
        lo += 1
    while hi > lo and extension[hi-1]:
        hi -= 1
    return a[lo:hi] if hi-lo >= 5 else a


def staged_command(target, pose, now, max_angular, tolerance):
    """Re-evaluate staged motion at control rate with CURRENT odometry.

    Caller must enforce fresh camera, inference and odometry first. This never
    advances the state machine itself: the caller latches arrival separately,
    and a subsequent fresh frame selects BRAKE/TURN. It also stops at the yaw
    goal and the no-boundary reacquisition deadline between inference frames.
    """
    if pose is None or not np.isfinite(pose).all():
        return 0., 0., 'odometry unavailable'
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pivot = np.array([[c, s], [-s, c]])@(np.asarray(target['corner_pivot_world'])-pose[:2])
    distance = float(np.linalg.norm(pivot))
    if now >= target.get('corner_blind_deadline', float('inf')):
        return 0., 0., 'boundary reacquisition deadline reached'
    budget = target.get('corner_blind_entry')
    if budget is not None and not target.get('corner_stationary'):
        budget['distance'] += float(np.linalg.norm(pose[:2]-budget['last_pose']))
        budget['last_pose'] = np.asarray(pose[:2]).copy()
        if budget['distance'] >= .03:
            return 0., 0., 'blind entry 3cm limit reached'
        if abs(wrap(pose[2]-target['corner_entry_yaw'])) > np.deg2rad(15):
            return 0., 0., 'blind entry heading deviation'
        max_angular = min(max_angular, .15)
    if target.get('corner_stationary'):
        if target.get('corner_heading_reached'):
            return 0., 0., 'turn complete; waiting for exit lane'
        if now >= target.get('corner_blind_deadline', float('inf')):
            return 0., 0., 'boundary reacquisition deadline reached'
        max_angular = min(max_angular, target.get('corner_angular_cap', max_angular))
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
    yaw = target.get('corner_entry_yaw', pose[2])
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    delta = np.asarray(target['corner_pivot_world'])-pose[:2]
    along, lateral = float(delta@axis), float(delta@np.array([-axis[1],axis[0]]))
    if budget is not None and abs(lateral) > tolerance:
        return 0., 0., 'blind entry lateral deviation'
    if now >= target.get('corner_progress_deadline', float('inf')):
        return 0., 0., 'entry progress timeout'
    if along < -tolerance or abs(lateral) > 2*tolerance:
        return 0., 0., 'entry overshoot or lateral deviation'
    if along <= tolerance:
        return 0., 0., 'entry plane reached; awaiting fresh confirmation'
    if distance <= tolerance or pivot[0] <= 0:
        return 0., 0., 'pivot reached; awaiting fresh confirmation'
    speed = min(target['corner_speed_cap'], .5*distance)
    curvature = 2*pivot[1]/max(distance**2, 1e-8)
    speed = min(speed, max_angular/max(abs(curvature), 1e-6))
    return float(speed), float(speed*curvature), f'entry remaining={along:.3f}m lateral={lateral:.3f}m'


@dataclass
class CornerConfig:
    staged_turn: bool = False
    relaxed_tracking: bool = False
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
        flags = ('staged_turn', 'relaxed_tracking')
        values = [getattr(self, k) for k in self.__dataclass_fields__ if k not in flags]
        if (not np.isfinite(values).all() or self.clearance_m < 0 or
                any(getattr(self, k) <= 0 for k in self.__dataclass_fields__
                    if k not in ('clearance_m', *flags))):
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


def sustained_s_bend(theta, stations, minimum_length):
    """Require two spatially supported, opposite heading excursions.

    Summing all positive/negative pixel jitter can invent an S. Instead each
    direction must change by >20 degrees across an observed arc >= one chord.
    No upper angle limit: wide S bends are not automatically right-angle turns.
    """
    theta, stations = np.asarray(theta), np.asarray(stations)
    supported = stations[None, :]-stations[:, None] >= minimum_length
    delta = theta[None, :]-theta[:, None]
    return bool(np.any(supported & (delta > np.deg2rad(20))) and
                np.any(supported & (delta < -np.deg2rad(20))))


def limit_bend_entry(target):
    """Do not aim beyond a measured bend before reaching its entrance.

    Only shorten a valid target along its OWN centre path. Never add straight
    travel, change lane offset, extrapolate pixels, or override a stopped plan.
    A 5cm chord suppresses tiny tangent spikes. Within 6cm of the entrance,
    release the cap so ordinary pursuit can start following the curve.
    """
    if target is None or target.get('held') or target.get('corner_staged'):
        return target
    if 'approach_path' in target:
        # A committed S route may replace center_path after an earlier cap.
        # Recompute display segments instead of carrying stale local geometry.
        target = dict(target)
        for key in ('approach_path', 'bend_remaining_path', 'bend_entry_limited',
                    'bend_entry_arc_m', 'bend_entry_distance_m'):
            target.pop(key, None)
    path = np.asarray(target.get('center_path', []), float)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 5 or
            not np.isfinite(path).all() or first_self_intersection(path) is not None):
        return target
    p = resample_chain(path, 100)
    s = arc_stations(p)
    if s[-1] < .10:
        return target
    span = max(2, int(.05/(s[-1]/99)))
    if span >= len(p)-2:
        return target
    chords = p[span:]-p[:-span]
    theta = np.unwrap(np.arctan2(chords[:,1], chords[:,0]))
    bends = np.flatnonzero(np.abs(theta-theta[0]) >= np.deg2rad(20))
    if not len(bends):
        return target
    i = int(bends[0])
    entry = p[i]
    if s[i] < .04 or entry[0] <= .05 or np.linalg.norm(entry) <= .06:
        return target
    _, idx, fraction = nearest_on_chain(np.array([[target['x_m'], target['y_m']]]), p)
    target_s = s[idx[0]]+fraction[0]*(s[idx[0]+1]-s[idx[0]])
    if target_s <= s[i]:
        return target
    limited = dict(target, x_m=float(entry[0]), y_m=float(entry[1]), adaptive=True,
                bend_entry_limited=True, bend_entry_arc_m=float(s[i]),
                bend_entry_distance_m=float(np.linalg.norm(entry)),
                # Keep approach and outgoing geometry separate. Full path is
                # retained for S continuity and exit-direction verification.
                approach_path=p[:i+1].tolist(),
                bend_remaining_path=p[i:].tolist())
    if target.get('s_route_tracking'):
        limited['s_route_target_arc_m'] = float(s[i])
    return limited


def first_sharp_corner_prefix(curve, config, minimum_entry_m=.12):
    """연속 직각 S의 첫 실제 두 다리만 선택한다. 부드러운 S는 바꾸지 않는다.

    8mm RDP 오차 내의 관측 골격에서 첫 두 다리가 각각 12cm 이상이고
    75~115도인 경우만 분리한다. 다음 코너 직전 3cm는 제외하며, 기존
    직선 잔차/출구 지지 검사를 다시 통과해야 한다. 외삽하지 않는다.
    """
    import cv2
    source=np.asarray(curve,float)
    if len(source)<5 or not np.isfinite(source).all():
        return None
    p=cv2.approxPolyDP(source.astype(np.float32).reshape(-1,1,2),.008,False).reshape(-1,2)
    if len(p)<3:
        return None
    if np.max(np.linalg.norm(source-nearest_on_chain(source,p)[0],axis=1))>.008+1e-6:
        return None
    delta=np.diff(p,axis=0)
    lengths=np.linalg.norm(delta,axis=1)
    selected=None
    for i in range(1,len(p)-1):
        angle=wrap(np.arctan2(delta[i,1],delta[i,0])-np.arctan2(delta[i-1,1],delta[i-1,0]))
        degrees=abs(np.rad2deg(angle))
        if 75<=degrees<=115:
            if lengths[i-1]<minimum_entry_m or lengths[i]<.12:
                return None  # 실제 짧은 첫 코너를 건너 다음 코너를 택하지 않는다.
            selected=i
            break
        if degrees>20 or lengths[i-1]>=config.segment_m:
            return None  # 짧은 작은 방향 변화만 선두 잡음으로 제외한다.
    if selected is None:
        return None
    i=selected
    end=p[i+1]-(.03 if len(p)>i+2 else 0.)*delta[i]/lengths[i]
    # 생 픽셀 골격의 접선 지터를 다시 분류에 넣지 않는다. 위치 오차가
    # 검증된 같은 관측 폴리라인의 첫 두 다리만 재샘플링한다.
    prefix=resample_chain(np.vstack((p[i-1:i+1],end)),100)
    if classify_boundary(prefix,config)['kind']!='CORNER':
        return None
    return prefix


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
    if sustained_s_bend(theta, (s[span:]+s[:-span])*.5, config.segment_m):
        return dict(kind='S_BEND', angle_deg=float(np.rad2deg(theta[-1]-theta[0])),
                    reason='opposite sustained heading changes')
    interval = max(1, int(config.window_m/(s[-1]/(len(p)-1))))
    deltas = [theta[min(i+interval, len(theta)-1)]-theta[i] for i in range(len(theta)-1)]
    strongest = max(deltas, key=abs, default=0.)
    result = dict(kind='NORMAL', angle_deg=float(np.rad2deg(strongest)), reason='smooth boundary')
    # Do not gate observed entry/exit lines on a short sliding heading window:
    # a rounded 90-degree bend may distribute rotation across several windows.
    # Two supported low-residual legs below are stronger geometric evidence.
    net_heading = abs(float(theta[-1]-theta[0]))
    if abs(strongest) < np.deg2rad(config.angle_deg) and net_heading < np.deg2rad(65):
        return result
    options = []
    for split in range(3, len(p)-3):
        if s[split] < config.segment_m or s[-1]-s[split] < config.segment_m:
            continue
        a, tin, ea = _line(p[:split+1])
        b, tout, eb = _line(p[split:])
        angle = wrap(np.arctan2(tout[1], tout[0])-np.arctan2(tin[1], tin[0]))
        if (abs(strongest) < np.deg2rad(config.angle_deg) and
                not np.deg2rad(65) <= abs(angle) <= np.deg2rad(115)):
            continue  # New evidence path is only for right-angle staging.
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
    if abs(strongest) < np.deg2rad(config.angle_deg):
        return result
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
        self.corner_history = deque(maxlen=40)
        self.s_bend_anchor = None
        self.s_route = None
        self.s_route_reacquire = None
        self.s_route_soft_valid = False
        self.floor_calibration = None
        self.local_lane_map = LocalLaneMap(retention_s=10.)

    @staticmethod
    def _boundary_overlap(actual, stored, tolerance=.025, cosine_min=.9, minimum=.02):
        """Compare shared arc support, not novel tails against old endpoints.

        Exclude only leading/trailing points projected beyond a stored endpoint
        in its tangent direction. Interior disagreements remain in the score.
        Resampling avoids segmentation vertex density bias.
        """
        actual, stored = np.asarray(actual,float), np.asarray(stored,float)
        detail = dict(reason='invalid boundary', overlap_m=0., ratio=0.)
        if any(p.ndim!=2 or p.shape[1]!=2 or len(p)<5 or
               not np.isfinite(p).all() for p in (actual,stored)):
            return False, detail, actual
        if (arc_stations(actual)[-1] < minimum or arc_stations(stored)[-1] < minimum or
                first_self_intersection(actual) is not None):
            return False, detail, actual
        actual=resample_chain(actual,min(300,max(20,int(arc_stations(actual)[-1]/.005)+1)))
        q,idx,fraction=nearest_on_chain(actual,stored)
        stations=arc_stations(stored)
        matched=stations[idx]+fraction*np.diff(stations)[idx]
        tangent=np.diff(stored,axis=0)[idx]
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-12)
        heading=np.gradient(actual,axis=0)
        cosines=np.sum(heading*tangent,axis=1)/np.maximum(np.linalg.norm(heading,axis=1),1e-12)
        beyond_start=(matched<=1e-6)&(np.sum((actual-q)*tangent,axis=1)<0)
        beyond_end=(matched>=stations[-1]-1e-6)&(np.sum((actual-q)*tangent,axis=1)>0)
        common=~(beyond_start|beyond_end)
        indices=np.flatnonzero(common)
        detail['new_points']=int(np.count_nonzero(~common))
        if len(indices)<5 or np.any(np.diff(indices)!=1):
            detail['reason']='no contiguous common support'
            return False,detail,actual
        distances=np.linalg.norm(actual-q,axis=1)[common]
        order=matched[common]
        good=(distances<=tolerance)&(cosines[common]>=cosine_min)
        overlap=float(order[-1]-order[0])
        backstep=float(np.min(np.diff(order)))
        detail.update(overlap_m=overlap,ratio=float(np.mean(good)),
                      max_error_m=float(np.max(distances)),
                      min_cosine=float(np.min(cosines[common])),backstep_m=backstep)
        ok=(overlap>=minimum and np.mean(good)>=.7 and
            np.max(distances)<=2*tolerance and backstep>=-.002)
        detail['reason']=('matched common support' if ok else
            'insufficient overlap' if overlap<minimum else
            'reversed correspondence' if backstep<-.002 else 'distance/tangent disagreement')
        return ok,detail,actual

    def _refresh_s_boundary(self, route, obs, pose, now):
        """Append a verified new real-boundary tail, preserving pending points."""
        if obs.get('curve') is None or route.get('boundary_world') is None:
            return
        actual=np.array([world_point(p,pose) for p in obs['curve']])
        stored=route['boundary_world']
        ok,detail,actual=self._boundary_overlap(actual,stored,.01,.95,.05)
        self.debug.update({'s_boundary_'+k:v for k,v in detail.items()})
        if not ok:
            return  # A failed refresh never erases or replaces verified history.
        _,idx,fraction=nearest_on_chain(actual,stored)
        stations=arc_stations(stored)
        order=stations[idx]+fraction*np.diff(stations)[idx]
        end_direction=stored[-1]-stored[-2]
        end_direction/=np.linalg.norm(end_direction)
        beyond=(order>=stations[-1]-1e-6)&((actual-stored[-1])@end_direction>.005)
        suffix=np.flatnonzero(beyond)
        appended=False
        if (len(suffix) and np.all(beyond[suffix[0]:]) and
                np.linalg.norm(actual[suffix[0]]-stored[-1])<=.02):
            joined=np.vstack((stored,actual[suffix[0]:]))
            if len(joined)<=600 and first_self_intersection(joined) is None:
                route['boundary_world']=joined
                appended=True
        route['boundary_verified_at']=now
        self.debug['s_boundary_tail_appended']=appended

    def _near_boundary_out_of_view(self, hidden, route, pose, strict=False):
        """Check recorded REAL boundary pixels, not the offset centre's x.

        Calibration is a mounting estimate, so this only supplements the
        short-prefix bound. Strict mode requires actual image exit, not merely
        the bottom image band. Missing calibration never grants an exception.
        """
        from .robot_projection import robot_floor_pixel, validate_calibration
        cal = self.floor_calibration
        boundary = route.get('boundary_world')
        if cal is None or boundary is None:
            return False
        try:
            validate_calibration(cal)
        except (ValueError, KeyError, TypeError):
            return False
        q, _, _ = nearest_on_chain(hidden, boundary)
        if np.max(np.abs(np.linalg.norm(hidden-q,axis=1)-route['width']/2)) > .04:
            return False
        c,s = np.cos(pose[2]),np.sin(pose[2])
        local = (q-pose[:2])@np.array([[c,-s],[s,c]])
        for x,y in local[:-2]:  # Endpoint may be the first newly visible sample.
            try:
                _,v=robot_floor_pixel(x,y,cal,cal['image_size'])
                if strict or v < .90*cal['image_size'][1]:
                    return False
            except ValueError as exc:
                if str(exc) not in ('Target outside image','Target behind camera'):
                    return False
        return True

    def _track_s_route(self, target, obs, pose, now, lookahead):
        """Commit observed S geometry in odom; append only a matching tail.

        Progress is projection onto a LOCAL arc window bounded by actual odom
        displacement, never the nearest point on a distant returning S leg.
        Fresh observations must support the next committed section. No blind
        replay, joining across missing bends, or rewriting the untraversed path.
        """
        recovery = self.s_route_reacquire
        self.s_route_reacquire = None  # Any rejected geometry breaks confirmation.
        previous_soft = self.s_route_soft_valid
        self.s_route_soft_valid = False
        soft_match = False
        relaxed = self.config.relaxed_tracking
        if pose is None or not np.isfinite(pose).all() or target.get('held'):
            raise ValueError('S route requires fresh pose and observed path')
        path = np.asarray(target.get('center_path', []), float)
        if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 5 or
                not np.isfinite(path).all() or first_self_intersection(path) is not None):
            raise ValueError('S route requires connected centre path')
        path = resample_chain(path, 100)
        world = np.array([world_point(p, pose) for p in path])
        route = dict(self.s_route) if self.s_route is not None else None
        if route is None:
            route = dict(path=world, pose=np.array(pose), time=now,
                         progress=0., side=obs['side'], width=float(obs['width']),
                         exit_world=world[-1].copy(), covered_m=0.)
            if obs.get('curve') is not None:
                route['boundary_world'] = np.array([world_point(p,pose) for p in obs['curve']])
        else:
            dt = now-route['time']
            travel = float(np.linalg.norm(pose[:2]-route['pose'][:2]))
            expired = dt > 2.
            yaw_delta = abs(wrap(pose[2]-route['pose'][2]))
            self.debug.update(s_route_match_age_s=float(dt),s_route_pose_delta_m=travel,
                              s_route_yaw_delta_deg=float(np.rad2deg(yaw_delta)))
            if (dt <= 0 or (not relaxed and (travel > .2 or
                    yaw_delta > np.deg2rad(45)))):
                raise ValueError('S route odometry/time gap; reset required')
            # Allow bounded settling before STOP, then require three stable
            # CURRENT poses and stricter geometry below. Never rearm just
            # because an old pose or deadline was overwritten.
            if not relaxed and expired and (travel > .05 or yaw_delta > np.deg2rad(20)):
                raise ValueError('S route expired away from verified pose; reset required')
            if obs['side'] != route['side'] or abs(obs['width']-route['width']) > .2*route['width']:
                raise ValueError('S route boundary identity/width changed')
            old = route['path']
            stations = arc_stations(old)
            # Restrict correspondence BEFORE nearest projection, preventing a
            # shortcut to another nearby leg even when the S folds in x.
            limit = min(stations[-1], route['progress']+1.5*travel+.005)
            samples = np.linspace(route['progress'], limit, 24)
            local = np.column_stack([np.interp(samples, stations, old[:, k]) for k in (0, 1)])
            progress = float(samples[np.argmin(np.linalg.norm(local-pose[:2], axis=1))])
            remaining = stations[-1]-progress
            if remaining < .02:
                raise ValueError('S route observed end reached')
            # Match the CURRENT visible prefix to the stored path. At most
            # 6cm of near-field (<14cm forward) straight-ish stored geometry
            # may be cropped. Never advance progress to that visible endpoint
            # or jump over a hidden bend to reach a distant matching S leg.
            _, start_idx, start_fraction = nearest_on_chain(world[:1], old)
            visible_start = float(stations[start_idx[0]]+
                                  start_fraction[0]*np.diff(stations)[start_idx[0]])
            crop = max(0., visible_start-progress)
            check_start = progress
            rotation_crop = False
            if crop > .015:
                hidden_s = np.linspace(progress, visible_start, 16)
                hidden = np.column_stack([np.interp(hidden_s, stations, old[:,k]) for k in (0,1)])
                c, sn = np.cos(pose[2]), np.sin(pose[2])
                hidden_base = (hidden-pose[:2])@np.array([[c,-sn],[sn,c]])
                tangents = np.diff(hidden, axis=0)
                headings = np.unwrap(np.arctan2(tangents[:,1],tangents[:,0]))
                heading_span = float(np.rad2deg(np.ptp(headings)))
                max_x = float(np.max(hidden_base[:,0]))
                optical_crop = self._near_boundary_out_of_view(hidden,route,pose)
                self.debug.update(s_route_crop_m=float(crop),s_route_crop_max_x_m=max_x,
                                  s_route_crop_heading_deg=heading_span,
                                  s_route_optical_crop=optical_crop)
                # A longer curved prefix can leave the camera during turning.
                # Never infer visibility from the offset centre itself, and
                # never advance odometric progress to the detected endpoint.
                rotation_crop = (crop <= .25 and heading_span <= 60 and
                    self._near_boundary_out_of_view(hidden, route, pose, strict=True))
                crop_limit = .20 if relaxed else .06
                if crop > crop_limit and not rotation_crop:
                    raise ValueError(f'S route crop too long ({crop:.3f}m > {crop_limit:.3f}m)')
                if max_x > (.28 if relaxed else .14) and not optical_crop:
                    raise ValueError(f'S route missing boundary not verified near/out of view (x={max_x:.3f}m)')
                if heading_span > (90 if relaxed else 20) and not rotation_crop:
                    raise ValueError(f'S route hidden bend too sharp ({heading_span:.1f}deg > {90 if relaxed else 20}deg)')
                check_start = visible_start
            rotation_crop = rotation_crop and (crop > .06 or heading_span > 20)
            self.debug['s_route_rotation_crop'] = rotation_crop
            if rotation_crop:
                # Episode anchor is NOT refreshed by successful partial frames.
                episode = route.get('rotation_crop_episode')
                if episode is None:
                    episode = dict(time=route['time'], pose=route['pose'].copy())
                delta = pose-episode['pose']
                if not relaxed and (now-episode['time'] > 2. or np.linalg.norm(delta[:2]) > .06 or
                        abs(wrap(delta[2])) > np.deg2rad(30)):
                    raise ValueError('S route rotation visibility budget exhausted')
                route['rotation_crop_episode'] = episode
            else:
                route.pop('rotation_crop_episode', None)
            check_s = np.linspace(check_start, min(check_start+.12, stations[-1]), 24)
            check = np.column_stack([np.interp(check_s, stations, old[:, k]) for k in (0, 1)])
            q, idx, fraction = nearest_on_chain(check, world)
            current_s = arc_stations(world)
            matched_s = current_s[idx]+fraction*np.diff(current_s)[idx]
            aligned = np.sum(np.gradient(check, axis=0)*np.diff(world, axis=0)[idx], axis=1)
            error = float(np.max(np.linalg.norm(q-check, axis=1)))
            span = float(matched_s[-1]-matched_s[0])
            required = float(min(.03 if relaxed else .05, (stations[-1]-check_start)*.8))
            backstep = float(np.min(np.diff(matched_s)))
            alignment = float(np.min(aligned))
            failures = [name for name,failed in (
                ('distance_gt_2cm',error>(.04 if relaxed else .02)), ('overlap_too_short',span<required),
                ('backstep_gt_2mm',backstep<(-.01 if relaxed else -.002)), ('opposing_tangent',alignment<=0)) if failed]
            # Record both the measured error and the active profile thresholds.
            self.debug.update(s_match_error_m=error,s_match_span_m=span,
                s_match_required_span_m=required,s_match_backstep_m=backstep,
                s_match_min_alignment=alignment,s_match_failures=failures,
                s_match_distance_limit_m=.04 if relaxed else .02,
                relaxed_tracking=relaxed)
            a,b=np.gradient(check,axis=0),np.diff(world,axis=0)[idx]
            cosines=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
            self.debug['s_match_min_cosine']=float(np.min(cosines))
            soft_match=(failures==['distance_gt_2cm'] and error<=.025 and
                span>=.10 and np.min(cosines)>=.98 and not rotation_crop and
                travel<=.05 and yaw_delta<=np.deg2rad(20))
            if failures and not soft_match:
                raise ValueError('S route new path skips or contradicts pending bend')
            if soft_match:
                # 2--2.5cm is not an instant waiver: stop for three stable fresh
                # confirmations. Once admitted, every frame must still satisfy
                # the stronger overlap/direction checks against the SAME path.
                count=3 if previous_soft and dt<=self.config.max_gap_s else 1
                anchor_pose=np.array(pose)
                offsets=q-check
                if (count<3 and recovery is not None and recovery.get('mode')=='distance'
                        and 0<now-recovery['time']<=self.config.max_gap_s
                        and np.linalg.norm(pose[:2]-recovery['anchor_pose'][:2])<=.005
                        and abs(wrap(pose[2]-recovery['anchor_pose'][2]))<=np.deg2rad(3)
                        and np.max(np.linalg.norm(offsets-recovery['offsets'],axis=1))<=.004):
                    count=recovery['count']+1
                    anchor_pose=recovery['anchor_pose']
                    offsets=recovery['offsets']
                self.debug['s_distance_reacquire_count']=count
                if count<3:
                    self.s_route_reacquire=dict(mode='distance',time=now,anchor_pose=anchor_pose,
                                               count=count,offsets=offsets.copy())
                    raise ValueError(f'S route small-distance confirmation {count}/3')
                self.debug['s_distance_tolerance_m']=.025
            if check_s[-1]-check_s[0] < (.03 if relaxed else .05):
                raise ValueError('S route insufficient visible overlap')
            self.debug.update(s_route_visible_overlap_m=float(matched_s[-1]-matched_s[0]),
                              s_route_near_crop_m=float(check_start-progress))
            if not relaxed and ((expired and not soft_match) or rotation_crop):
                a,b=np.gradient(check,axis=0),np.diff(world,axis=0)[idx]
                cosines=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
                if (np.max(np.linalg.norm(q-check,axis=1)) > .01 or
                        np.min(cosines) < .95 or matched_s[-1]-matched_s[0] < .08):
                    raise ValueError('S route reacquisition requires precise 8cm overlap')
            if rotation_crop:
                # Also match the REAL observed boundary, in ordered arc space.
                # A centre path coincidence alone cannot identify an S leg.
                curve = np.asarray(obs.get('curve', []), float)
                if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 5 or
                        not np.isfinite(curve).all()):
                    raise ValueError('S route rotation needs observed boundary')
                actual = np.array([world_point(p, pose) for p in curve])
                stored = route['boundary_world']
                distance, cosine, support = (.035,.80,.03) if relaxed else (.01,.95,.08)
                ok,detail,_=self._boundary_overlap(actual,stored,distance,cosine,support)
                self.debug.update({'s_rotation_boundary_'+k:v for k,v in detail.items()})
                # 연결된 직각 꼭짓점의 수치 접선 한 점은 반대 방향일 수 있다.
                # 현재 중앙선 10cm가 1cm/0.98 이내로 대응하고 실경계의
                # 90% 이상이 대응할 때만 이 꼭짓점 이상치를 허용한다.
                junction_match = (target.get('preserve_pending_s') and
                    detail.get('ratio', 0.) >= .90 and error <= .01 and
                    span >= .10 and float(np.min(cosines)) >= .98 and
                    backstep >= -.002)
                if (not ok or detail['max_error_m']>distance or
                        (detail['min_cosine']<cosine and not junction_match)):
                    raise ValueError('S route rotation boundary mismatch')
                self.debug['s_rotation_junction_match'] = bool(junction_match)
            if expired and not soft_match and not relaxed:
                count = 1
                anchor_pose = np.array(pose)
                if (recovery is not None and recovery.get('mode')!='distance'
                        and 0 < now-recovery['time'] <= self.config.max_gap_s
                        and np.linalg.norm(pose[:2]-recovery['anchor_pose'][:2]) <= .005
                        and abs(wrap(pose[2]-recovery['anchor_pose'][2])) <= np.deg2rad(3)):
                    count = recovery['count']+1
                    anchor_pose = recovery['anchor_pose']
                if count < 3:
                    self.s_route_reacquire = dict(time=now,anchor_pose=anchor_pose,count=count)
                    self.debug['s_route_reacquire_count'] = count
                    raise ValueError(f'S route stationary reacquisition {count}/3')
                self.debug['s_route_reacquire_count'] = 3
            # Keep the whole pending bend. Extend only after a 5 cm matching
            # tail, with <=1 cm seam and aligned outgoing tangent.
            tail_s = np.linspace(max(progress, stations[-1]-.06), stations[-1], 16)
            tail = np.column_stack([np.interp(tail_s, stations, old[:, k]) for k in (0, 1)])
            tq, ti, tf = nearest_on_chain(tail, world)
            ts = current_s[ti]+tf*np.diff(current_s)[ti]
            a, b = old[-1]-old[-2], world[min(ti[-1]+1, len(world)-1)]-world[ti[-1]]
            cosine = float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
            if (tail_s[-1]-tail_s[0] >= .05 and np.max(np.linalg.norm(tq-tail, axis=1)) <= .01
                    and np.all(np.diff(ts) >= -.002) and cosine >= np.cos(np.deg2rad(20))):
                suffix = world[current_s > ts[-1]+.005]
                if len(suffix):
                    joined = np.vstack((old, suffix))
                    if first_self_intersection(joined) is None:
                        old = joined
            # Trim only travelled samples; retain one interpolation point.
            keep = stations > progress
            origin = np.array([np.interp(progress, stations, route['path'][:, k]) for k in (0, 1)])
            travelled_count = int(np.count_nonzero(~keep))
            old = np.vstack((origin, old[travelled_count:]))
            # Bounded sample count, without resampling/replacing pending points.
            old = old[:600]
            route.update(path=old, progress=0., pose=np.array(pose), time=now,
                         covered_m=route['covered_m']+progress)
        old = route['path']
        stations = arc_stations(old)
        # Euclidean gap to the first untraversed point is part of lookahead.
        gap = float(np.linalg.norm(old[0]-pose[:2]))
        target_s = min(stations[-1], max(0., lookahead-gap))
        goal = np.array([np.interp(target_s, stations, old[:, k]) for k in (0, 1)])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        local = (old-pose[:2])@rotation
        point = (goal-pose[:2])@rotation
        if point[0] <= .05:
            raise ValueError('S route target not forward')
        self._refresh_s_boundary(route,obs,pose,now)
        self.s_route = route  # Commit only after all geometry/target checks.
        self.s_route_soft_valid = soft_match
        if soft_match:
            target=dict(target,corner_speed_cap=min(.02,target.get('corner_speed_cap',.02)),
                        s_distance_reacquired=True)
        self.debug.update(s_route_remaining_m=float(stations[-1]),
                          s_route_progress_m=float(route['covered_m']),
                          s_route_target_arc_m=float(target_s),
                          s_route_mode='odom committed pending bend')
        return limit_bend_entry(dict(target, x_m=float(point[0]), y_m=float(point[1]),
                    center_path=local.tolist(), s_route_tracking=True,
                    s_route_target_arc_m=float(target_s),
                    adaptive=target_s >= stations[-1]))

    def _saved_s_target(self, obs, pose, now, lookahead, cause):
        """Brief odometry-corrected continuation, never a new observation.

        All budgets start at the last VERIFIED route. A returned target must
        not modify route time, pose, path or progress, so repeated failures
        cannot renew the allowance. Hardware/transport freshness stays outside.
        """
        route = self.s_route
        if route is None or pose is None or not np.isfinite(pose).all():
            return None
        age = now-route['time']
        travel = float(np.linalg.norm(pose[:2]-route['pose'][:2]))
        yaw = abs(wrap(pose[2]-route['pose'][2]))
        self.debug.update(s_history_age_s=float(age), s_history_travel_m=travel,
                          s_history_yaw_deg=float(np.rad2deg(yaw)))
        if not 0 < age < 2. or travel > .04 or yaw > np.deg2rad(20):
            self.debug['s_history_rejected'] = 'fixed time/motion budget exhausted'
            return None
        if obs is not None:
            if (obs['side'] != route['side'] or
                    abs(obs['width']-route['width']) > .2*route['width']):
                return None
            # Reject contradictory current boundaries, even if a caller reports
            # a crop error. A partial matching fragment need not cover the goal.
            curve = np.asarray(obs.get('curve', []), float)
            stored = route.get('boundary_world')
            if (stored is None or curve.ndim != 2 or curve.shape[1] != 2 or
                    len(curve) < 5 or not np.isfinite(curve).all()):
                return None
            actual = np.array([world_point(p, pose) for p in curve])
            ok,detail,_=self._boundary_overlap(actual,stored)
            self.debug.update({'s_history_boundary_'+k:v for k,v in detail.items()})
            if not ok:
                self.debug['s_history_rejected'] = 'current boundary contradicts saved route'
                return None
        path = route['path']
        stations = arc_stations(path)
        limit = min(stations[-1], route['progress']+1.5*travel+.005)
        samples = np.linspace(route['progress'],limit,24)
        section = np.column_stack([np.interp(samples,stations,path[:,k]) for k in (0,1)])
        progress = float(samples[np.argmin(np.linalg.norm(section-pose[:2],axis=1))])
        if stations[-1]-progress < .05:
            return None  # Never extrapolate beyond the saved observed end.
        start = np.array([np.interp(progress,stations,path[:,k]) for k in (0,1)])
        goal_s = min(stations[-1],progress+max(0.,lookahead-np.linalg.norm(start-pose[:2])))
        goal = np.array([np.interp(goal_s,stations,path[:,k]) for k in (0,1)])
        c,s=np.cos(pose[2]),np.sin(pose[2])
        rotation=np.array([[c,-s],[s,c]])
        point=(goal-pose[:2])@rotation
        if point[0] <= .05:
            return None
        local=(np.vstack((start,path[stations>progress]))-pose[:2])@rotation
        cap=.02*min(1.,2.-age)  # Taper during second second; zero at expiry.
        result=limit_bend_entry(dict(x_m=float(point[0]),y_m=float(point[1]),
            center_path=local.tolist(),width_m=route['width'],normal_width_m=route['width'],
            boundary_count=0 if obs is None else 1,inferred=True,visible_side=route['side'],
            corner_speed_cap=cap,s_route_history=True,s_history_age_s=float(age)))
        self.debug.update(mode='S_HISTORY',reason='bounded saved route: '+cause)
        self.block_recovery=True  # Do not chain generic lane-loss holds after expiry.
        return result

    def _s_bend_feature(self, obs, feature, now, pose):
        """Keep a cropped S identity only against recent odom-aligned evidence.

        Fresh full S observations anchor identity; partial frames never renew
        the two-second lifetime. Commands always use CURRENT measured geometry.
        """
        if obs is None or pose is None or not np.isfinite(pose).all():
            self.s_bend_anchor = None
            return feature
        if feature['kind'] == 'S_BEND':
            self.s_bend_anchor = dict(time=now, side=obs['side'], width=obs['width'],
                curve=np.array([world_point(p, pose) for p in obs['curve']]))
            return feature
        anchor = self.s_bend_anchor
        if anchor is None:
            return feature
        if not 0 < now-anchor['time'] <= 2.:
            self.s_bend_anchor = None
            return feature
        c, s = np.cos(pose[2]), np.sin(pose[2])
        reference = (anchor['curve']-pose[:2])@np.array([[c, -s], [s, c]])
        if (obs['side'] != anchor['side'] or
                abs(obs['width']-anchor['width']) > .2*anchor['width'] or
                not match_corner_fragment(obs['curve'], reference)['fragment_ok']):
            self.s_bend_anchor = None
            return feature
        return dict(feature, kind='S_BEND', reason='current curve matches recent S bend')

    def _s_bend_target(self, obs, ordinary, lookahead):
        """Follow an observed curved centre, never synthesize a pivot for S.

        Reuse an already validated centre whenever possible. A fallback uses
        arc-ordered normals and existing cusp/overlap checks, not y(x) refitting
        or a guessed Bezier exit. Invalid near folds still stop.
        """
        if ordinary is not None:
            return dict(ordinary, corner_speed_cap=self.config.speed_mps)
        side, width = obs['side'], float(obs['width'])
        if side not in ('left', 'right') or not np.isfinite(width) or not .04 < width <= 2.:
            raise ValueError('invalid S bend boundary')
        curve = resample_chain(obs['curve'], 100)
        other = obs.get('other')
        if other is None:
            path, note = center_path_from_boundary(curve, width*(-.5 if side == 'left' else .5))
        else:
            other = resample_chain(other, 100)
            path, _ = center_from_pair(curve, other) if side == 'left' else center_from_pair(other, curve)
            note = None
        point, adaptive = select_lookahead(path, lookahead)
        return dict(x_m=float(point[0]), y_m=float(point[1]), center_path=path.tolist(),
                    adaptive=adaptive, width_m=width, normal_width_m=width,
                    inferred=other is None, boundary_count=1 if other is None else 2,
                    visible_side=side, actual_curve=curve.tolist(),
                    width_source=obs.get('width_source', 'configured'),
                    corner_speed_cap=self.config.speed_mps, s_bend_path=True,
                    path_note=note)

    def _history_feature(self, obs, feature, now, pose):
        """Two-second vote window in odom, never replay an unobserved command.

        Require >=3 compatible observations and >=50% of recent frames, with
        at least one real CORNER anchor. Cropped, aligned measured segments
        may support that anchor, but never refresh it. Other NORMAL/UNKNOWN
        frames count against the vote. S bends, opposite turns,
        missing pose/observation and inconsistent current geometry cannot turn.
        """
        history = self.corner_history
        while history and now-history[0]['time'] > 2.:
            history.popleft()
        record = dict(time=now, side=obs['side'] if obs else None, kind=feature['kind'])
        if obs is not None and pose is not None and feature['kind'] == 'CORNER':
            record.update(feature=dict(feature), width=obs['width'],
                          curve=np.array([world_point(p, pose) for p in obs['curve']]),
                          point=world_point(feature['corner'], pose),
                          yaw=wrap(np.arctan2(feature['tout'][1],feature['tout'][0])+pose[2]),
                          pose=np.array(pose))
        history.append(record)
        if (obs is None or pose is None or feature['kind'] == 'S_BEND' or
                not self.config.staged_turn or self.confirmed_exit is not None):
            return feature, 0
        candidates = [r for r in history if 'point' in r and r['side'] == obs['side']]
        if not candidates:
            return feature, 0
        anchor = candidates[-1]
        if not 75 <= abs(anchor['feature']['angle_deg']) <= 115:
            return feature, 0
        # Never override the latest observed opposite direction with old votes.
        votes = [r for r in candidates
                 if np.sign(r['feature']['angle_deg']) == np.sign(anchor['feature']['angle_deg'])
                 and np.linalg.norm(r['point']-anchor['point']) <= self.config.landmark_gate_m
                 and abs(wrap(r['yaw']-anchor['yaw'])) <= np.deg2rad(self.config.heading_gate_deg)]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        reference = (anchor['curve']-pose[:2])@rotation
        current = np.asarray(obs['curve'])
        # Most CURRENT samples must lie on the recorded corner. Overlap-only
        # matching can otherwise mistake a straight line extending past the
        # old bend for a partial view of that bend.
        distances = np.linalg.norm(current[:, None, :]-reference[None, :, :], axis=2).min(axis=1)
        if (curve_match_error(np.asarray(obs['curve']), reference) > .025 or
                np.quantile(distances, .9) > .035 or
                abs(obs['width']-anchor['width']) > .2*anchor['width']):
            return feature, 0
        # Distinguish a cropped view of a known corner from evidence that the
        # road really became straight. Partial matches support persistence of
        # ONE real corner observation; they never create or refresh that anchor.
        # Require a substantial measured segment, less coverage than the full
        # corner, and a tangent consistent with one of its measured legs.
        if feature['kind'] in ('NORMAL', 'UNKNOWN') and len(current) >= 5:
            current_length = arc_stations(current)[-1]
            reference_length = arc_stations(reference)[-1]
            _, tangent, residual = _line(current)
            delta = anchor['pose'][2]-pose[2]
            turn = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
            aligned = max(float(tangent@(turn@anchor['feature'][key]))
                          for key in ('tin', 'tout')) >= np.cos(np.deg2rad(20))
            if (self.config.segment_m <= current_length < .85*reference_length
                    and residual <= .008 and aligned):
                record['partial_anchor'] = anchor['time']
                record['kind'] = 'PARTIAL_CORNER'
        partials = [r for r in history if r.get('partial_anchor') == anchor['time']]
        evidence = len(votes)+len(partials)
        if evidence < self.config.confirm_frames or evidence/len(history) < .5:
            if record['kind'] == 'PARTIAL_CORNER':
                return dict(kind='PARTIAL_CORNER', angle_deg=anchor['feature']['angle_deg'],
                            reason='matched partial corner; awaiting evidence'), 0
            return feature, 0
        if feature['kind'] == 'CORNER':
            return feature, evidence
        # A partial current boundary may preserve a well-supported corner, but
        # only after matching its odometry-transformed real boundary above.
        recovered = dict(anchor['feature'], reason='odometry-aligned corner history')
        # Preserve the actual measured full corner, not only today's cropped
        # entry fragment, so later exit-only views can still be associated.
        recovered['observed_reference'] = reference.copy()
        old_pose = anchor['pose']
        for key in ('entry', 'exit', 'corner'):
            recovered[key] = (world_point(recovered[key], old_pose)-pose[:2])@rotation
        delta = old_pose[2]-pose[2]
        turn = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
        for key in ('tin', 'tout'):
            recovered[key] = turn@recovered[key]
        return recovered, evidence

    def update(self, observation, ordinary, now, pose, lookahead, max_angular,
               no_boundaries=False):
        """Keep measured map evidence independent of committed S-route shape."""
        local_map = self.local_lane_map
        fresh = local_map.prepare(now, pose)
        target = self._update(observation, ordinary, now, pose, lookahead,
                              max_angular, no_boundaries)
        # Only the route-shape conflict can use this alternative. Current
        # geometry must already be valid; no empty-lane or staged-turn bypass.
        reason = self.debug.get('reason', '')
        map_eligible = reason == 'S bend: S route new path skips or contradicts pending bend'
        if self.config.relaxed_tracking:
            # Current measured geometry may supersede an expired committed S;
            # never authorize driving from an empty or held observation.
            map_eligible = map_eligible or reason.startswith(('S bend: S route expired',
                'S bend: S route crop', 'S bend: S route missing boundary',
                'S bend: S route hidden bend', 'S bend: S route reacquisition',
                'S bend: S route rotation', 'S bend: S route insufficient visible'))
        if (fresh and target is None and ordinary is not None and observation
                and self.staged is None and self.confirmed_exit is None
                and self.s_route is not None
                and observation.get('side') == self.s_route.get('side')
                and map_eligible):
            target = local_map.current_target(observation, ordinary, now, pose, lookahead,
                                             relaxed=self.config.relaxed_tracking)
            if target is not None:
                self.block_recovery = True  # No extra blind hold after this target.
                self.debug.update(mode='LOCAL_MAP', reason='fresh near lane confirmed by odom map',
                                  local_map_votes=target['local_map_votes'])
                if self.config.relaxed_tracking and not ordinary.get('preserve_pending_s'):
                    self.s_route = None
                    self.s_route_reacquire = None
                    self.s_route_soft_valid = False
                    self.debug['s_route_replaced_by_current_map'] = True
                elif ordinary.get('preserve_pending_s'):
                    self.debug['s_route_preserved_by_current_map'] = True
        if fresh and ordinary is not None and observation:
            local_map.remember(observation, now, pose)
        self.debug.update(local_map_frames=len(local_map.frames), local_map_retention_s=10.)
        return target

    def _update(self, observation, ordinary, now, pose, lookahead, max_angular,
                no_boundaries=False):
        cfg = self.config
        if self.last_time is not None and now <= self.last_time:
            self.debug.update(reason='duplicate or out-of-order observation')
            return None
        if self.last_time is not None and now-self.last_time > cfg.max_gap_s:
            self.s_route_soft_valid = False
            self.s_bend_anchor = None
            self.previous = None
            self.count = 0
            self.exit_count = 0
            if self.staged is not None:
                self.staged['exit_count'] = 0
        self.last_time = now
        if self.staged is not None:
            return self._staged_update(observation, ordinary, now, pose, lookahead,
                                       no_boundaries)
        if observation is None:
            self.s_route_soft_valid = False
            self.s_bend_anchor = None
            self.s_route_reacquire = None
            self._history_feature(None, dict(kind='UNKNOWN'), now, pose)
            active = self.s_route is not None or self.block_recovery or self.state not in ('NORMAL', 'S_BEND')
            self.previous = None
            self.count = 0
            self.exit_count = 0
            self.block_recovery = active
            self.debug = dict(mode='UNKNOWN', reason='no fresh trusted boundary')
            if no_boundaries:
                saved = self._saved_s_target(None,pose,now,lookahead,'empty current detection')
                if saved is not None:
                    return saved
            return None if active else ordinary
        ordinary = limit_bend_entry(ordinary)
        feature = classify_boundary(observation['curve'], cfg)
        if not observation.get('sharp_s_prefix'):
            feature = self._s_bend_feature(observation, feature, now, pose)
        feature, history_votes = self._history_feature(observation, feature, now, pose)
        self.debug = dict(mode=feature['kind'], angle_deg=feature['angle_deg'], reason=feature['reason'],
                          history_votes=history_votes, history_frames=len(self.corner_history))
        preserve_unknown = (ordinary is not None and ordinary.get('preserve_pending_s') and
                            feature['kind'] == 'UNKNOWN')
        if (self.s_route is not None and not observation.get('sharp_s_prefix') and
                (feature['kind'] == 'CORNER' or preserve_unknown)):
            # A cropped S shows only ONE bend, often classified CORNER. The
            # label alone is not a contradiction. Route it through the same
            # current-centre/odom correspondence checks as a full S. Those
            # checks still reject a genuinely different turn or distant leg.
            feature = dict(feature, kind='S_BEND')
            self.debug.update(mode='S_BEND',
                              s_route_observation='partial bend; validating existing route')
        if feature['kind'] != 'CORNER':
            if self.confirmed_exit is not None:
                allowed = feature['kind'] in ('NORMAL', 'S_BEND') or (
                    cfg.relaxed_tracking and feature['kind'] == 'UNKNOWN' and
                    ordinary is not None and not ordinary.get('held', False))
                if pose is None or ordinary is None or not allowed:
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
            if (feature['kind'] == 'UNKNOWN' and cfg.relaxed_tracking
                    and ordinary is not None and not ordinary.get('held')
                    and observation.get('side') in ('left', 'right')):
                # Classification failure is not perception failure. A freshly
                # validated single-line offset/pair centre still drives PP.
                # Do not substitute a stale S leg or a synthetic blind turn.
                path = np.asarray(ordinary.get('center_path', []), float)
                if (path.ndim == 2 and path.shape[1] == 2 and len(path) >= 5
                        and np.isfinite(path).all()
                        and np.isfinite([ordinary.get('x_m', np.nan),
                                         ordinary.get('y_m', np.nan)]).all()
                        and ordinary['x_m'] > 0
                        and first_self_intersection(path) is None):
                    self.s_route = None
                    self.s_route_reacquire = None
                    self.s_route_soft_valid = False
                    self.block_recovery = False
                    self.debug.update(classification_reason=feature['reason'],
                        reason='UNKNOWN classification; following current measured centre',
                        unknown_following=True)
                    return dict(ordinary, unknown_following=True,
                                corner_speed_cap=min(cfg.speed_mps,
                                    ordinary.get('corner_speed_cap', cfg.speed_mps)))
            if feature['kind'] == 'S_BEND' or (self.s_route is not None and feature['kind'] == 'NORMAL'):
                try:
                    target = self._s_bend_target(observation, ordinary, lookahead)
                    target = self._track_s_route(target, observation, pose, now, lookahead)
                    if (feature['kind'] == 'NORMAL' and
                            np.linalg.norm(pose[:2]-self.s_route['exit_world']) <= .05):
                        self.s_route = None  # Physically reached the committed S exit.
                except ValueError as exc:
                    self.block_recovery = True
                    self.debug.update(reason='S bend: '+str(exc))
                    if str(exc).startswith(('S route crop too long',
                            'S route missing boundary not verified',
                            'S route hidden bend too sharp',
                            'S route insufficient visible overlap')):
                        saved=self._saved_s_target(observation,pose,now,lookahead,str(exc))
                        if saved is not None:
                            return saved
                    return None
                self.debug.update(reason='observed S bend centre path')
                return target
            if feature['kind']=='UNKNOWN' and feature['reason']=='insufficient observed legs':
                self.s_route_soft_valid = False
                self.s_route_reacquire = None
                saved=self._saved_s_target(observation,pose,now,lookahead,feature['reason'])
                if saved is not None:
                    return saved
            return None if self.block_recovery else ordinary
        self.block_recovery = True
        corner_distance = float(np.linalg.norm(feature['corner']))
        # The ordinary centre path remains authoritative while it is valid
        # and the robot front is still beyond the controller's lookahead from
        # the corner. A candidate alone must not replace that path with STOP.
        # A sub-75-degree curve with a valid observed centre is NOT a
        # stationary corner. Do not discard it just for entering a radius.
        # No centre-path metadata means legacy/test targets keep old gating.
        gentle_centre = (abs(feature['angle_deg']) < 75 and ordinary is not None
                          and not ordinary.get('held') and
                          len(ordinary.get('center_path', [])) >= 5)
        centre_can_approach = (ordinary is not None and
                               (corner_distance > lookahead+cfg.front_m or gentle_centre) and
                               self.confirmed_exit is None)
        self.debug.update(direction='LEFT' if feature['angle_deg'] > 0 else 'RIGHT',
                          corner_distance_m=corner_distance,
                          boundary_side=observation['side'],
                          lane_width_m=float(observation['width']),
                          width_source=observation.get('width_source', 'unknown'))
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
        self.count = max(self.count, history_votes)
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
        if cfg.staged_turn and 75 <= abs(feature['angle_deg']) <= 115:
            if centre_can_approach and len(ordinary.get('center_path', [])) >= 5:
                # 실제 관측 중앙 경로가 유효하고 코너가 아직 먼 경우에는
                # 진입 피벗을 확정하지 않는다. 조기 확정하면 긴 접근 동안
                # 카메라 크롭/새 출구 길이로 저장 경계의 대응을 먼저 잃는다.
                self.block_recovery = False
                self.debug.update(reason='ordinary centre path approaches distant confirmed corner')
                return dict(ordinary, corner_speed_cap=cfg.speed_mps)
            try:
                self._start_staged(observation, feature, pose, now)
            except ValueError as exc:
                self.debug.update(reason=str(exc))
                if centre_can_approach:
                    self.block_recovery = False
                    self.debug.update(reason='ordinary centre path remains valid; '+str(exc))
                    return dict(ordinary, corner_speed_cap=cfg.speed_mps)
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

    def staged_observation(self, masks, calibration, pose):
        """확정된 코너의 CURRENT 경계만 대응한다. PP 경로 생성과 분리한다.

        진입 끝에서 PP 최소 x 밖으로 잘린 선도 고정된 odom 코너에 대응하면
        관측으로 유효하다. 옛 마스크/중앙 경로를 발행하거나 새 코너를 만들지
        않는다. 대응하는 현재 마스크가 정확히 하나일 때만 반환한다.
        """
        from .metric_lane import floor_curves, connected_floor_curve
        s = self.staged
        if s is None or pose is None:
            return None
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        reference = (s['curve']-pose[:2])@np.array([[c, -sn], [sn, c]])
        candidates = []
        opposite_candidates = []
        opposite = s.get('opposite_curve')
        opposite_reference = None if opposite is None else (
            opposite-pose[:2])@np.array([[c, -sn], [sn, c]])
        for mask_index, mask in enumerate(masks):
            curves = connected_floor_curve(mask, calibration, 2.)
            if not curves:
                curves = floor_curves([mask], calibration, 2., .05, 2.,
                                      connected_geometry=True)
            accepted = []
            for curve in curves:
                if s.get('sharp_s_prefix'):
                    prefix=first_sharp_corner_prefix(curve,self.config)
                    if prefix is not None:
                        curve=prefix
                # 현재 새로 보인 출구의 길이가 저장 당시보다 길다는 이유만으로
                # 같은 코너의 관측을 실패시키지 않는다. 목표/회전각은 고정이다.
                current = trim_corner_endpoint_extensions(curve, reference)
                if match_corner_fragment(current, reference)['fragment_ok']:
                    accepted.append(current)
            if len(accepted) == 1:
                candidates.append((mask_index, accepted[0]))
            if opposite_reference is not None:
                other = [trim_corner_endpoint_extensions(curve, opposite_reference) for curve in curves]
                other = [curve for curve in other
                         if match_corner_fragment(curve, opposite_reference)['fragment_ok']]
                if len(other) == 1:
                    opposite_candidates.append((mask_index, other[0]))
        if len(candidates) == 1:
            return dict(curve=candidates[0][1], side=s['side'], width=s['width'],
                        source_mask_index=candidates[0][0],
                        width_source='committed corner')
        if not candidates and len(opposite_candidates) == 1:
            return dict(curve=opposite_candidates[0][1], source_mask_index=opposite_candidates[0][0],
                        side='right' if s['side']=='left' else 'left', width=s['width'],
                        width_source='committed corner', staged_opposite_validated=True)
        return None

    def _mapped_entry_support(self, obs, feature, pose, now):
        """동일 진입선의 과거 실관측만 odom 변환해 현재 잘린 입구를 보완한다.

        10초 지도 내 두 독립 관측이 현재 입구와 8mm/10도 이내로 대응해야 한다.
        출구·코너·회전각은 현재 관측 그대로이며 보이지 않은 선을 외삽하지 않는다.
        """
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        inverse = np.array([[c, -sn], [sn, c]])
        entry, tin = feature['entry'], feature['tin']
        normal = np.array([-tin[1], tin[0]])
        votes = []
        for timestamp, curves, width in self.local_lane_map.frames:
            if not 0 < now-timestamp <= self.local_lane_map.retention_s or obs['side'] not in curves or abs(width-obs['width']) > .02:
                continue
            reference = (curves[obs['side']]-pose[:2])@inverse
            rel = reference-entry
            station = rel@tin
            valid = (np.abs(rel@normal) <= .008) & (station <= .06)
            indices = np.flatnonzero(valid)
            if len(indices) < 5:
                continue
            runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)
            for run in runs:
                p = reference[run]
                st = station[run]
                if len(p) < 5 or st.min() >= -.01 or st.max() < .025:
                    continue
                tangent = p[-1]-p[0]
                if (np.linalg.norm(tangent) < .04 or
                        tangent@tin/np.linalg.norm(tangent) < np.cos(np.deg2rad(10))):
                    continue
                votes.append((timestamp, p, float(st.min())))
                break
        if len(votes) < 2:
            return feature
        # Both observations must independently support the extended start.
        start = sorted(v[2] for v in votes)[1]
        reference = min(votes, key=lambda v: abs(v[2]-start))[1]
        current = np.asarray(obs['curve'])
        current_start = float((current[0]-entry)@tin)
        prefix = reference[(reference-entry)@tin < min(0., current_start)-1e-6]
        joined = np.vstack((prefix, current)) if len(prefix) else current.copy()
        if first_self_intersection(joined) is not None:
            return feature
        result = dict(feature, entry=entry+start*tin,
                      observed_reference=joined,
                      mapped_entry_support_m=-start)
        self.debug['mapped_entry_support_m'] = -start
        return result

    def _start_staged(self, obs, feature, pose, now):
        """Intersect offset entry/exit lines to locate an axle-centre pivot.

        Only the measured entry span is used for approach. The stationary spin
        is NOT a footprint-safe Bezier plan or an obstacle-clearance guarantee.
        """
        feature = self._mapped_entry_support(obs, feature, pose, now)
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
        self.debug.update(entry_travel_m=float(travel), entry_support_m=support,
                          pivot_base=(a+travel*tin).tolist())
        # 검은/흰 경계의 지지 구간과 오프셋 중앙선의 지지 구간은 다르다.
        # 안쪽 L에서는 miter 교점이 경계 꼭짓점보다 d*tan(theta/2)만큼
        # 앞으로 나온다. 이를 모두 외삽으로 취급하면 폭 20cm의 정상적인
        # 90도 코너가 정확히 10cm 초과했다는 이유로 항상 거부된다.
        cross = float(tin[0]*tout[1]-tin[1]*tout[0])
        denominator = 1.+float(tin@tout)
        extension = max(0., -offset*cross/max(denominator, 1e-12))
        exit_support = float((feature['exit']-feature['corner'])@tout)
        if extension > 0 and (denominator < .25 or
                exit_support < extension+self.config.segment_m):
            raise ValueError('corner entry: insufficient measured exit for miter support')
        centre_support = support+extension
        self.debug.update(entry_boundary_support_m=support,
                          entry_miter_extension_m=extension,
                          entry_center_support_m=centre_support)
        # Search up to 2cm along/across the entry centreline,
        # scoring the complete stationary rotation rather than just the axle.
        # Small lateral shifts are approached by the existing pursuit control,
        # never by changing the stored pivot after entry has started.
        original_travel = float(travel)
        candidates = []
        measured = np.asarray(feature.get('observed_reference', obs['curve']))
        rel = measured-feature['entry']
        station = rel@tin
        on_entry = (np.abs(rel@normal(tin)) <= .005) & (station <= support)
        measured_start = min(0., float(np.min(station[on_entry]))) if np.any(on_entry) else 0.
        low, high = max(measured_start, travel-.02), min(centre_support, travel+.02)
        if low <= high:
            heading = np.arctan2(tin[1], tin[0])
            delta_heading = wrap(np.arctan2(tout[1], tout[0])-heading)
            angles = heading+np.linspace(0., delta_heading, 31)
            cfg = self.config
            footprint = np.array([[x, y]
                for x in (-cfg.rear_m, 0., cfg.front_m)
                for y in (-cfg.half_width_m, 0., cfg.half_width_m)])
            cs, sn = np.cos(angles), np.sin(angles)
            rotated = np.stack((cs[:, None]*footprint[:, 0]-sn[:, None]*footprint[:, 1],
                                sn[:, None]*footprint[:, 0]+cs[:, None]*footprint[:, 1]), axis=2)
            bounds = [(obs['side'], np.asarray(feature.get('observed_reference', obs['curve'])))]
            if obs.get('other') is not None:
                bounds.append(('left' if obs['side'] == 'right' else 'right', np.asarray(obs['other'])))
            grid = np.unique(np.r_[np.linspace(low, high, 11), np.clip(travel, low, high)])
            for candidate, lateral_shift in [(v, d) for v in grid for d in (-.02, -.01, 0., .01, .02)]:
                point = a+candidate*tin+lateral_shift*normal(tin)
                query = (rotated+point).reshape(-1, 2)
                margin = float('inf')
                for side, bound in bounds:
                    q, idx, frac = nearest_on_chain(query, bound)
                    tangent = np.diff(bound, axis=0)[idx]
                    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
                    inward = (tangent[:, 0]*(query-q)[:, 1]-tangent[:, 1]*(query-q)[:, 0])
                    inward *= -1 if side == 'left' else 1
                    # Unobserved endpoint extension cannot certify a new pivot.
                    if np.any(idx+frac <= .001) or np.any(idx+frac >= len(bound)-1.001):
                        margin = -float('inf')
                        break
                    inward_sign = -1 if side == 'left' else 1
                    turn_cross = tin[0]*tout[1]-tin[1]*tout[0]
                    if side == obs['side'] and turn_cross*inward_sign > 0.:
                        # Near a right-angle OUTER boundary the lane is the
                        # union of two corridors, not a constant-distance
                        # rounded tube. Clipping distance to width incorrectly
                        # removes the inside square of a painted L junction.
                        delta = query-feature['corner']
                        d1 = inward_sign*(tin[0]*delta[:, 1]-tin[1]*delta[:, 0])
                        d2 = inward_sign*(tout[0]*delta[:, 1]-tout[1]*delta[:, 0])
                        n1, n2 = inward_sign*normal(tin), inward_sign*normal(tout)
                        inner_corner = np.linalg.solve(np.vstack((n1, n2)), np.full(2, width))
                        inner = np.linalg.norm(delta-inner_corner, axis=1)
                        # Distance to the excluded inner wedge (also valid at
                        # 85 degrees): project onto each ray only within it.
                        proj1 = delta+(width-d1)[:, None]*n1
                        proj2 = delta+(width-d2)[:, None]*n2
                        inner = np.minimum(inner, np.where(proj1@n2 >= width, abs(width-d1), np.inf))
                        inner = np.minimum(inner, np.where(proj2@n1 >= width, abs(width-d2), np.inf))
                        outside = (d1 > width) & (d2 > width)
                        inner[outside] = -np.minimum(d1[outside]-width, d2[outside]-width)
                        clearance = np.minimum(np.minimum(d1, d2), inner)
                    else:
                        clearance = np.minimum(inward, width-inward)
                    margin = min(margin, float(np.min(clearance)))
                candidates.append((margin-cfg.clearance_m,
                                   -float(np.hypot(candidate-travel, lateral_shift)),
                                   float(candidate), lateral_shift))
        best = max(candidates) if candidates else None
        original = next((c for c in candidates if abs(c[2]-original_travel) < 1e-9 and c[3] == 0.), None)
        # Preserve a supported legacy goal if no demonstrably interior new
        # candidate exists. Only the final support gate has a 10mm tolerance;
        # the candidate search and footprint scoring remain unchanged.
        selected_interior = best is not None and best[0] >= .004 and (original is None or original[0] < .01)
        lateral_shift = 0.
        if selected_interior:
            travel = best[2]
            lateral_shift = best[3]
        pivot = a+travel*tin+lateral_shift*normal(tin)
        self.debug.update(pivot_original_travel_m=original_travel,
                          pivot_shift_m=float(travel-original_travel),
                          pivot_lateral_shift_m=lateral_shift,
                          pivot_measured_entry_start_m=measured_start,
                          pivot_rotation_margin_m=(best[0] if best and np.isfinite(best[0]) else None),
                          entry_travel_m=float(travel), pivot_base=pivot.tolist())
        # travel is measured from the first VISIBLE entry sample, not from
        # the robot. A pivot 3 mm into that span can still be 20 cm ahead of
        # the axle. Requiring 5 mm here wrongly rejects supported pivots.
        # Permit at most 10mm of endpoint fitting/calibration discrepancy.
        # This is not extra blind travel or a relaxation of footprint/drift
        # checks, nor a claim of physically verified calibration accuracy.
        support_start = measured_start if selected_interior else 0.
        support_overrun = max(0., support_start-float(travel), float(travel)-centre_support)
        self.debug.update(entry_support_tolerance_m=.010,
                          entry_support_overrun_m=support_overrun)
        if support_overrun > .010+1e-12:
            raise ValueError('corner entry: pivot outside observed entry support '
                             f'(travel={travel:.3f}m, support={centre_support:.3f}m)')
        if pivot[0] <= 0 or abs(np.arctan2(tin[1], tin[0])) > np.deg2rad(30):
            raise ValueError('corner entry: robot not aligned with entry')
        self.staged = dict(pivot=world_point(pivot, pose),
            pivot_selection={k: v for k, v in self.debug.items() if k.startswith('pivot_')},
            curve=np.array([world_point(p, pose) for p in
                            feature.get('observed_reference', obs['curve'])]),
            side=obs['side'], width=width, started=now, brake_at=None, matched_at=now,
            sharp_s_prefix=bool(obs.get('sharp_s_prefix')),
            matched_pose=np.asarray(pose).copy(), blind_entry=None, blind_eligible=True,
            entry_yaw=wrap(np.arctan2(tin[1],tin[0])+pose[2]),
            progress_at=now, progress_remaining=float(pivot@tin), fault=None,
            spin_at=None, exit_count=0,
            exit_yaw=wrap(np.arctan2(tout[1], tout[0])+pose[2]))
        if obs.get('sharp_s_prefix'):
            # 동일 현재 첫 직각을 3회 확정한 뒤에만 연속 S 경로를 교체한다.
            # 단일 후보나 크롭 프레임만으로 미완료 경로를 지우지 않는다.
            self.s_route=None
            self.s_route_reacquire=None
            self.s_route_soft_valid=False
            self.s_bend_anchor=None
        # 안쪽 선은 진입 피벗에 도착하기 전에 카메라 뒤로 사라진다.
        # 고정 목표는 그대로 두고, 현재 보이는 반대쪽 실경계로 관측만
        # 이어받을 수 있도록 같은 폭의 miter 경계를 대응 기준으로 보관한다.
        other = obs.get('other')
        source = 'measured'
        # 양쪽이 검출됐어도 반대 선은 아직 직선 진입부만 보일 수 있다.
        # 그 짧은 직선만 저장하면 뒤에 드러난 실제 출구를 대응할 수 없다.
        # 현재 두 다리로 확정한 코너 + 폭의 예상 경계는 관측 대응용이다.
        # 이동 목표는 바꾸지 않으며 새로운 실제 픽셀이 맞아야 이어받는다.
        exit_supported = False
        if other is not None and len(other) >= 5:
            delta = np.diff(np.asarray(other), axis=0)
            length = np.linalg.norm(delta, axis=1)
            aligned = (delta@tout)/np.maximum(length, 1e-12) >= np.cos(np.deg2rad(15))
            exit_supported = np.sum(length[aligned]) >= cfg.segment_m
        if not exit_supported:
            try:
                from .connected_path import connected_miter_center
                _, other = connected_miter_center(
                    np.asarray(feature.get('observed_reference', obs['curve'])),
                    width*(-1 if obs['side']=='left' else 1))
                source = 'inferred_width'
            except ValueError:
                pass  # 폭 경계 생성 실패 시 기존 실제 반대 선 관측은 보존한다.
        self.staged['opposite_curve'] = None if other is None else np.array([
            world_point(point, pose) for point in other])
        self.staged['opposite_source'] = source

    def observe_arrival(self, pose, now):
        """Latch arrival using fresh control-rate odometry; never command motion.

        Caller enforces odom freshness, even while perception is stopped. This
        records position only: the controller must still enforce ALL safety
        checks before motion. A real boundary match within two seconds is
        required; synthetic/missing frames never renew that evidence.
        """
        s, tol = self.staged, self.config.pivot_tolerance_m
        gap_budget = s.get('blind_entry') if s is not None else None
        if (s is None or s.get('fault') or s['brake_at'] is not None or
                pose is None or not np.isfinite(pose).all() or
                not s.get('blind_eligible', True) or
                not 0 <= now-s.get('matched_at', -float('inf')) < 2. or
                abs(wrap(pose[2]-s['entry_yaw'])) > np.deg2rad(15) or
                now >= s['started']+20.):
            return
        if gap_budget is not None and (gap_budget['distance']+
                np.linalg.norm(pose[:2]-gap_budget['last_pose'])) >= .03:
            return
        axis = np.array([np.cos(s['entry_yaw']), np.sin(s['entry_yaw'])])
        delta = s['pivot']-pose[:2]
        if (abs(float(delta@axis)) <= tol and
                abs(float(delta@np.array([-axis[1], axis[0]]))) <= tol):
            s['brake_at'] = now

    def _staged_update(self, obs, ordinary, now, pose, lookahead, no_boundaries=False):
        s, cfg = self.staged, self.config
        self.block_recovery = True  # Never fall through to blind lane-search spin.
        self.debug = dict(mode='ENTRY', reason='approaching observed pivot')
        self.debug['pivot_world'] = s['pivot'].tolist()
        self.debug['opposite_reference_source'] = s.get('opposite_source', 'none')
        self.debug.update(s.get('pivot_selection', {}))
        if s.get('fault'):
            self.debug.update(mode='ENTRY_FAULT', reason=s['fault'])
            return None
        blind = obs is None and no_boundaries and s.get('blind_eligible', False)
        self.debug['arrival_latched'] = s['brake_at'] is not None
        # ENTRY spends only the last measured frame's 2-second budget.
        # Once arrival is latched, BRAKE/TURN use one independent, fixed
        # deadline. Neither empty frames nor fresh matches restart that clock.
        turn_deadline = (s['brake_at']+cfg.brake_seconds+cfg.spin_timeout_s
                         if s['brake_at'] is not None else None)
        if blind and turn_deadline is None and not 0 <= now-s.get('matched_at', -float('inf')) < 2.:
            self.debug.update(mode='ENTRY_WAIT', reason='entry reacquisition deadline reached')
            return None
        if turn_deadline is not None and now >= turn_deadline and not s.get('heading_reached'):
            self.debug.update(mode='TURN_TIMEOUT', reason='spin time limit; reset required')
            return None
        # Once arrival is committed, perception is only an EXIT observation.
        # A single unconfirmed/wrong-side/new line must not cancel the turn.
        if s['brake_at'] is not None:
            if pose is None or not np.isfinite(pose).all():
                self.debug.update(mode='TURN_WAIT', reason='fresh odometry required')
                return None
            return self._stationary_update(ordinary, now, pose)
        opposite = (obs is not None and obs.get('staged_opposite_validated', False)
                    and obs['side'] != s['side'] and s.get('opposite_curve') is not None)
        if pose is None or (not blind and (obs is None or (obs['side'] != s['side'] and not opposite))):
            s['blind_eligible'] = False
            self.debug.update(mode='ENTRY_WAIT', reason='fresh same-side boundary and odometry required')
            return None
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        inverse = np.array([[c, sn], [-sn, c]])
        pivot = inverse@(s['pivot']-pose[:2])
        axis = np.array([np.cos(s['entry_yaw']), np.sin(s['entry_yaw'])])
        delta = s['pivot']-pose[:2]
        along = float(delta@axis)
        lateral = float(delta@np.array([-axis[1],axis[0]]))
        self.debug.update(remaining_forward_m=along, lateral_error_m=lateral,
                          robot_world=pose[:2].tolist())
        if blind and s['brake_at'] is None:
            # Only bridge a near-pivot camera blind spot, never search ahead
            # using an arbitrary old goal. Budget belongs to the last REAL
            # matched frame and is shared with control-rate commands.
            if s.get('blind_entry') is None:
                if (now-s['matched_at'] > .8 or not 0 <= along <= cfg.pivot_tolerance_m+.03
                        or abs(lateral) > cfg.pivot_tolerance_m
                        or abs(wrap(pose[2]-s['entry_yaw'])) > np.deg2rad(15)):
                    s['blind_eligible'] = False
                    self.debug.update(mode='ENTRY_WAIT', reason='blind entry requires recent near aligned pivot')
                    return None
                s['blind_entry'] = dict(last_pose=s['matched_pose'][:2].copy(), distance=0.)
            if (s['blind_entry']['distance']+
                    np.linalg.norm(pose[:2]-s['blind_entry']['last_pose'])) >= .03:
                self.debug.update(mode='ENTRY_WAIT', reason='blind entry 3cm limit reached')
                return None
        reference = ((s['opposite_curve'] if opposite else s['curve'])-pose[:2])@inverse.T
        if opposite:
            self.debug['entry_boundary_handover'] = s['opposite_source']
        # A fresh line must agree with the odometry-transformed observation;
        # movement never continues merely because a historic pivot exists.
        match = (dict(fragment_ok=True, fragment_reason='bounded odometry continuation')
                 if blind else match_corner_fragment(obs['curve'], reference))
        self.debug.update(match)
        if not match['fragment_ok']:
            s['blind_eligible'] = False
            self.debug.update(mode='ENTRY_WAIT', reason='boundary no longer matches stored corner')
            return None
        if not blind:
            s['matched_at'] = now
            s['matched_pose'] = np.asarray(pose).copy()
            s['blind_entry'], s['blind_eligible'] = None, True
        target = dict(x_m=float(pivot[0]), y_m=float(pivot[1]), inferred=True,
            boundary_count=0 if blind else 1, visible_side=s['side'] if blind else obs['side'],
            actual_curve=[] if blind else obs['curve'].tolist(),
            width_m=s['width'], normal_width_m=s['width'], corner_speed_cap=cfg.entry_speed_mps,
            corner_staged=True, adaptive=True, corner_pivot_world=s['pivot'].tolist(),
            corner_entry_yaw=s['entry_yaw'],
            corner_entry_deadline=s['started']+20.)
        if blind:
            target.update(corner_blind_deadline=s['matched_at']+2., corner_angular_cap=.15)
            if s.get('blind_entry') is not None:
                target['corner_blind_entry'] = s['blind_entry']
            self.debug['last_match_age_s'] = now-s['matched_at']
        elif 'source_mask_index' in obs:
            target['source_mask_index'] = obs['source_mask_index']
        if s['brake_at'] is None:
            if now-s['started'] > 20.:
                self.debug.update(mode='ENTRY_TIMEOUT', reason='entry time limit; reset required')
                return None
            tolerance = cfg.pivot_tolerance_m
            if along < -tolerance or abs(lateral) > 2*tolerance:
                s['fault'] = 'entry overshoot or lateral deviation; reset required'
                self.debug.update(mode='ENTRY_FAULT', reason=s['fault'])
                return None
            if along <= tolerance and abs(lateral) <= tolerance:
                s['brake_at'] = now
            elif along <= tolerance:
                s['fault'] = 'entry plane reached outside lateral tolerance; reset required'
                self.debug.update(mode='ENTRY_FAULT', reason=s['fault'])
                return None
            elif pivot[0] <= 0:
                self.debug.update(mode='ENTRY_WAIT', reason='pivot passed; no reverse correction')
                return None
            else:
                if s['progress_remaining']-along >= .005:
                    s['progress_remaining'], s['progress_at'] = along, now
                if now-s['progress_at'] >= 3.:
                    s['fault'] = 'entry made less than 5mm progress in 3s; reset required'
                    self.debug.update(mode='ENTRY_FAULT', reason=s['fault'])
                    return None
                target['corner_progress_deadline'] = s['progress_at']+3.
                self.debug.update(remaining_m=float(np.linalg.norm(pivot)))
                target['corner_speed_cap'] = min(cfg.entry_speed_mps, .5*float(np.linalg.norm(pivot)))
                if blind:
                    target['corner_speed_cap'] = min(target['corner_speed_cap'], .015)
                    self.debug.update(mode='ENTRY_ODOM', reason='short near-pivot camera gap',
                                      blind_distance_m=s['blind_entry']['distance'])
                return target
        return self._stationary_update(ordinary, now, pose)

    def _stationary_update(self, ordinary, now, pose):
        """Fixed odom turn first; fresh near-path confirmation only releases EXIT.

        Deliberately not a boundary match: perception cannot move the pivot,
        change its exit yaw or reset the turn deadline. Sensors/drift still gate
        commands on the robot. No speculative forward motion is returned.
        """
        s, cfg = self.staged, self.config
        c, sn = np.cos(pose[2]), np.sin(pose[2])
        pivot = np.array([[c, sn], [-sn, c]])@(s['pivot']-pose[:2])
        target = dict(x_m=float(pivot[0]), y_m=float(pivot[1]), inferred=True,
            boundary_count=0, visible_side=None, source_mask_index=-1, actual_curve=[],
            width_m=s['width'], normal_width_m=s['width'], corner_speed_cap=0.,
            corner_angular_cap=.15, corner_staged=True, adaptive=True,
            corner_pivot_world=s['pivot'].tolist(), corner_entry_yaw=s['entry_yaw'],
            corner_entry_deadline=s['started']+20.)
        self.debug.update(mode='BRAKE', reason='stop before stationary turn', arrival_latched=True)
        target.update(corner_stationary=True, corner_spin_yaw=s['exit_yaw'],
                      corner_spin_deadline=s['brake_at']+cfg.brake_seconds+cfg.spin_timeout_s,
                      corner_brake_until=s['brake_at']+cfg.brake_seconds)
        # Do not carry the ENTRY deadline into the stationary turn. Keep the
        # conservative angular cap, sensor freshness and pivot drift checks.
        target.pop('corner_blind_deadline', None)
        target.pop('corner_blind_entry', None)
        self.debug.update(turn_remaining_s=max(0., target['corner_spin_deadline']-now),
                          turn_deadline=target['corner_spin_deadline'])
        if now < s['brake_at']+cfg.brake_seconds:
            return target
        if now >= target['corner_spin_deadline'] and not s.get('heading_reached'):
            self.debug.update(mode='TURN_TIMEOUT', reason='20 second spin limit; reset required')
            return None
        error = wrap(s['exit_yaw']-pose[2])
        self.debug.update(mode='PIVOT_TURN', reason='stationary turn to exit heading',
                          heading_error_deg=float(np.rad2deg(error)))
        near_exit = (ordinary is not None and not ordinary.get('held') and
                     .05 < ordinary['x_m'] <= .30 and
                     abs(np.arctan2(ordinary['y_m'], ordinary['x_m'])) <= np.deg2rad(20))
        aligned = abs(error) <= np.deg2rad(12)
        if aligned:
            s['heading_reached'] = True
        if s.get('heading_reached'):
            target['corner_heading_reached'] = True
            self.debug.update(mode='EXIT_REACQUIRE', reason='turn complete; waiting for confirmed near exit lane')
        self.debug['exit_lane_confirmations'] = s['exit_count']+1 if aligned and near_exit else 0
        s['exit_count'] = s['exit_count']+1 if aligned and near_exit else 0
        if s['exit_count'] >= cfg.exit_frames:
            result = dict(ordinary, corner_speed_cap=cfg.entry_speed_mps)
            self.reset()
            return result
        return target
