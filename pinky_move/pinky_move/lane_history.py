"""Bounded measured-boundary history for association, never blind commands."""
from collections import deque
import numpy as np
from .metric_lane import (curve_match_error, floor_curves, supported_chain,
                         arc_stations, resample_chain, nearest_on_chain,
                         first_self_intersection, select_lookahead)


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c,-s],[s,c]])


class SpatialLaneArchive:
    """Measured odom points in a rolling +/-1m box, without time expiry.

    Quantised world cells bound memory and merge repeated observations. Only
    actual observed boundaries enter this archive; it is not a steering path.
    Retention follows spatial range, not the recent association vote timeout.
    """
    def __init__(self, extent_m=1., cell_m=.01):
        self.extent_m, self.cell_m = extent_m, cell_m
        self.cells = {'left': {}, 'right': {}}
        self.stamp = None

    def clear(self):
        self.cells = {'left': {}, 'right': {}}
        self.stamp = None

    def remember(self, stamp, pose, observation):
        if (stamp is None or pose is None or np.asarray(pose).shape != (3,)
                or not np.isfinite(pose).all()
                or (self.stamp is not None and stamp <= self.stamp)):
            return
        self.stamp = int(stamp)
        r = rotation(pose[2])
        # Prune by CURRENT base coordinates, including the rear half-plane.
        for cells in self.cells.values():
            if cells:
                keys = list(cells)
                world = np.array([cells[k] for k in keys])
                local = (world-pose[:2])@r
                outside = np.any(np.abs(local) > self.extent_m, axis=1)
                for key, discard in zip(keys, outside):
                    if discard:
                        del cells[key]
        if not observation or observation.get('side') not in self.cells:
            return
        curves = {observation['side']: observation.get('curve', [])}
        if observation.get('other') is not None:
            curves['right' if observation['side'] == 'left' else 'left'] = observation['other']
        for side, curve in curves.items():
            p = np.asarray(curve, float)
            if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 2
                    or not np.isfinite(p).all()):
                continue
            # Archive the measured chain, not the virtual centre/target.
            p = p[np.all(np.abs(p) <= self.extent_m, axis=1)]
            world = p@r.T+pose[:2]
            for q in world:
                key = tuple(np.rint(q/self.cell_m).astype(int))
                self.cells[side][key] = q.copy()

    def snapshot(self):
        return {side: np.array(list(cells.values())).reshape(-1, 2)
                for side, cells in self.cells.items()}


class LaneHistory:
    def __init__(self, max_frames=5, max_age_s=2.):
        self.frames = deque(maxlen=max_frames)
        self.max_age_ns = int(max_age_s*1e9)

    def clear(self):
        self.frames.clear()

    def remember(self, stamp, pose, observation):
        if stamp is None or pose is None or not observation:
            return
        if self.frames and stamp <= self.frames[-1][0]:
            return  # Replayed frames never add a vote or refresh history age.
        if not np.isfinite(pose).all():
            return
        while self.frames and stamp-self.frames[0][0] > self.max_age_ns:
            self.frames.popleft()
        curves = {observation['side']: np.asarray(observation['curve'],float)}
        if observation.get('other') is not None:
            curves['right' if observation['side']=='left' else 'left'] = np.asarray(observation['other'],float)
        if any(p.ndim != 2 or p.shape[1] != 2 or len(p)<5 or not np.isfinite(p).all()
               for p in curves.values()):
            return
        world = {side: p@rotation(pose[2]).T+pose[:2] for side,p in curves.items()}
        self.frames.append((int(stamp),world))

    def associate_curves(self, curves, stamp, pose):
        """Require two distinct recent observations agreeing on one candidate."""
        if stamp is None or pose is None or not np.isfinite(pose).all():
            return None
        references = [(t,p) for t,p in self.frames if 0 < stamp-t <= self.max_age_ns]
        if len(references)<2:
            return None
        matches=[]
        for index, curve in curves:
            for side in ('left','right'):
                errors=[]
                for _, observed in references:
                    if side not in observed:
                        continue
                    # World observations -> base frame at CURRENT camera time.
                    local=(observed[side]-pose[:2])@rotation(pose[2])
                    error=curve_match_error(curve,local)
                    if error <= .025:
                        errors.append(error)
                if len(errors)>=2:
                    matches.append((float(np.median(errors)),index,side))
        matches.sort()
        if not matches or (len(matches)>1 and matches[1][0]-matches[0][0]<.02):
            return None
        return matches[0][1],matches[0][2]

    def match(self, masks, calibration, stamp, pose, lo, hi, maximum, degree):
        if pose is None or stamp is None or len(self.frames)<2 or not masks:
            return None
        candidates=[]
        for index, mask in enumerate(masks):
            curves=floor_curves([mask],calibration,maximum,lo,hi,degree)
            if len(curves)==1:
                candidates.append((index,supported_chain(curves[0],lo,hi)))
        return self.associate_curves(candidates,stamp,pose)


class LocalLaneMap:
    """Ten-second odom observations, NOT ten seconds of blind driving.

    Keep measured left/right boundaries separately; never insert virtual lines.
    Stored geometry confirms current near observations rather than overwriting
    them with an old complete S. All returned steering points remain on the
    caller's freshly validated, normal-offset centre path.
    """
    def __init__(self, retention_s=10.):
        self.retention_s = retention_s
        self.frames = deque(maxlen=600)
        self.last_time = None
        self.last_pose = None

    def prepare(self, now, pose):
        if pose is None or np.asarray(pose).shape != (3,) or not np.isfinite(pose).all():
            return False
        if not np.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            return False
        # Odom discontinuities invalidate the shared coordinate frame.
        if self.last_pose is not None:
            yaw = (pose[2]-self.last_pose[2]+np.pi)%(2*np.pi)-np.pi
            if np.linalg.norm(np.asarray(pose)[:2]-self.last_pose[:2]) > .20 or abs(yaw) > np.pi/4:
                self.frames.clear()
        self.last_time, self.last_pose = now, np.asarray(pose).copy()
        while self.frames and now-self.frames[0][0] > self.retention_s:
            self.frames.popleft()
        return True

    def remember(self, observation, now, pose):
        if not observation or observation.get('side') not in ('left', 'right'):
            return
        curves = {observation['side']: observation['curve']}
        if observation.get('other') is not None:
            curves['right' if observation['side']=='left' else 'left'] = observation['other']
        world = {}
        for side, curve in curves.items():
            p = np.asarray(curve, float)
            if (p.ndim != 2 or p.shape[1] != 2 or len(p)<5 or not np.isfinite(p).all()
                    or first_self_intersection(p) is not None):
                return
            p = resample_chain(p, 80)
            outside = np.flatnonzero(np.linalg.norm(p, axis=1) > 1.)
            if len(outside):
                p = p[:outside[0]]  # Never join across a cropped gap.
            if len(p)<5:
                return
            world[side] = p@rotation(pose[2]).T+pose[:2]
        self.frames.append((now, world, float(observation.get('width', 0.))))

    def current_target(self, observation, ordinary, now, pose, lookahead, relaxed=False):
        """Confirm a near path using two prior same-side measured frames.

        This is only an alternative to stale S-route matching, not a bypass
        for perception errors, staged turns, missing odom or missing lanes.
        Retained ten-second evidence must include a match within two seconds.
        """
        if not observation or ordinary is None or ordinary.get('held'):
            return None
        side = observation.get('side')
        width = float(observation.get('width', 0.))
        p = np.asarray(observation.get('curve', []), float)
        if p.ndim != 2 or len(p)<5 or width<=0:
            return None
        p = resample_chain(p, 80)
        p = p[arc_stations(p)<=.10]
        if len(p)<5 or arc_stations(p)[-1]<(.03 if relaxed else .06):
            return None
        votes=[]
        for time, curves, old_width in reversed(self.frames):
            if not 0 < now-time <= self.retention_s or side not in curves or abs(width-old_width)>.03:
                continue
            reference=(curves[side]-pose[:2])@rotation(pose[2])
            q, idx, fraction=nearest_on_chain(p, reference)
            a,b=np.gradient(p,axis=0),np.diff(reference,axis=0)[idx]
            cosine=np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
            stations=arc_stations(reference)
            along=stations[idx]+fraction*np.diff(stations)[idx]
            if (np.max(np.linalg.norm(p-q,axis=1))<=(.035 if relaxed else .015)
                    and np.min(cosine)>=(.80 if relaxed else .95)
                    and np.min(np.diff(along))>=(-.01 if relaxed else -.002)
                    and along[-1]-along[0]>=(.03 if relaxed else .05)):
                votes.append(time)
            if len(votes)>=2:
                break
        if len(votes)<(1 if relaxed else 2) or now-max(votes)>2.:
            return None
        path=np.asarray(ordinary.get('center_path', []),float)
        if path.ndim!=2 or len(path)<5 or first_self_intersection(path) is not None:
            return None
        path=resample_chain(path,80)
        path=path[arc_stations(path)<=.10]
        if len(path)<5 or arc_stations(path)[-1]<.04:
            return None
        try:
            point, adaptive=select_lookahead(path,lookahead)
        except ValueError:
            return None
        result=dict(ordinary,center_path=path.tolist(),x_m=float(point[0]),y_m=float(point[1]),
                    adaptive=adaptive,local_map_assisted=True,local_map_votes=len(votes),
                    corner_speed_cap=min(.02,ordinary.get('corner_speed_cap',.02)))
        for key in ('approach_path','bend_remaining_path','bend_entry_limited'):
            result.pop(key,None)
        return result
