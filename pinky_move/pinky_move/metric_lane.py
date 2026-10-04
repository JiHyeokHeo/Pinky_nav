"""Local lane geometry and Pure Pursuit, independent of ROS and Nav2.

Coordinates throughout: robot forward x, left y, metres. Mask processing,
polynomial fitting, classification, path construction and control are separate
functions so they can be replayed/tested without motors or a YOLO model.
"""
import cv2
import numpy as np
from .robot_projection import robot_floor_points


def arc_stations(points):
    """Distance along the supplied connected chain; never reorder by x."""
    p = np.asarray(points, float)
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def supported_chain(points, lo, hi):
    """Keep the FIRST contiguous observed run in range, not disjoint pieces.

    Joining pieces on either side of a missing/out-of-range bend creates an
    unobserved shortcut. The near run is deliberately preferred to a longer
    distant run. Samples are not extrapolated to the clipping boundaries.
    """
    p = np.asarray(points, float)
    good = np.flatnonzero((p[:, 0] >= lo) & (p[:, 0] <= hi))
    if not len(good):
        return p[:0]
    run = np.split(good, np.flatnonzero(np.diff(good) > 1)+1)[0]
    return p[run]


def resample_chain(points, count=60):
    p = np.asarray(points, float)
    s = arc_stations(p)
    keep = np.r_[True, np.diff(s) > 1e-8]
    p, s = p[keep], s[keep]
    if len(p) < 2:
        raise ValueError('degenerate connected curve')
    stations = np.linspace(0., s[-1], count)
    return np.column_stack([np.interp(stations, s, p[:, axis]) for axis in (0, 1)])


def nearest_on_chain(points, reference):
    """Closest segment projection, with ordered segment index and fraction."""
    p, r = np.asarray(points, float), np.asarray(reference, float)
    d = np.diff(r, axis=0)
    denom = np.sum(d*d, axis=1)
    t = np.clip(np.sum((p[:, None]-r[None, :-1])*d, axis=2) /
                np.maximum(denom, 1e-12), 0., 1.)
    projected = r[None, :-1]+t[:, :, None]*d
    dist2 = np.sum((p[:, None]-projected)**2, axis=2)
    indices = np.argmin(dist2, axis=1)
    row = np.arange(len(p))
    return projected[row, indices], indices, t[row, indices]


def curve_match_error(a, b, lo=.05, hi=2.):
    """Compare connected geometry and tangent, including nearly lateral bends.

    Only overlapping interior support counts. Endpoint-only proximity cannot
    identify a boundary. Normal distances replace unsafe interpolation on x.
    """
    a, b = supported_chain(a, lo, hi), supported_chain(b, lo, hi)
    if len(a) < 2 or len(b) < 2:
        return float('inf')
    a, b = resample_chain(a), resample_chain(b)
    q, indices, fraction = nearest_on_chain(a, b)
    interior = (indices+fraction > .05) & (indices+fraction < len(b)-1.05)
    runs = np.split(np.flatnonzero(interior), np.flatnonzero(np.diff(np.flatnonzero(interior)) > 1)+1)
    runs = [run for run in runs if len(run) >= 3 and arc_stations(a[run])[-1] >= .04]
    if not runs:
        return float('inf')
    run = max(runs, key=len)
    # Reversed traversal or a different turn is not the same observed boundary.
    ta = np.gradient(a, axis=0)[run]
    tb = np.diff(b, axis=0)[indices[run]]
    cosine = np.sum(ta*tb, axis=1)/np.maximum(
        np.linalg.norm(ta, axis=1)*np.linalg.norm(tb, axis=1), 1e-12)
    if np.median(cosine) < .82 or np.any(np.diff(indices[run]+fraction[run]) < -.1):
        return float('inf')
    return float(np.median(np.linalg.norm(a[run]-q[run], axis=1)))


def short_curve_match_error(a, b):
    """Match the WHOLE short observation, not endpoint proximity alone.

    The regular interior matcher loses a little support at both ends; a 4 cm
    complete stripe can therefore fail to match itself. Limit this exception
    to 4–6 cm chains agreeing everywhere within 5 mm in traversal order.
    """
    a, b = np.asarray(a), np.asarray(b)
    if not all(.04 <= arc_stations(p)[-1] <= .06 for p in (a, b)):
        return float('inf')
    a, b = resample_chain(a), resample_chain(b)
    delta = np.linalg.norm(a-b, axis=1)
    ta, tb = np.diff(a, axis=0), np.diff(b, axis=0)
    cosine = np.sum(ta*tb, axis=1)/np.maximum(
        np.linalg.norm(ta, axis=1)*np.linalg.norm(tb, axis=1), 1e-12)
    return float(np.median(delta)) if max(delta) <= .005 and min(cosine) >= .95 else float('inf')


def first_self_intersection(points):
    """First nonadjacent segment crossing/touch; None for a simple open path."""
    p = np.asarray(points, float)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for j in range(2, len(p)-1):
        a, b = p[:j-1], p[1:j]
        c, d = p[j], p[j+1]
        ab, cd = b-a, d-c
        denominator = cross(ab, cd)
        nonparallel = abs(denominator) > 1e-12
        safe = np.where(nonparallel, denominator, 1.)
        t, u = cross(c-a, cd)/safe, cross(c-a, ab)/safe
        if np.any(nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            return j
        # Collinear overlap, not just strict crossings, is also invalid.
        collinear = (~nonparallel) & (abs(cross(c-a, ab)) < 1e-10)
        norm = np.maximum(np.sum(ab*ab, axis=1), 1e-12)
        tc, td = np.sum((c-a)*ab, axis=1)/norm, np.sum((d-a)*ab, axis=1)/norm
        if np.any(collinear & (np.maximum(tc, td) >= 0) & (np.minimum(tc, td) <= 1)):
            return j
    return None


class LaneImageIdentity:
    """Carry measured left/right mask identities between recent camera frames.

    Backward optical flow maps CURRENT pixels onto the PREVIOUS image. Only a
    unique mask overlap is accepted; an old image is never a driving target.
    The images are reduced to 160x120 to keep this inexpensive on the robot.
    """
    def __init__(self, tracking_gap_s=.8):
        self.tracking_gap_s = tracking_gap_s
        self.previous_gray = None
        self.previous_masks = {}
        self.previous_at = None
        self.pending_gray = None
        self.pending_masks = []

    @staticmethod
    def _prepare(frame, masks):
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        small_masks = [cv2.resize(np.asarray(mask, np.uint8), (160, 120),
                                  interpolation=cv2.INTER_NEAREST) for mask in masks]
        return gray, small_masks

    def match(self, frame, masks, now_s):
        """Return a unique (mask index, side) match or None for this frame."""
        self.pending_gray, self.pending_masks = self._prepare(frame, masks)
        if (self.previous_gray is None or not self.previous_masks or not masks or
                self.previous_at is None or not 0 <= now_s-self.previous_at <= self.tracking_gap_s):
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.pending_gray, self.previous_gray, None, .5, 2, 15, 2, 5, 1.1, 0)
        yy, xx = np.indices((120, 160), dtype=np.float32)
        map_x, map_y = xx+flow[:, :, 0], yy+flow[:, :, 1]
        candidates = []
        for side, old_mask in self.previous_masks.items():
            warped = cv2.remap(old_mask, map_x, map_y, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
            warped = cv2.dilate(warped, np.ones((3, 3), np.uint8)) > 0
            for index, current in enumerate(self.pending_masks):
                current = current > 0
                intersection = np.count_nonzero(warped & current)
                union = np.count_nonzero(warped | current)
                if union:
                    candidates.append((intersection/union, index, side))
        candidates.sort(reverse=True)
        if (not candidates or candidates[0][0] < .18 or
                (len(candidates) > 1 and candidates[0][0]-candidates[1][0] < .08)):
            return None
        return candidates[0][1], candidates[0][2]

    def commit(self, now_s, roles):
        """Save masks whose roles were verified by metric geometry or flow."""
        self.previous_gray = self.pending_gray
        self.previous_masks = {side: self.pending_masks[index].copy()
                               for index, side in roles.items()
                               if side in ('left', 'right') and
                               0 <= index < len(self.pending_masks)}
        self.previous_at = now_s


def fit_connected_approach(points, degree=3):
    """Retain the first verified local section of a long connected curve.

    Grow an observed prefix from 8 to 28 cm in 4 cm increments. Stop at the
    FIRST failed section; never search past a bad bend for a better distant
    fit. Existing residual, self-intersection and downstream offset checks
    remain mandatory. This is local prefix fitting, not straight extrapolation.
    """
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<5 or not np.isfinite(points).all():
        raise ValueError('invalid connected approach')
    stations=arc_stations(points)
    last=None
    for length in (.08,.12,.16,.20,.24,.28):
        if stations[-1]<length:
            break
        prefix=points[stations<=length]
        try:
            if len(prefix)<5 or np.any(np.diff(arc_stations(prefix))>.03):
                break
            fit=fit_boundary(prefix,degree)
            distances=np.linalg.norm(prefix-nearest_on_chain(prefix,fit)[0],axis=1)
            rms=float(np.sqrt(np.mean(distances**2)))
            if rms>.006 or np.max(distances)>.008:
                break
            last=(prefix,fit,rms,float(arc_stations(prefix)[-1]))
        except ValueError:
            break
    if last is None:
        raise ValueError('no supported connected approach')
    return last


def floor_curves(masks, calibration, max_forward_m=2.0,
                 path_min_m=.05, path_max_m=None, degree=3, recover_short_hook=False,
                 component_recovery=True):
    """Compare row/column extraction instead of committing to one image axis.

    Perspective can make a legitimate near stripe too thick for column scans,
    while a far bend can be too wide for row scans. Validate EACH candidate's
    fit in the actual controller support interval before comparing them. Prefer
    a lower-residual fit; among fits within 2 mm RMS, prefer more observed support.
    Two valid but disagreeing candidates remain ambiguous, never spliced together.
    """
    curves = []
    for mask in masks:
        candidates = []
        for by_column in (False, True):
            for points in _axis_floor_curves([mask], calibration, max_forward_m, by_column):
                hi = max_forward_m if path_max_m is None else path_max_m
                near = supported_chain(points, path_min_m, hi)
                try:
                    fit = fit_boundary(near, degree)
                except ValueError:
                    continue
                projected, _, _ = nearest_on_chain(near, fit)
                rms = float(np.sqrt(np.mean(np.sum((near-projected)**2, axis=1))))
                candidates.append((points, fit, rms, float(arc_stations(near)[-1])))
        if len(candidates) == 2:
            a, b = candidates[0][1], candidates[1][1]
            error = min(curve_match_error(a, b), curve_match_error(b, a))
            if np.isfinite(error) and error > .035:
                continue
        if candidates:
            best_rms = min(candidate[2] for candidate in candidates)
            accurate = [candidate for candidate in candidates if candidate[2] <= best_rms+.002]
            chosen = max(accurate, key=lambda candidate: candidate[3])
            # A short endpoint hook can fit very accurately while discarding
            # the main stripe. Prefer a single much longer observed chain if
            # its fit remains within 6 mm RMS. Do not join chains, relax the
            # fit/self-intersection checks, or flatten a full-sized corner.
            longer = [candidate for candidate in candidates
                      if chosen[3] < .08 and candidate[3] >= max(.24, 3*chosen[3])
                      and candidate[2] <= .006]
            if recover_short_hook and len(longer) == 1:
                chosen = longer[0]
            # At a lateral turn, wide near rows fail the run-width gate while
            # a thin FAR outgoing leg still fits. A successful axis fit is not
            # evidence that its starting point is the nearest visible boundary.
            # Prefer a real connected prefix only in this specific far-start
            # case. Never splice it to the far leg or extrapolate a missing bend.
            chosen_near = supported_chain(chosen[0], path_min_m, hi)
            if (len(masks) == 1 and chosen[3] >= .08 and
                    chosen_near[0,0] > max(.20,path_min_m+.06)):
                connected = connected_floor_curve(mask, calibration, max_forward_m)
                if len(connected) == 1:
                    prefix = supported_chain(connected[0],path_min_m,hi)
                    local_candidate = None
                    try:
                        fit = fit_boundary(prefix,degree)
                        q,_,_ = nearest_on_chain(prefix,fit)
                        rms = float(np.sqrt(np.mean(np.sum((prefix-q)**2,axis=1))))
                        if rms<=.006:
                            local_candidate=(connected[0],fit,rms,float(arc_stations(prefix)[-1]))
                    except ValueError:
                        pass
                    if local_candidate is None:
                        try:
                            local_candidate=fit_connected_approach(prefix,degree)
                        except ValueError:
                            pass
                    if local_candidate is not None:
                        prefix=supported_chain(local_candidate[0],path_min_m,hi)
                        if (prefix[0,0] <= path_min_m+.04 and
                                chosen_near[0,0]-prefix[0,0] >= .06 and
                                arc_stations(prefix)[-1] >= .075):
                            chosen = local_candidate
            curves.append(chosen[0])
    if not curves and len(masks) == 1:
        # Only failed axis extraction reaches this path. Apply the same metric
        # fitting checks to the connected observation before accepting it.
        for points in connected_floor_curve(masks[0], calibration, max_forward_m):
            try:
                hi = max_forward_m if path_max_m is None else path_max_m
                fit_boundary(supported_chain(points, path_min_m, hi), degree)
            except ValueError:
                continue
            curves.append(points)
    if not curves and component_recovery and len(masks) == 1:
        component = near_connected_component(masks[0])
        if component is not None:
            curves = floor_curves([component], calibration, max_forward_m,
                                  path_min_m, path_max_m, degree,
                                  recover_short_hook, component_recovery=False)
    return curves


def near_connected_component(mask):
    """Failure-only removal of <=2px bridges to an entirely distant fragment.

    Never merge gaps or choose between two near stripes. Keep an actual large
    connected component, and require every rejected component above the near
    image band. This does not establish left/right identity by itself.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    binary = (mask > 0).astype(np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    significant = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA] >= 30]
    if len(significant) < 2:
        return None
    near = [i for i in significant if stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT] >= .65*mask.shape[0]]
    if len(near) != 1:
        return None
    index = near[0]
    if stats[index,cv2.CC_STAT_AREA] < max(100,.15*np.count_nonzero(binary)):
        return None
    return (labels == index).astype(np.uint8)


def _axis_floor_curves(masks, calibration, max_forward_m, by_column):
    """Extract continuous runs in one axis; reject wide or ambiguous branches."""
    if not masks:
        return []
    height, width = masks[0].shape
    curves = []
    for mask in masks:
        # Both orientations use the same branch/width gates. Neither orientation
        # establishes the boundary's left/right identity.
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if by_column:
            # Thin segmentation bridges can otherwise become a long column
            # track between unrelated white regions. Remove <=2px connectors
            # before this alternate sampler; do not fill gaps or join regions.
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                      np.ones((3, 3), np.uint8))
        sampled = binary.T if by_column else binary
        scan_height, scan_width = sampled.shape
        tracks = []
        # Include distant visible rows too; projection, not a fixed near_y,
        # decides which rays intersect the configured ground range.
        for row in range(scan_height-8, -1, -4):
            xs = np.flatnonzero(sampled[row])
            if len(xs) < 2:
                continue
            # Separate disjoint runs before finding centres. A merged YOLO
            # instance can contain the true stripe and a distant white patch;
            # their combined median can land on bare floor.
            runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1)+1)
            used = set()
            for run in runs:
                if len(run) < 2 or run[-1]-run[0] > scan_width*.2:
                    continue
                centre = float(np.median(run))
                options = []
                for index, track in enumerate(tracks):
                    if index in used or not 0 < track[-1][1]-row <= 12:
                        continue
                    prediction = track[-1][0]
                    if len(track) >= 2:
                        dx = track[-1][0]-track[-2][0]
                        dy = track[-1][1]-track[-2][1]
                        prediction += dx/dy*(row-track[-1][1])
                    if abs(centre-prediction) <= max(16., scan_width*.08):
                        options.append((abs(centre-prediction), index))
                options.sort()
                if options and (len(options) == 1 or options[1][0]-options[0][0] > 8):
                    index = options[0][1]
                    tracks[index].append((centre, row))
                    used.add(index)
                else:
                    tracks.append([(centre, row)])
                    used.add(len(tracks)-1)
        tracks = sorted((t for t in tracks if len(t) >= 5), key=len, reverse=True)
        if not tracks or (len(tracks)>1 and len(tracks[1]) >= .7*len(tracks[0])):
            continue  # Two similarly supported branches are genuinely ambiguous.
        pixels = np.asarray([(row, centre) if by_column else (centre, row)
                             for centre, row in tracks[0]], dtype=float)
        try:
            projected = robot_floor_points(pixels, calibration, (width, height), max_forward_m)
        except ValueError:
            continue
        points = projected[np.isfinite(projected).all(axis=1)]
        if len(points) < 5:
            continue
        # Reverse the whole chain if necessary, NEVER sort individual x values.
        # The nearer endpoint establishes traversal, not left/right identity.
        if points[-1, 0] < points[0, 0]:
            points = points[::-1].copy()
        curves.append(points)
    return curves


def connected_floor_curve(mask, calibration, max_forward_m=2.):
    """Failure-only centreline extraction, preserving image connectivity.

    Zhang-Suen thinning needs no extra runtime dependency. Follow from the
    uniquely nearest image endpoint; stop at a branch rather than choosing an
    unseen shortcut. Never join components or sort floor coordinates by x.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return []
    height, width = mask.shape
    small = cv2.resize((mask > 0).astype(np.uint8),
                       ((width+1)//2, (height+1)//2), interpolation=cv2.INTER_NEAREST)
    _, _, stats, _ = cv2.connectedComponentsWithStats(small, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if not len(areas) or np.count_nonzero(areas >= max(12, .05*max(areas))) != 1:
        return []  # A merged instance containing two real stripes is ambiguous.
    a = np.pad(small, 1)
    for iteration in range(128):
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            neighbours = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i+1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            triplets = ((p[0]*p[2]*p[4], p[2]*p[4]*p[6]) if phase == 0 else
                        (p[0]*p[2]*p[6], p[0]*p[4]*p[6]))
            remove = ((a[1:-1, 1:-1] == 1) & (neighbours >= 2) &
                      (neighbours <= 6) & (transitions == 1) &
                      (triplets[0] == 0) & (triplets[1] == 0))
            changed |= bool(remove.any())
            a[1:-1, 1:-1][remove] = 0
        if not changed:
            break
    else:
        return []  # Bounded work; incomplete thinning is not a reliable path.
    vertices = set(map(tuple, np.argwhere(a[1:-1, 1:-1])))

    def adjacent(point):
        row, col = point
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc) or (row+dr, col+dc) not in vertices:
                    continue
                # Avoid diagonal triangles around an orthogonally linked bend.
                if dr and dc and ((row+dr, col) in vertices or (row, col+dc) in vertices):
                    continue
                result.append((row+dr, col+dc))
        return result

    endpoints = sorted((v for v in vertices if len(adjacent(v)) == 1), reverse=True)
    if (not endpoints or endpoints[0][0] < .65*small.shape[0] or
            (len(endpoints) > 1 and endpoints[0][0]-endpoints[1][0] < 4)):
        return []
    chain, seen = [endpoints[0]], {endpoints[0]}
    while True:
        onward = [v for v in adjacent(chain[-1]) if v not in seen]
        if len(onward) != 1:
            break  # Retain only the connected prefix before an ambiguous fork.
        chain.append(onward[0])
        seen.add(onward[0])
    if len(chain) < 12:
        return []
    pixels = np.array([(col*2., row*2.) for row, col in chain])
    projected = robot_floor_points(pixels, calibration, (width, height), max_forward_m)
    indices = np.flatnonzero(np.isfinite(projected).all(axis=1))
    if not len(indices):
        return []
    indices = np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)[0]
    return [projected[indices]] if len(indices) >= 5 else []


def fit_boundary(points, degree=3):
    """Fit (x(s), y(s)) on observed arc length, including lateral corner exits.

    No x sorting or extrapolation. Prefer the simplest polynomial within 2 mm
    of the best residual. Reject spikes, crossings and reversed tangents.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 5 or not np.isfinite(p).all():
        raise ValueError('insufficient finite boundary points')
    s = arc_stations(p)
    keep = np.r_[True, np.diff(s) > 1e-6]
    p, s = p[keep], s[keep]
    if len(p) < 5 or s[-1] < .04 or degree not in (1, 2, 3):
        raise ValueError('insufficient polynomial support')
    if first_self_intersection(p) is not None:
        raise ValueError('boundary self intersection')
    # Uniform arc sampling prevents densely sampled lateral pixels dominating.
    samples = resample_chain(p, max(60, len(p)))
    ss = np.linspace(0., s[-1], len(samples))
    candidates = []
    for d in range(1, degree+1):
        polys = [np.polynomial.Polynomial.fit(ss, samples[:, i], d) for i in (0, 1)]
        predicted = np.column_stack([poly(ss) for poly in polys])
        residual = np.linalg.norm(predicted-samples, axis=1)
        candidates.append((float(np.sqrt(np.mean(residual**2))), float(max(residual)), polys))
    best_error = min(c[0] for c in candidates)
    error, maximum, polys = next(c for c in candidates if c[0] <= best_error+.002)
    if error > .025 or maximum > .06:
        raise ValueError('boundary fit residual too large')
    stations = np.linspace(0., s[-1], 60)
    fitted = np.column_stack([poly(stations) for poly in polys])
    if first_self_intersection(fitted) is not None:
        raise ValueError('fitted boundary self intersection')
    return fitted


class OffsetCurveError(ValueError):
    """Invalid offset plus its connected, pre-failure prefix (if any)."""
    def __init__(self, message, prefix):
        super().__init__(message)
        self.prefix = prefix


def normal_offset(curve, distance, role='offset'):
    """Offset along unit normal (-dy/ds, dx/ds), not a fixed displacement.

    Positive distance moves LEFT of the boundary's forward tangent. Therefore
    a left boundary uses -lane_width/2, and a right boundary +lane_width/2.
    Reject self-folding offset paths instead of sorting them into a fake path.
    """
    p = np.asarray(curve, float)
    if len(p) < 3 or not np.isfinite(p).all() or not np.isfinite(distance):
        raise ValueError('invalid offset curve')
    tangents = np.gradient(p, axis=0)
    if np.any(np.linalg.norm(tangents, axis=1) < 1e-8):
        raise ValueError('degenerate boundary tangent')
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    shifted = p + distance*normals
    # A decrease in x can be a legitimate bend. A reversal relative to the
    # source tangent is an actual offset cusp and must not be driven through.
    reversed_segments = np.flatnonzero(
        np.sum(np.diff(shifted, axis=0)*np.diff(p, axis=0), axis=1) <= 1e-12)
    crossing = first_self_intersection(shifted)
    failures = reversed_segments.tolist()+([] if crossing is None else [crossing])
    if failures:
        # Leave one sample before the first faulty segment as a margin. Do not
        # jump to a later valid-looking section beyond a cusp or intersection.
        prefix = shifted[:max(0, min(failures)-1)]
        raise OffsetCurveError(f'{role} curve folds back (offset={distance:.3f}m)', prefix)
    return shifted


def center_path_from_boundary(curve, distance):
    """Use only a long enough, forward, connected prefix before a distant cusp.

    Near cusps still stop. The controller never follows the invalid remainder;
    the shortened observed path makes lookahead adaptive rather than extending
    it. Display-only full-width boundaries remain separately optional.
    """
    try:
        return normal_offset(curve, distance, 'center_path'), None
    except OffsetCurveError as exc:
        p = exc.prefix
        if (len(p) < 5 or arc_stations(p)[-1] < .06 or np.any(p[:, 0] <= .05)
                or first_self_intersection(p) is not None):
            raise
        return p, 'center_path truncated before distant fold'


def fit_drivable_boundary(near, distance, degree=3, fitted=None):
    """Refit a shorter OBSERVED S-bend prefix only after full offset fails.

    Never apply to a sharp corner or straighten it into an artificial road.
    Bound deviation from observations to 8 mm; retain cusp/intersection checks.
    """
    fitted = fit_boundary(near, degree) if fitted is None else fitted
    try:
        centre, warning = center_path_from_boundary(fitted, distance)
        return fitted, centre, warning
    except OffsetCurveError as original:
        # Local import avoids a module cycle: corner classification uses the
        # geometry primitives above, but does not invoke this path builder.
        from .lane_corner import classify_boundary, CornerConfig
        if classify_boundary(near, CornerConfig())['kind'] != 'S_BEND':
            raise
        stations = arc_stations(near)
        for length in (.26, .22, .18, .15, .12):
            if length >= stations[-1]-.02:
                continue
            prefix = near[stations <= length]
            try:
                candidate = fit_boundary(prefix, degree)
                errors = np.linalg.norm(prefix-nearest_on_chain(prefix, candidate)[0], axis=1)
                if max(errors) > .008:
                    continue
                centre = normal_offset(candidate, distance, 'center_path')
                if arc_stations(centre)[-1] < .06 or np.any(centre[:, 0] <= .05):
                    continue
                select_lookahead(centre, .22)  # Also checks intersection/forward support.
                return candidate, centre, 'S-bend observed prefix refit'
            except ValueError:
                continue
        raise original


def near_pixel_side(mask):
    """Recovery hint only: median near-field pixels relative to image centre.

    Ignore distant-only masks and a central 10% dead band. This is not a
    persistent identity: a tracked curve can cross the image centre in a bend.
    Sample up to 18% of image height above the detected bottom (formerly 8%),
    while keeping the upper cutoff at 65% of image height.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    height, width = mask.shape
    bottom = int(rows.max())
    if bottom < .65 * height:
        return None
    near = cols[rows >= max(.65 * height, bottom - .18 * height)]
    if len(near) < 10:
        return None
    offset = float(np.median(near)) - width / 2
    if abs(offset) <= .05 * width:
        return None
    return 'right' if offset > 0 else 'left'


def distinct_near_masks(masks):
    """Drop only strongly overlapping same-side segmentation duplicates.

    Near pixels must agree on side and overlap >=90%, AND whole-mask IoU must
    be >=85%. A nearby candidate is not sufficient to erase a different far
    branch. Keep the first (YOLO order) mask unchanged; never union masks.
    """
    kept, evidence = [], []
    for mask in masks:
        side = near_pixel_side(mask)
        binary = mask > 0 if isinstance(mask, np.ndarray) and mask.ndim == 2 else None
        duplicate = False
        if side is not None:
            for old_side, old in evidence:
                if old_side != side or old is None or old.shape != binary.shape:
                    continue
                start = int(binary.shape[0]*.65)
                near_union = np.count_nonzero(binary[start:] | old[start:])
                near_overlap = np.count_nonzero(binary[start:] & old[start:])
                full_union = np.count_nonzero(binary | old)
                if (near_union and full_union and near_overlap/near_union >= .9 and
                        np.count_nonzero(binary & old)/full_union >= .85):
                    duplicate = True
                    break
        if not duplicate:
            kept.append(mask)
            evidence.append((side, binary))
    return kept


def unique_near_boundary(masks, calibration, max_forward_m, lo, hi, degree):
    """Candidate for temporal confirmation when distant masks cannot pair.

    Exactly one reliable near stripe must be at least 6 cm ahead of all other
    projected starts. An unprojectable mask is ignored only if wholly distant
    in the image. This is not permission to bypass an invalid measured width.
    """
    observations = []
    for index, mask in enumerate(masks):
        curves = floor_curves([mask], calibration, max_forward_m, lo, hi, degree)
        if len(curves) != 1:
            rows = np.nonzero(mask)[0]
            if not len(rows) or rows.max() >= .65*mask.shape[0]:
                return None
            continue
        try:
            near = supported_chain(curves[0], lo, hi)
            fitted = fit_boundary(near, degree)
            side = initial_boundary_side(fitted)
        except ValueError:
            return None
        observations.append((float(near[0,0]), index, side, curves[0]))
    observations.sort(key=lambda item: item[0])
    if not observations:
        return None
    x, index, side, curve = observations[0]
    if (x > .28 or near_pixel_side(masks[index]) != side or
            (len(observations) > 1 and observations[1][0]-x < .06)):
        return None
    return index, side, curve


def center_from_pair(left, right):
    """Pair forward stations, or normal intersections for a lateral corner.

    The legacy forward-station construction remains valid only for x-monotone
    curves. A lateral path uses unique normal intersections with increasing
    correspondence along the opposite boundary, never arbitrary nearest ends.
    """
    if (np.any(np.diff(left[:, 0]) <= 0) or np.any(np.diff(right[:, 0]) <= 0)
            or min(np.ptp(left[:, 0]), np.ptp(right[:, 0])) < .04):
        return _normal_pair(left, right)
    lo = max(left[0, 0], right[0, 0], .05)
    hi = min(left[-1, 0], right[-1, 0])
    # Accept a short *observed* segment; lookahead uses its endpoint instead
    # of extrapolating unseen road. Never join non-overlapping boundaries.
    if hi-lo < .02:
        # On a diagonal/turn the two boundaries may have no common x even
        # though they have real, unique correspondences along their normals.
        # Reuse the existing strict normal matcher; do not extend either line.
        try:
            return _normal_pair(left, right)
        except ValueError as exc:
            raise ValueError(f'no common observed lane interval: x_overlap={hi-lo:.4f}m, '
                             f'minimum=0.0200m; normal fallback: {exc}') from exc
    xs = np.linspace(lo, hi, 60)
    ly = np.interp(xs, left[:, 0], left[:, 1])
    ry = np.interp(xs, right[:, 0], right[:, 1])
    widths = ly-ry
    if np.any(widths <= .04) or np.any(widths > 2.):
        raise ValueError('crossing or implausibly separated boundaries')
    return np.column_stack((xs, (ly+ry)/2)), widths


def _normal_pair(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    tangents = np.gradient(left, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    inward = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    segments = np.diff(right, axis=0)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    entries = []
    runs = []
    for i, (p, normal) in enumerate(zip(left, inward)):
        delta = right[:-1]-p
        denom = cross(normal, segments)
        safe = np.where(abs(denom) > 1e-10, denom, 1.)
        width = cross(delta, segments)/safe
        fraction = cross(delta, normal)/safe
        cos = segments@tangents[i]/np.maximum(np.linalg.norm(segments, axis=1), 1e-12)
        hits = np.flatnonzero((abs(denom) > 1e-10) & (width > .04) &
                             (width < 2.) & (fraction >= 0) & (fraction < 1.) & (cos > .82))
        if len(hits) != 1:
            if entries:
                runs.append(entries)
                entries = []  # Never bridge an ambiguous or missing interval.
            continue
        j = hits[0]
        station = j+fraction[j]
        if entries and station <= entries[-1][3]:
            runs.append(entries)
            entries = []
            continue
        entries.append((i, p+.5*width[j]*normal, width[j], station))
    if entries:
        runs.append(entries)
    # Three ordered correspondences suffice if they also span 2 cm below.
    # Keep unique intersections, tangent agreement and gap rejection unchanged.
    # A noisy leading singleton must not hide later usable support. Evaluate
    # separate contiguous runs in near-to-far input order; never concatenate
    # them across a missing/ambiguous interval or interpolate across the gap.
    longest = max((len(run) for run in runs), default=0)
    for run in runs:
        if len(run) < 3:
            continue
        centre = np.array([e[1] for e in run])
        widths = np.array([e[2] for e in run])
        if (arc_stations(centre)[-1] < .02 or first_self_intersection(centre) is not None
                or np.ptp(widths) > max(.05, .5*np.median(widths))):
            continue
        return centre, widths
    if longest < 3:
        raise ValueError(f'no common observed lane interval: normal_pairs={longest}, minimum=3, runs={len(runs)}')
    raise ValueError('inconsistent normal lane pair')


def optional_virtual_boundary(curve, distance):
    """A full-width display boundary is optional; centre-path validity is not.

    Never sort or use a folded virtual boundary for control/classification.
    Only this specific display-geometry failure is recoverable.
    """
    try:
        return normal_offset(curve, distance, 'virtual_boundary'), None
    except ValueError as exc:
        if 'virtual_boundary curve folds back' not in str(exc):
            raise
        return None, str(exc)


def select_lookahead(path, distance):
    """First forward intersection with the lookahead circle of radius distance.

    Ld is Euclidean range, not simply x. If the visible path ends before the
    circle, use its endpoint and report adaptive=True; never extrapolate a lane.
    """
    p = np.asarray(path, float)
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or
            not np.isfinite(p).all() or not np.isfinite(distance) or distance <= 0):
        raise ValueError('invalid lookahead path/distance')
    if p[0, 0] <= .05:
        raise ValueError('no forward centre path')
    invalid = np.flatnonzero(p[:, 0] <= .05)
    if len(invalid):
        p = p[:invalid[0]]  # Never join across an invalid segment.
    if len(p)<2 or first_self_intersection(p) is not None:
        raise ValueError('no forward centre path')
    if np.linalg.norm(p[0]) >= distance:
        return p[0].copy(), True
    for a, b in zip(p, p[1:]):
        d = b-a
        aa = float(d@d); bb = float(2*a@d); cc = float(a@a-distance**2)
        disc = bb*bb-4*aa*cc
        if aa <= 1e-12 or disc < 0:
            continue
        roots = sorted(((-bb-np.sqrt(disc))/(2*aa), (-bb+np.sqrt(disc))/(2*aa)))
        for fraction in roots:
            if 0 <= fraction <= 1:
                return a+fraction*d, False
    return p[-1].copy(), True


def classify_lanes(curves, reference_x, previous=None):
    """Label boundary roles at a near robot reference, preserving recent roles.

    This is not YOLO instance order. With history, match geometry to the prior
    left/right curves even when an S bend moves them across image centre.
    Multiple similarly plausible instances for one side are treated as ambiguous.
    """
    labelled = {}
    for curve in curves:
        scores = []
        if previous:
            for side in ('left', 'right'):
                old = previous.get(side+'_curve')
                if old is None:
                    continue
                old = np.asarray(old, float)
                score = curve_match_error(curve, old)
                if np.isfinite(score):
                    scores.append((score, side))
        scores.sort()
        if scores:
            if scores[0][0] > .10 or (len(scores)>1 and scores[1][0]-scores[0][0]<.02):
                continue
            score, side = scores[0]
        else:
            try:
                side = initial_boundary_side(curve, reference_x)
            except ValueError:
                continue
            score = abs(float(curve[0, 1]))
        labelled.setdefault(side, []).append((score, curve))
    result = {}
    for side, candidates in labelled.items():
        candidates.sort(key=lambda a: a[0])
        if len(candidates)>1 and candidates[1][0]-candidates[0][0] < .02:
            raise ValueError('multiple ambiguous boundaries on same side')
        result[side] = candidates[0][1]
    return result


def metric_target(masks, calibration, lookahead=.28, previous=None, degree=3,
                  max_forward_m=2., path_min_m=.05, path_max_m=None):
    masks = distinct_near_masks(masks)
    if len(masks) < 2:
        raise ValueError('two lane boundaries required')
    curves = []
    path_max_m = lookahead+.20 if path_max_m is None else path_max_m
    projected = floor_curves(masks, calibration, max_forward_m, path_min_m, path_max_m, degree)
    rejected = []
    for p in projected:
        try:
            near = supported_chain(p, path_min_m, path_max_m)
            curves.append(fit_boundary(near, degree))
        except ValueError as exc:
            rejected.append(str(exc))
            continue
    lanes = classify_lanes(curves, min(lookahead, .30), previous)
    if ('left' not in lanes or 'right' not in lanes) and not previous and len(masks) == 2:
        # A robot off centre may see BOTH boundaries on the same side. Recover
        # roles only from a unique ordered geometric pair, not mask order.
        # Individual extraction also enables connected-mask fallback when one
        # member failed ordinary row/column extraction in the pair batch.
        candidates = []
        for mask in masks:
            one = floor_curves([mask], calibration, max_forward_m, path_min_m, path_max_m, degree)
            if len(one) != 1:
                break
            try:
                candidates.append(fit_boundary(supported_chain(one[0], path_min_m, path_max_m), degree))
            except ValueError:
                break
        pairings = []
        if len(candidates) == 2:
            for left, right in (candidates, candidates[::-1]):
                try:
                    path, widths = _normal_pair(left, right)
                    # Require real positive ordering and useful support, not
                    # coincident duplicate detections or a crossing pair.
                    if np.min(widths) < .04 or first_self_intersection(path) is not None:
                        continue
                    select_lookahead(path, lookahead)
                    pairings.append(dict(left=left, right=right))
                except ValueError:
                    continue
        if len(pairings) == 1:
            lanes = pairings[0]
    if 'left' not in lanes or 'right' not in lanes:
        detail = ', '.join(sorted(set(rejected))) or 'side association ambiguous'
        raise ValueError('no unambiguous left/right boundary pair '
                         f'(projected={len(projected)}, fitted={len(curves)}; {detail})')
    path, widths = center_from_pair(lanes['left'], lanes['right'])
    point, adaptive = select_lookahead(path, lookahead)
    _, segment, fraction = nearest_on_chain([point], path)
    j, f = int(segment[0]), float(fraction[0])
    width = float(widths[j]*(1-f)+widths[j+1]*f)
    forward_pair = (np.all(np.diff(lanes['left'][:, 0]) > 0) and
                    np.all(np.diff(lanes['right'][:, 0]) > 0) and
                    min(np.ptp(lanes['left'][:, 0]), np.ptp(lanes['right'][:, 0])) >= .04 and
                    min(lanes['left'][-1, 0], lanes['right'][-1, 0])-
                    max(lanes['left'][0, 0], lanes['right'][0, 0], .05) >= .02)
    tangent = path[j+1]-path[j]
    normal_width = width*abs(tangent[0])/np.linalg.norm(tangent) if forward_pair else width
    if previous and abs(point[1]-previous['y_m']) > max(.05, .25*normal_width):
        raise ValueError('centre path discontinuity')
    return dict(x_m=float(point[0]), y_m=float(point[1]), width_m=width,
                normal_width_m=float(normal_width), boundary_count=2, inferred=False,
                adaptive=adaptive, center_path=path.tolist(), width_source='measured',
                left_curve=lanes['left'].tolist(), right_curve=lanes['right'].tolist())


def initial_boundary_side(curve, lookahead=.28):
    """Cold-start identity from the nearest observed 2 cm band.

    Steering lookahead is NOT the side-classification reference: an S bend may
    cross y=0 farther ahead without changing from left to right. Retain the
    lookahead argument for callers, but never use it to move this initial band.
    If the nearest band itself straddles the centre, do not guess a side.
    """
    s = arc_stations(curve)
    if s[-1] < .02 or curve[0, 0] < .05:
        raise ValueError('no near reference for initial boundary')
    y = np.interp([0., .01, .02], s, curve[:, 1])
    if np.all(y > .01):
        return 'left'
    if np.all(y < -.01):
        return 'right'
    raise ValueError('initial boundary overlaps robot reference')


class LaneWidthEstimator:
    """Remember a robust ground-plane width from pairs at a fixed sample tick.

    One narrow/wide segmentation glitch cannot replace the last accepted width.
    A genuinely changed width needs three consistent, spaced measurements.
    """

    def __init__(self, interval_s=.5, tolerance_m=.02, minimum_m=0.):
        self.interval_s = interval_s
        self.tolerance_m = tolerance_m
        self.minimum_m = minimum_m
        self.value = None
        self.last_tick = None
        self.candidates = []

    def observe(self, width, now_s):
        if not np.isfinite([width, now_s]).all() or not self.minimum_m <= width <= 1.5:
            raise ValueError('lane width implausibly narrow or invalid; remeasure pair')
        if self.last_tick is not None and now_s-self.last_tick < self.interval_s:
            return self.value
        self.last_tick = now_s
        if (self.candidates and
                abs(width-float(np.median(self.candidates))) > self.tolerance_m):
            self.candidates = []
        self.candidates.append(float(width))
        self.candidates = self.candidates[-3:]
        if len(self.candidates) == 3:
            candidate = float(np.median(self.candidates))
            if self.value is None:
                self.value = candidate
            elif abs(candidate-self.value) <= self.tolerance_m:
                self.value = .7*self.value+.3*candidate
            else:
                # A sustained real width change can replace the old value.
                self.value = candidate
        return self.value


class MetricLaneTracker:
    """Track the visible boundary and rebuild the centre on every result.

    No odometry compensation: match consecutive observed curves, inferred
    single-line speed is limited by the caller. A zero timeout allows continuous
    visible-line tracking. Configured width permits a single-line cold start;
    two consecutive measured pairs replace that width. Never renew the width
    confirmation timestamp using inferred geometry.
    """
    def __init__(self, minimum_lane_width_m=0., width_measure_interval_s=.5, tracking_gap_s=.8):
        if not np.isfinite(tracking_gap_s) or tracking_gap_s <= 0:
            raise ValueError('invalid tracking gap')
        self.tracking_gap_s = tracking_gap_s
        self.confirmed = None
        self.confirmed_at = None
        self.streak = 0
        self.previous = None
        self.observed_curve = None
        self.observed_side = None
        self.observed_at = None
        self.initial_side = None
        self.bootstrap_hint = None
        self.pair_hint = None
        self.pair_transition = None
        self.preferred_observation_side = None
        self.context_only_indices = []
        self.requires_pair = False
        self.recovery = None
        self.recovery_side = None
        self.width_estimator = LaneWidthEstimator(
            interval_s=width_measure_interval_s,
            minimum_m=minimum_lane_width_m)

    @staticmethod
    def _same_curve(a, b):
        """Require shared support, close lateral position and similar tangent."""
        return curve_match_error(a, b, .15, .45) <= .035

    def update(self, masks, calibration, now_s, lookahead=.28, timeout_s=0.,
               lane_width=0., degree=3, max_forward_m=2.,
               path_min_m=.05, path_max_m=None, image_side_hint=None, image_match=None,
               recovery_side_hint=None):
        # Bootstrap/recovery is a transaction: no confirmed geometry or role
        # transition survives failed fitting, offset, target or continuity checks.
        self.last_observation = None  # Fresh measured boundary, never a virtual curve.
        self.context_only_indices = []
        bootstrap = self.confirmed is None
        fields = ('confirmed', 'confirmed_at', 'previous', 'observed_curve',
                  'observed_side', 'observed_at', 'requires_pair')
        before = {name: getattr(self, name) for name in fields}
        try:
            return self._update(masks, calibration, now_s, lookahead, timeout_s,
                                lane_width, degree, max_forward_m, path_min_m,
                                path_max_m, image_side_hint, image_match, recovery_side_hint)
        except ValueError:
            self.pair_transition = None
            if bootstrap:
                for name, value in before.items():
                    setattr(self, name, value)
            raise

    def _update(self, masks, calibration, now_s, lookahead=.28, timeout_s=0.,
                lane_width=0., degree=3, max_forward_m=2.,
                path_min_m=.05, path_max_m=None, image_side_hint=None, image_match=None,
                recovery_side_hint=None):
        """Build a fresh path; lane_width>0 permits a single-line cold start.

        lane_width is a configured normal distance, NOT a measured width. A real
        pair replaces it after two confirmations. A lost identity cannot silently
        bootstrap as the opposite side. In continuous mode, three consistent
        observations can reacquire the locked side using configured width.
        """
        if (not np.isfinite([now_s, timeout_s, lane_width, lookahead]).all()
                or timeout_s < 0 or lane_width < 0 or lane_width > 2 or lookahead <= 0):
            raise ValueError('invalid inference timeout')
        path_max_m = lookahead+.20 if path_max_m is None else path_max_m
        if len(masks) != 1:
            self.bootstrap_hint = None
        if len(masks) < 2:
            self.pair_hint = None
            self.pair_transition = None
        if (not np.isfinite([path_min_m, path_max_m]).all() or
                not .05 <= path_min_m < path_max_m <= max_forward_m or
                not path_min_m <= lookahead <= path_max_m):
            raise ValueError('invalid metric path interval')
        if self.confirmed_at is not None and (now_s < self.confirmed_at or
                (timeout_s > 0 and now_s-self.confirmed_at > timeout_s)):
            self.confirmed = None
            self.confirmed_at = None
            self.previous = None
            self.streak = 0
            self.observed_curve = None
            self.observed_side = None
            self.observed_at = None
            self.requires_pair = True
            self.preferred_observation_side = None
        if (len(masks) >= 2 and self.observed_side in ('left','right') and
                self.observed_curve is not None and self.observed_at is not None and
                0 <= now_s-self.observed_at <= self.tracking_gap_s):
            # An isolated upper stripe must not replace a CURRENT near lock.
            # Keep connected far parts of that lock intact for turn prediction.
            projected = [floor_curves([mask],calibration,max_forward_m,
                          path_min_m,path_max_m,degree) for mask in masks]
            anchors=[]
            for index,curves in enumerate(projected):
                if len(curves)!=1:
                    continue
                near=supported_chain(curves[0],path_min_m,path_max_m)
                if (len(near)>=5 and near[0,0]<=lookahead and
                        (image_match==(index,self.observed_side) or
                         self._same_curve(curves[0],self.observed_curve))):
                    anchors.append(index)
            if len(anchors)==1:
                anchor=anchors[0]
                connected=cv2.dilate((masks[anchor]>0).astype(np.uint8),np.ones((3,3),np.uint8))
                context=[]
                for index,curves in enumerate(projected):
                    if index==anchor or len(curves)!=1:
                        continue
                    rows=np.nonzero(masks[index])[0]
                    near=supported_chain(curves[0],path_min_m,path_max_m)
                    if (len(rows) and rows.max()<.65*masks[index].shape[0] and
                            len(near)>=5 and np.min(near[:,0])>lookahead+.06 and
                            not np.any(connected & (masks[index]>0))):
                        context.append(index)
                if context:
                    retained=[i for i in range(len(masks)) if i not in context]
                    mapped=(retained.index(image_match[0]),image_match[1]) if (
                        image_match is not None and image_match[0] in retained) else None
                    try:
                        result=self.update([masks[i] for i in retained],calibration,now_s,
                            lookahead,timeout_s,lane_width,degree,max_forward_m,path_min_m,path_max_m,
                            mapped[1] if mapped and len(retained)==1 else image_side_hint,
                            mapped,recovery_side_hint)
                    finally:
                        self.context_only_indices=context
                    if result.get('inferred'):
                        selected=result.get('source_mask_index',0 if len(retained)==1 else -1)
                        if selected>=0:
                            result['source_mask_index']=retained[selected]
                    result['context_only_indices']=context
                    return result
        if len(masks) >= 2:
            try:
                target = metric_target(masks, calibration, lookahead, self.previous,
                                       degree, max_forward_m, path_min_m, path_max_m)
            except ValueError as pair_error:
                self.streak = 0
                # A second YOLO instance need not be the opposite boundary.
                # Continue only a UNIQUE geometric match to the locked boundary;
                # never choose an arbitrary mask merely to keep driving.
                if self.observed_side and self.observed_curve is not None:
                    candidates = []
                    for index, mask in enumerate(masks):
                        curves = floor_curves([mask], calibration, max_forward_m,
                                              path_min_m, path_max_m, degree)
                        flow_match = image_match == (index, self.observed_side)
                        if len(curves) == 1 and (flow_match or
                                self._same_curve(curves[0], self.observed_curve)):
                            candidates.append((index, mask, flow_match))
                    if len(candidates) == 1:
                        index, mask, flow_match = candidates[0]
                        target = self.update([mask], calibration, now_s, lookahead,
                                           timeout_s, lane_width, degree, max_forward_m,
                                           path_min_m, path_max_m,
                                           self.observed_side if flow_match else None)
                        target['source_mask_index'] = index
                        return target
                if (self.confirmed is None and self.observed_side is None and lane_width > 0
                        and ('no unambiguous left/right boundary pair' in str(pair_error)
                             or 'no common observed lane interval' in str(pair_error))):
                    candidate = unique_near_boundary(
                        masks, calibration, max_forward_m, path_min_m, path_max_m, degree)
                    if candidate is not None:
                        index, side, curve = candidate
                        old = self.pair_hint
                        count = 1
                        if (old is not None and old[1] == side and 0 < now_s-old[0] <= self.tracking_gap_s
                                and self._same_curve(curve, old[2])):
                            count = old[3]+1
                        self.pair_hint = (now_s, side, curve.copy(), count)
                        if count < 3:
                            raise ValueError(f'confirming unique near boundary {side}: {count}/3')
                        target = self.update([masks[index]], calibration, now_s, lookahead,
                                             timeout_s, lane_width, degree, max_forward_m,
                                             path_min_m, path_max_m)
                        target['source_mask_index'] = index
                        target['pair_fallback'] = 'temporally confirmed near boundary'
                        return target
                self.pair_hint = None
                self.recovery = None
                raise
            # A valid pair's width is checked separately. A narrow/outlying
            # pair must stop and remeasure, never turn into a one-line fallback.
            self.pair_hint = None
            saved_width = self.width_estimator.observe(
                target['normal_width_m'], now_s)
            if (saved_width is not None and
                    abs(target['normal_width_m']-saved_width) >
                    self.width_estimator.tolerance_m):
                raise ValueError('lane width changed; remeasuring pair')
            trusted_width = saved_width if saved_width is not None else target['normal_width_m']
            # A newly visible stripe must not instantly replace a locked
            # single boundary or switch the corner observer from RIGHT to LEFT.
            # Validate the pair's width first: narrow pairs cannot bypass STOP.
            if (self.previous is not None and self.previous.get('boundary_count') == 1
                    and self.observed_side in ('left', 'right')):
                side = self.observed_side
                candidates = []
                if (self.observed_at is not None and
                        0 <= now_s-self.observed_at <= self.tracking_gap_s):
                    for index, mask in enumerate(masks):
                        curves = floor_curves([mask], calibration, max_forward_m,
                                              path_min_m, path_max_m, degree)
                        flow = image_match == (index, side)
                        if len(curves) == 1 and (flow or self._same_curve(curves[0], self.observed_curve)):
                            candidates.append((index, mask, flow))
                if len(candidates) != 1:
                    raise ValueError('new pair lacks unique continuing boundary')
                previous_transition = self.pair_transition
                self.pair_transition = None
                index, mask, flow = candidates[0]
                # Build CURRENT geometry of the same boundary, never replay an
                # old target while waiting for a new opposite stripe.
                continuation = self.update([mask], calibration, now_s, lookahead,
                    timeout_s, lane_width, degree, max_forward_m, path_min_m,
                    path_max_m, side if flow else None)
                compatible = (self._same_curve(np.asarray(target[side+'_curve']), self.observed_curve)
                    and self._same_curve(np.asarray(target['center_path']),
                                         np.asarray(continuation['center_path'])))
                count = 0
                if compatible:
                    count = 1
                    other = 'right' if side == 'left' else 'left'
                    if (previous_transition is not None and previous_transition['side'] == side
                            and 0 < now_s-previous_transition['time'] <= self.tracking_gap_s
                            and self._same_curve(np.asarray(target[other+'_curve']),
                                                 previous_transition['other'])):
                        count = previous_transition['count']+1
                    self.pair_transition = dict(side=side,time=now_s,count=count,
                                               other=np.asarray(target[other+'_curve']).copy())
                if count < 3:
                    continuation.update(source_mask_index=index,
                        pair_transition_count=count,
                        pair_fallback='retain current boundary while confirming opposite stripe')
                    return continuation
                self.preferred_observation_side = side
                self.pair_transition = None
                target['pair_transition_count'] = count
                self.streak = max(self.streak, 1)  # Three observed pairs already confirmed.
            else:
                self.pair_transition = None
            side = self.preferred_observation_side or self.observed_side or 'left'
            other = 'right' if side == 'left' else 'left'
            self.last_observation = dict(curve=np.asarray(target[side+'_curve']), side=side,
                                         other=np.asarray(target[other+'_curve']),
                                         width=trusted_width, width_source='measured')
            self.streak += 1
            self.previous = target
            if self.streak >= 2:
                self.confirmed = dict(target, normal_width_m=trusted_width,
                                      width_m=trusted_width)
                self.confirmed_at = now_s
                self.observed_curve = None
                self.observed_side = None
                self.observed_at = None
                self.requires_pair = False
                self.recovery = None
            return target
        self.streak = 0
        curves = None  # Reuse projection only within this one update/frame.
        bootstrapped_here = False
        if not masks:
            self.recovery = None
            self.bootstrap_hint = None
            # The node controls brief hold/deceleration. Do not renew tracking
            # timestamps, but allow the SAME boundary to return before gap expiry.
            raise ValueError('no boundaries detected')
        if len(masks) == 1 and self.confirmed is None:
            curves = floor_curves(masks, calibration, max_forward_m,
                                  path_min_m, path_max_m, degree)
            self.initial_side = None
            if len(curves) == 1:
                # A curved boundary may cross y=0 without changing identity.
                # Compare to the last observed geometry BEFORE cold-start sign.
                if (self.requires_pair and self.observed_side and
                        (image_side_hint == self.observed_side or
                         (self.observed_curve is not None and
                          self._same_curve(curves[0], self.observed_curve)))):
                    self.initial_side = self.observed_side
                elif self.requires_pair and recovery_side_hint in ('left', 'right'):
                    # Only after history/flow matching failed. Never drive on
                    # one pixel-side guess; three fresh geometric confirmations
                    # below are required before accepting the new role.
                    self.initial_side = recovery_side_hint
                elif not self.requires_pair:
                    try:
                        self.initial_side = initial_boundary_side(curves[0], lookahead)
                        self.bootstrap_hint = None
                    except ValueError as exc:
                        if 'overlaps robot reference' not in str(exc):
                            self.bootstrap_hint = None
                            raise
                        # Pixel side is weak evidence, not an instantaneous
                        # replacement for metric identity. Require three fresh,
                        # geometrically continuous observations before bootstrap.
                        hint = near_pixel_side(masks[0])
                        if hint is None:
                            self.bootstrap_hint = None
                            raise
                        old = self.bootstrap_hint
                        count = 1
                        if (old is not None and old[1] == hint and
                                0 < now_s-old[0] <= self.tracking_gap_s and
                                self._same_curve(curves[0], old[2])):
                            count = old[3]+1
                        self.bootstrap_hint = (now_s, hint, curves[0].copy(), count)
                        if count < 3:
                            raise ValueError(f'initial boundary overlaps robot reference; '
                                             f'confirming near-pixel {hint}: {count}/3')
                        self.initial_side = hint
            else:
                self.bootstrap_hint = None
            if self.requires_pair and lane_width > 0 and timeout_s == 0:
                # Stay stopped during confirmation. An opposite-side detection,
                # a long gap or inconsistent geometry restarts confirmation.
                if self.initial_side is None:
                    self.recovery = None
                    raise ValueError('reacquisition requires the locked boundary side')
                curve = curves[0]
                previous = self.recovery
                count = 1
                if (previous is not None and self.recovery_side == self.initial_side
                        and 0 < now_s-previous[0] <= self.tracking_gap_s
                        and self._same_curve(curve, previous[1])):
                    count = previous[2]+1
                self.recovery = (now_s, curve.copy(), count)
                self.recovery_side = self.initial_side
                if count < 3:
                    raise ValueError(f'reacquiring {self.initial_side} boundary: {count}/3')
                self.requires_pair = False
                self.previous = None  # Old target belongs to the pre-stop frame.
                # Keep the identity confirmation until geometry succeeds. A
                # folded centre path must not restart the 1/3 identity loop.
            if self.requires_pair or lane_width <= 0 or self.initial_side is None:
                raise ValueError(f'initial boundary={self.initial_side}; lane width requires real pair')
            near = supported_chain(curves[0], path_min_m, path_max_m)
            actual = fit_boundary(near, degree)
            self.last_observation = dict(curve=near.copy(), side=self.initial_side,
                                         width=lane_width, width_source='configured')
            direction = -1 if self.initial_side == 'left' else 1
            try:
                actual, _, _ = fit_drivable_boundary(near, direction*lane_width*.5, degree, actual)
            except OffsetCurveError:
                # Retry ONLY a failed short-hook bootstrap. Successful normal
                # paths and existing tracking are never replaced by this rule.
                alternative = floor_curves(
                    masks, calibration, max_forward_m, path_min_m, path_max_m,
                    degree, recover_short_hook=True)
                if (len(alternative) != 1 or
                        initial_boundary_side(alternative[0], lookahead) != self.initial_side):
                    raise
                alternate_near = supported_chain(alternative[0], path_min_m, path_max_m)
                alternate_actual = fit_boundary(alternate_near, degree)
                # Keep last_observation on the original if recovery also folds,
                # so the independent corner policy still sees the true corner.
                center_path_from_boundary(alternate_actual, direction*lane_width*.5)
                curves, near, actual = alternative, alternate_near, alternate_actual
                self.last_observation = dict(curve=near.copy(), side=self.initial_side,
                                             width=lane_width, width_source='configured')
            virtual, _ = optional_virtual_boundary(actual, direction*lane_width)
            self.confirmed = dict(normal_width_m=lane_width, width_m=lane_width,
                                  x_m=lookahead, width_source='configured',
                                  **{self.initial_side+'_curve': actual.tolist()})
            self.confirmed_at = now_s
            self.observed_side = self.initial_side
            self.observed_curve = curves[0]
            self.observed_at = now_s
            bootstrapped_here = True
        if len(masks) != 1 or self.confirmed is None:
            raise ValueError('single line requires recent confirmed pair')
        if curves is None:
            curves = floor_curves(masks, calibration, max_forward_m,
                                  path_min_m, path_max_m, degree)
        if len(curves) != 1:
            raise ValueError('no reliable visible boundary')
        if self.observed_curve is not None and not self._same_curve(curves[0], self.observed_curve):
            # Bootstrap may have rejected a tiny endpoint hook and selected
            # the long stripe. Keep using it when the SAME long observation
            # matches history, instead of reverting to the hook next frame.
            alternatives = floor_curves(masks, calibration, max_forward_m,
                                        path_min_m, path_max_m, degree, recover_short_hook=True)
            if len(alternatives) == 1 and self._same_curve(alternatives[0], self.observed_curve):
                curves = alternatives
        curve = curves[0]
        matches = []
        reference_time = self.observed_at if self.observed_at is not None else self.confirmed_at
        if reference_time is None or not 0 <= now_s-reference_time <= self.tracking_gap_s:
            # Image flow can preserve a role across vehicle rotation, but a
            # stale ground curve cannot be driven. Stop now and require the
            # ordinary three-frame recovery on the next observations.
            if (image_side_hint in ('left', 'right') and
                    image_side_hint+'_curve' in self.confirmed):
                self.observed_side = image_side_hint
                self.observed_curve = curve.copy()
                self.observed_at = now_s
            self.confirmed = None
            self.confirmed_at = None
            self.requires_pair = True
            self.recovery = None
            raise ValueError('reference tracking gap; stop and reacquire boundary')
        sides = (self.observed_side,) if self.observed_side else ('left', 'right')
        for side in sides:
            old = (self.observed_curve if self.observed_side else
                   np.array(self.confirmed[side+'_curve']))
            # This exact observation was just classified and validated above;
            # it need not prove 4 cm of interior overlap with ITSELF. Across
            # different frames the ordinary overlap/tangent matcher remains.
            error = (0. if bootstrapped_here and side == self.initial_side else
                     curve_match_error(curve, old, .15, max(.45, lookahead+.15)))
            if not np.isfinite(error):
                error = short_curve_match_error(curve, old)
            if np.isfinite(error):
                matches.append((error, side))
        matches.sort()
        if (matches and matches[0][0] <= .035 and
                (len(matches) == 1 or matches[1][0]-matches[0][0] >= .02)):
            side = matches[0][1]
        elif (image_side_hint in ('left', 'right') and
              (self.observed_side is None or image_side_hint == self.observed_side) and
              image_side_hint+'_curve' in self.confirmed):
            side = image_side_hint
        else:
            raise ValueError('visible boundary identity ambiguous')
        # Identity and driveability are different observations. A freshly
        # matched boundary remains visible even when its normal-offset path
        # folds. Keep tracking it for recovery, but never renew the driving
        # target or confirmed width here. Failed geometry still raises below.
        self.observed_curve = curve.copy()
        self.observed_side = side
        self.observed_at = now_s
        width = self.confirmed['normal_width_m']
        if not .04 <= width <= 1.5:
            raise ValueError('normal lane width not reliable')
        # Smooth only the near observed span. Derivatives of raw segmentation
        # vertices amplify pixel noise and can fold an otherwise valid offset.
        near = supported_chain(curve, path_min_m, path_max_m)
        if len(near)<5 or arc_stations(near)[-1] < .04:
            raise ValueError('insufficient smooth curve support')
        real = fit_boundary(near, degree)
        self.last_observation = dict(curve=near.copy(), side=side, width=width,
                                     width_source=self.confirmed.get('width_source', 'measured'))
        direction = -1 if side == 'left' else 1
        real, centres, center_warning = fit_drivable_boundary(near, direction*width*.5, degree, real)
        virtual, virtual_warning = optional_virtual_boundary(real, direction*width)
        point, adaptive = select_lookahead(centres, lookahead)
        x, y = map(float, point)
        if self.previous and abs(y-self.previous['y_m']) > max(.05, width*.25):
            raise ValueError('inferred target discontinuity')
        target = dict(x_m=x, y_m=y, width_m=self.confirmed['width_m'],
                      normal_width_m=width, boundary_count=1, inferred=True,
                      visible_side=side, inference_age_s=now_s-self.confirmed_at,
                      width_source=self.confirmed.get('width_source', 'measured'),
                      center_path=centres.tolist(),
                      center_path_warning=center_warning,
                      actual_curve=np.asarray(real).tolist(),
                      virtual_curve=[] if virtual is None else virtual.tolist(),
                      virtual_boundary_warning=virtual_warning,
                      adaptive=adaptive)
        target[side+'_curve'] = real.tolist()
        # Only valid virtual geometry may help associate the returning pair.
        # An omitted/folded boundary must never enter matching references.
        if virtual is not None:
            target[('right' if side == 'left' else 'left')+'_curve'] = virtual.tolist()
        self.previous = target
        self.recovery = None
        self.observed_curve = curve.copy()
        self.observed_side = side
        self.observed_at = now_s
        self.preferred_observation_side = side
        return target


def pursuit(target, speed, max_angular=.15):
    """Pure Pursuit: alpha=atan2(y,x), kappa=2*sin(alpha)/hypot(x,y).

    Algebraically kappa=2*y/(x*x+y*y), then angular speed = v*kappa.
    Saturation is explicit; this controller drives forward, not an in-place turn.
    """
    x, y = target['x_m'], target['y_m']
    if not np.isfinite([x, y, speed, max_angular]).all() or x <= 0 or speed < 0 or max_angular <= 0:
        raise ValueError('invalid pursuit target')
    return float(np.clip(2*speed*y/(x*x+y*y), -max_angular, max_angular))


def loss_speed_scale(age, hold_seconds, stop_seconds):
    """Briefly retain the last target, then linearly decelerate to zero.

    Evaluated on EVERY control tick, not only on inference frames. Callers must
    still stop immediately for camera loss, stale inference or ambiguous geometry.
    """
    if not np.isfinite([age, hold_seconds, stop_seconds]).all() or not 0 <= hold_seconds < stop_seconds:
        raise ValueError('invalid loss timing')
    if age < 0 or age >= stop_seconds:
        return 0.
    if age <= hold_seconds:
        return 1.
    return float((stop_seconds-age)/(stop_seconds-hold_seconds))
