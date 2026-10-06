"""Explicit simulation-only white-paint input, derived from camera pixels."""
import cv2
import numpy as np
from .robot_projection import robot_floor_points
from .metric_lane import resample_chain


def miter_center(curve, distance):
    """Simplify pixel jitter, then offset a connected line along its normals.

    At a sharp vertex intersect the two offset legs (miter join). Unlike a
    pointwise noisy normal this preserves the corner's actual centre vertex.
    The curve stays in observation order; it is never sorted by forward x.
    """
    p = cv2.approxPolyDP(np.asarray(curve, np.float32).reshape(-1, 1, 2),
                         .008, False).reshape(-1, 2).astype(float)
    if len(p) < 2:
        raise ValueError('white boundary too short')
    d = np.diff(p, axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-8)
    n = np.column_stack((-t[:, 1], t[:, 0]))
    shift = np.vstack((n[0], (n[:-1]+n[1:])/2, n[-1]))
    if len(p) > 2:
        divisor = np.sum(shift[1:-1]*n[:-1], axis=1)
        if np.any(divisor < .25):
            raise ValueError('white boundary has a reversing cusp')
        shift[1:-1] /= divisor[:, None]
    return resample_chain(p+distance*shift, 100)


def white_center_path(candidates, lane_width):
    """Use the nearest ground-supported stripe on each side of this lane.

    All inputs come from camera rays, not the simulator's course/ground truth.
    Far three-stripe-road edges cannot become the active lane's boundary.
    """
    chosen = {}
    for binary, short, chains in candidates:
        curve = chains[0] if chains else short
        near = curve[:max(3, len(curve)//12)]
        lateral = float(np.median(near[:, 1]))
        if abs(lateral) < .025 or np.min(near[:, 0]) > .32:
            continue
        side = 'left' if lateral > 0 else 'right'
        try:
            center = miter_center(curve, -lane_width/2 if side == 'left' else lane_width/2)
        except ValueError:
            continue
        score = float(np.linalg.norm(center[0]))
        if side not in chosen or score < chosen[side][0]:
            chosen[side] = (score, binary, curve, center)
    if not chosen:
        raise ValueError('no near white boundary')
    # Each boundary is offset by its local tangent/normal. Prefer the more
    # completely observed centre; pairing by x would destroy a lateral leg.
    side, item = min(chosen.items(), key=lambda row: (row[1][0], -len(row[1][2])))
    return item[3], chosen, side


def white_target(path, lookahead, count, side):
    path = np.asarray(path, float)
    # A corner's offset can start behind the axle. Trim by closest path
    # station, NOT x sorting; a lateral exit remains a valid turn target.
    start = int(np.argmin(np.linalg.norm(path, axis=1)))
    path = path[start:]
    if len(path) < 2:
        raise ValueError('white observed path exhausted')
    stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    # Advance along the connected observed stripe, including a lateral leg.
    # The robot-to-prefix gap is intentional: reducing this preview too far
    # made the three-stripe S test switch to the returning road (v6 failure).
    distance = min(lookahead, stations[-1])
    point = np.array([np.interp(distance, stations, path[:, k]) for k in (0, 1)])
    adaptive = stations[-1] < lookahead
    return dict(x_m=float(point[0]), y_m=float(point[1]), path=path,
                boundary_count=count, inferred=count == 1, visible_side=side,
                adaptive=adaptive, width_m=None, center_path=path, white_path=True)


def white_floor_masks(frame, calibration):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    binary = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 175)).astype(np.uint8)
    binary[:2] = binary[-2:] = 0
    binary[:, :2] = binary[:, -2:] = 0
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return []
    points = robot_floor_points(np.column_stack([xs, ys]), calibration,
                               (frame.shape[1], frame.shape[0]), 1.2)
    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= .08)
    binary[ys[~valid], xs[~valid]] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return [(labels == i).astype(np.uint8) for i in range(1, count)
            if stats[i, cv2.CC_STAT_AREA] >= 60]


def semantic_center_path(masks, calibration, lane_width):
    """YOLO/flow가 승인한 CURRENT 마스크만 연결 경로로 변환한다.

    흰 영상 전체를 재탐색하지 않는다. OpenCV 비교와 같은 골격/법선
    경로를 사용하되 semantic gate는 YoloWhiteSupplement가 유지한다.
    추론 worker에서 실행하여 odometry callback을 막지 않는다.
    """
    from .metric_lane import connected_floor_curve, floor_curves
    candidates = []
    for mask in masks:
        chains = connected_floor_curve(mask, calibration, 1.2, pixel_step=4)
        if len(chains) == 1:
            candidates.append((mask, chains[0], chains))
        elif not chains:
            # 횡방향 코너는 골격 끝점들이 같은 이미지 높이에 있어 골격의
            # traversal 방향을 정할 수 없다. OpenCV 성공 모드와 동일한
            # 행/열 투영 sampler로 CURRENT 관측만 복구한다.
            curves = floor_curves([mask], calibration, 1.2, .08, .70)
            if len(curves) == 1:
                candidates.append((mask, curves[0], []))
    return white_center_path(candidates, lane_width)
