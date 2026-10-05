"""연결된 관측 경계의 법선/miter 오프셋. 검출기·ROS·시뮬 GT와 무관하다."""
import cv2
import numpy as np


def connected_miter_center(curve, distance, tolerance_m=.008):
    """실제 관측 순서로 이어진 선분을 offset하고 꼭짓점에서 교차시킨다.

    일반 피팅/normal offset이 실패했을 때만 사용하는 복구 도구다.
    RDP는 픽셀 잡음만 줄이며 연결되지 않은 선, 누락된 출구를 만들지 않는다.
    뒤집힌 offset 선분, 교차, 과도한 miter는 허용하지 않는다.
    """
    from .metric_lane import first_self_intersection, resample_chain, nearest_on_chain
    source = np.asarray(curve, float)
    if (source.ndim != 2 or source.shape[1] != 2 or len(source) < 5 or
            not np.isfinite(source).all() or not np.isfinite(distance) or
            not 0 < tolerance_m <= .008):
        raise ValueError('invalid connected boundary')
    if first_self_intersection(source) is not None:
        raise ValueError('connected boundary intersects itself')
    p = cv2.approxPolyDP(source.astype(np.float32).reshape(-1, 1, 2),
                         tolerance_m, False).reshape(-1, 2).astype(float)
    if len(p) < 2:
        raise ValueError('connected boundary too short')
    if np.max(np.linalg.norm(source-nearest_on_chain(source, p)[0], axis=1)) > tolerance_m+1e-6:
        raise ValueError('connected simplification exceeds observed support')
    delta = np.diff(p, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError('degenerate connected boundary')
    tangent = delta / lengths[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    shift = np.vstack((normal[0], (normal[:-1]+normal[1:])/2, normal[-1]))
    if len(p) > 2:
        divisor = np.sum(shift[1:-1]*normal[:-1], axis=1)
        if np.any(divisor < .25):
            raise ValueError('connected boundary has a reversing cusp')
        shift[1:-1] /= divisor[:, None]
    offset = p+distance*shift
    if (np.any(np.sum(np.diff(offset, axis=0)*delta, axis=1) <= 1e-12) or
            first_self_intersection(offset) is not None):
        raise ValueError('connected center offset reverses or intersects')
    return resample_chain(p, 100), resample_chain(offset, 100)
