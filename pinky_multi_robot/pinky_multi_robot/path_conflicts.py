"""Geometry used by the central two-robot path reservation check."""

import math


def remaining_path(point, path):
    """Project onto a non-looping route; retain the route ahead of that point."""
    if len(path) < 2:
        return list(path)
    candidates = []
    for index, (a, b) in enumerate(zip(path, path[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = max(0., min(1., ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / length)) if length else 0.
        projected = (a[0]+t*dx, a[1]+t*dy)
        candidates.append((math.dist(point, projected), index, projected))
    _, index, projected = min(candidates)
    return [projected] + list(path[index+1:])


def path_prefix(path, distance):
    """Include at most distance metres ahead, interpolating the last segment."""
    result = list(path[:1])
    for a, b in zip(path, path[1:]):
        length = math.dist(a, b)
        if length > distance:
            ratio = distance / length
            result.append((a[0]+ratio*(b[0]-a[0]), a[1]+ratio*(b[1]-a[1])))
            break
        result.append(b)
        distance -= length
    return result


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.dist(point, start)
    fraction = max(0.0, min(1.0, (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared))
    return math.hypot(point[0] - start[0] - fraction * dx,
                      point[1] - start[1] - fraction * dy)


def _segments_intersect(a, b, c, d):
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    ab_c, ab_d = cross(a, b, c), cross(a, b, d)
    cd_a, cd_b = cross(c, d, a), cross(c, d, b)
    if ab_c == ab_d == cd_a == cd_b == 0:
        return (max(min(a[0], b[0]), min(c[0], d[0])) <=
                min(max(a[0], b[0]), max(c[0], d[0])) and
                max(min(a[1], b[1]), min(c[1], d[1])) <=
                min(max(a[1], b[1]), max(c[1], d[1])))
    return ab_c * ab_d <= 0 and cd_a * cd_b <= 0


def segment_distance(a, b, c, d):
    """Minimum Euclidean distance between two closed 2-D segments."""
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d),
               _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b),
               _point_segment_distance(d, a, b))


def path_distance(first, second):
    """Return the closest separation of two polylines, including endpoints."""
    if not first or not second:
        raise ValueError('Cannot compare empty paths')
    first_segments = list(zip(first, first[1:])) or [(first[0], first[0])]
    second_segments = list(zip(second, second[1:])) or [(second[0], second[0])]
    return min(segment_distance(a, b, c, d)
               for a, b in first_segments for c, d in second_segments)


def point_path_distance(point, path):
    if not path:
        raise ValueError('Cannot compare an empty path')
    segments = list(zip(path, path[1:])) or [(path[0], path[0])]
    return min(_point_segment_distance(point, a, b) for a, b in segments)
