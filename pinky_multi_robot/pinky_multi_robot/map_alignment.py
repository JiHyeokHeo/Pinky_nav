"""Advisory laser-endpoint versus occupancy-map alignment estimate.

This is deliberately not a navigation safety gate: dynamic objects, glass,
and repetitive corridors can make the score misleading in either direction.
"""

import math


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))


def scan_map_match(grid, scan, transform, tolerance_m=0.05, max_samples=90):
    """Return (matched, comparable) finite laser endpoints near map obstacles."""
    resolution = grid.info.resolution
    width, height = grid.info.width, grid.info.height
    if resolution <= 0 or width <= 0 or height <= 0 or not scan.ranges:
        return 0, 0

    map_yaw = _yaw(grid.info.origin.orientation)
    map_cos, map_sin = math.cos(map_yaw), math.sin(map_yaw)
    laser_yaw = _yaw(transform.transform.rotation)
    laser_cos, laser_sin = math.cos(laser_yaw), math.sin(laser_yaw)
    offset = transform.transform.translation
    origin = grid.info.origin.position
    radius = max(1, math.ceil(tolerance_m / resolution))
    step = max(1, math.ceil(len(scan.ranges) / max_samples))
    matched = comparable = 0

    for index in range(0, len(scan.ranges), step):
        distance = scan.ranges[index]
        if not math.isfinite(distance) or distance < scan.range_min or distance >= scan.range_max:
            continue
        angle = scan.angle_min + index * scan.angle_increment
        lx, ly = distance * math.cos(angle), distance * math.sin(angle)
        wx = offset.x + laser_cos * lx - laser_sin * ly
        wy = offset.y + laser_sin * lx + laser_cos * ly
        dx, dy = wx - origin.x, wy - origin.y
        gx = math.floor((map_cos * dx + map_sin * dy) / resolution)
        gy = math.floor((-map_sin * dx + map_cos * dy) / resolution)
        if not (0 <= gx < width and 0 <= gy < height):
            continue
        cell = grid.data[gy * width + gx]
        if cell < 0:
            continue
        comparable += 1
        found = False
        for y in range(max(0, gy - radius), min(height, gy + radius + 1)):
            for x in range(max(0, gx - radius), min(width, gx + radius + 1)):
                if grid.data[y * width + x] >= 65:
                    found = True
                    break
            if found:
                break
        matched += int(found)
    return matched, comparable
