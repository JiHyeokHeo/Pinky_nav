"""Pure geometry and state helpers for camera lane following."""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


@dataclass(frozen=True)
class LaneCenterEstimate:
    """Estimated image-space lane center at one horizontal sample band."""

    center_x: float
    lane_width: float
    boundary_count: int


def sample_mask_x(mask, y_ratio, band_ratio=0.03, minimum_pixels=6):
    """Return the median x coordinate where a binary mask crosses a band."""
    if mask.ndim != 2 or mask.size == 0:
        return None
    height = mask.shape[0]
    center_y = int(round((height - 1) * float(y_ratio)))
    half_band = max(1, int(round(height * float(band_ratio) / 2.0)))
    top = max(0, center_y - half_band)
    bottom = min(height, center_y + half_band + 1)
    _, x_coordinates = np.nonzero(mask[top:bottom])
    if x_coordinates.size < int(minimum_pixels):
        return None
    return float(np.median(x_coordinates))


def lane_candidates_at(mask_list, y_ratio, band_ratio=0.03):
    """Sample every lane instance and return sorted x coordinates."""
    candidates = []
    for mask in mask_list:
        x_coordinate = sample_mask_x(mask, y_ratio, band_ratio)
        if x_coordinate is not None:
            candidates.append(x_coordinate)
    return sorted(candidates)


def estimate_lane_center(candidates, image_width, expected_center,
                         expected_lane_width, minimum_width_ratio=0.18,
                         maximum_width_ratio=0.95):
    """Choose the most plausible lane-boundary pair or infer from one side."""
    width = float(image_width)
    if width <= 0.0 or not candidates:
        return None

    expected_center = float(expected_center)
    expected_lane_width = max(1.0, float(expected_lane_width))
    minimum_width = width * float(minimum_width_ratio)
    maximum_width = width * float(maximum_width_ratio)

    valid_pairs = []
    for left, right in combinations(sorted(candidates), 2):
        pair_width = right - left
        if not minimum_width <= pair_width <= maximum_width:
            continue
        midpoint = (left + right) / 2.0
        score = (
            abs(midpoint - expected_center) / width
            + 0.35 * abs(pair_width - expected_lane_width) / width
        )
        valid_pairs.append((score, midpoint, pair_width))

    if valid_pairs:
        _, midpoint, pair_width = min(valid_pairs, key=lambda item: item[0])
        return LaneCenterEstimate(midpoint, pair_width, 2)

    half_lane = expected_lane_width / 2.0
    hypotheses = []
    for boundary_x in candidates:
        if boundary_x < expected_center:
            inferred_center = boundary_x + half_lane
        else:
            inferred_center = boundary_x - half_lane
        score = abs(inferred_center - expected_center)
        hypotheses.append((score, inferred_center))

    _, inferred_center = min(hypotheses, key=lambda item: item[0])
    inferred_center = min(width - 1.0, max(0.0, inferred_center))
    return LaneCenterEstimate(inferred_center, expected_lane_width, 1)


def select_near_lane(mask_list, image_width, expected_center,
                     expected_lane_width, near_ratio=0.78,
                     minimum_ratio=0.66, step=0.04, band_ratio=0.04,
                     minimum_width_ratio=0.18, maximum_width_ratio=0.95):
    """Prefer the nearest valid pair, then the nearest single boundary.

    Search only the configured near band. Never count Crossline as a lane or
    move the search to the horizon just to obtain two boundaries.
    """
    if not 0.0 <= minimum_ratio <= near_ratio <= 1.0 or step <= 0.0:
        raise ValueError('Invalid near lane sampling range or step')
    ratios = [near_ratio]
    while ratios[-1] > minimum_ratio + 1e-6:
        ratios.append(max(minimum_ratio, ratios[-1] - step))
    fallback = None
    for ratio in ratios:
        candidates = lane_candidates_at(mask_list, ratio, band_ratio)
        estimate = estimate_lane_center(
            candidates, image_width, expected_center, expected_lane_width,
            minimum_width_ratio, maximum_width_ratio)
        if estimate is None:
            continue
        selection = (estimate, ratio, len(candidates))
        if estimate.boundary_count == 2:
            return selection
        if fallback is None:
            fallback = selection
    return fallback if fallback is not None else (None, near_ratio, 0)


def consistent_single_far_center(mask_list, near_center, near_ratio,
                                far_ratio, far_lane_width, band_ratio=0.04):
    """Keep a single far boundary's side anchored to the SAME near mask.

    Comparing a curved far boundary to the previous inferred far centre can
    alternate left/right classifications even for an unchanged image.
    """
    pairs = []
    for mask in mask_list:
        far_x = sample_mask_x(mask, far_ratio, band_ratio)
        if far_x is not None:
            pairs.append((sample_mask_x(mask, near_ratio, band_ratio), far_x))
    if len(pairs) != 1 or pairs[0][0] is None:
        return None
    near_x, far_x = pairs[0]
    if abs(near_x - near_center) < 1e-6:
        return None
    side = 1.0 if near_x < near_center else -1.0
    return far_x + side * far_lane_width / 2.0


def near_priority_command(near_center, far_center, image_width,
                          lateral_gain, maximum_angular_speed, deadband=0.04):
    """Steer from near lateral error; far curvature is a slowdown hint only.

    Returns (angular_z, preview_severity). All distances here are normalized
    image coordinates, not calibrated robot-frame metres.
    """
    half = max(1.0, image_width / 2.0)
    error = (near_center - image_width / 2.0) / half
    magnitude = max(0.0, abs(error) - max(0.0, deadband))
    angular = -math.copysign(magnitude * lateral_gain, error)
    limit = abs(maximum_angular_speed)
    preview = min(1.0, abs(far_center - near_center) / half)
    return max(-limit, min(limit, angular)), preview


def steering_command(near_center, far_center, image_width,
                     lateral_gain, heading_gain, maximum_angular_speed):
    """Return ROS angular.z; a target to image-right produces a right turn."""
    half_width = max(1.0, float(image_width) / 2.0)
    image_center = float(image_width) / 2.0
    lateral_error = (float(near_center) - image_center) / half_width
    heading_error = (float(far_center) - float(near_center)) / half_width
    raw_command = -(
        float(lateral_gain) * lateral_error
        + float(heading_gain) * heading_error
    )
    limit = abs(float(maximum_angular_speed))
    return min(limit, max(-limit, raw_command))


def crossline_is_close(mask, trigger_y_ratio, minimum_area_ratio=0.002):
    """Return true when a credible crossline is near the robot."""
    if mask.ndim != 2 or mask.size == 0:
        return False
    y_coordinates, _ = np.nonzero(mask)
    if y_coordinates.size == 0:
        return False
    area_ratio = y_coordinates.size / float(mask.size)
    bottom_ratio = float(y_coordinates.max()) / max(1, mask.shape[0] - 1)
    return (
        area_ratio >= float(minimum_area_ratio)
        and bottom_ratio >= float(trigger_y_ratio)
    )
