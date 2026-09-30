"""Approximate camera-ground forward distance; never robot lateral position."""
import numpy as np


def forward_cm(u, v, calibration, image_size):
    if list(image_size) != calibration['image_size']:
        raise ValueError('Calibration image size mismatch')
    if not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        raise ValueError('Pixel outside image')
    h = np.asarray(calibration['image_to_ground_homography'], dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError('Invalid homography')
    mapped = h @ np.array([u, v, 1.0])
    if abs(mapped[2]) < 1e-8:
        raise ValueError('Near horizon')
    distance = float(mapped[1] / mapped[2])
    low, high = calibration['recommended_forward_range_cm']
    if not np.isfinite(distance) or not low <= distance <= high:
        raise ValueError('Outside calibrated range')
    return distance
