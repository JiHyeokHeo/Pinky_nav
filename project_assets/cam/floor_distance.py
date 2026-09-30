"""Pixel -> approximate forward ground distance. No robot motion commands.

Usage: python3 floor_distance.py 320 350
"""
import argparse
import json
from pathlib import Path
import numpy as np


def forward_cm(u, v, calibration, image_size=(640,480)):
    if list(image_size) != calibration['image_size']:
        raise ValueError('Calibration image size mismatch')
    if not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        raise ValueError('Pixel outside image')
    h = np.asarray(calibration['image_to_ground_homography'], dtype=float)
    mapped = h @ np.array([u,v,1.0])
    if abs(mapped[2]) < 1e-8:
        raise ValueError('Pixel near ground horizon')
    distance = float(mapped[1]/mapped[2])
    low, high = calibration['recommended_forward_range_cm']
    if not np.isfinite(distance) or not low <= distance <= high:
        raise ValueError(f'{distance:.1f} cm: outside calibrated range {low}-{high} cm')
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('u', type=float)
    parser.add_argument('v', type=float)
    args = parser.parse_args()
    data = json.loads(Path(__file__).with_name('calibration.json').read_text())
    try:
        print(f'Camera-ground forward distance (approx): {forward_cm(args.u,args.v,data):.1f} cm')
    except ValueError as error:
        parser.error(str(error))
