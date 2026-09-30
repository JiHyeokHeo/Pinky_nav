"""Offline 9x6-inner-corner checkerboard intrinsic calibration from video.

Unit checker spacing: extrinsic translations are in squares, not metres.
No robot motion and no automatic deployment or driving enable.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def calibrate(objects, corners, size):
    return cv2.calibrateCamera(objects, corners, size, None, None)


def errors(objects, corners, k, d, rotations, translations):
    return [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, k, d)[0]
                - p)**2, axis=2))))
            for o, p, r, t in zip(objects, corners, rotations, translations)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-edge-blur', type=float, default=3.0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; use a new output name')
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(args.video)
    objects, corners, frames, sharpness = [], [], [], []
    obj = np.zeros((54, 3), np.float32)
    obj[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    index = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        index += 1
        if (index-1) % 5:
            continue
        if im.shape[:2] != (480, 640):
            raise ValueError('Expected 640x480 production stream')
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        found, points = cv2.findChessboardCornersSB(gray, (9, 6))
        if not found:
            continue
        # Edge rise distance in pixels, smaller means sharper.
        blur = float(cv2.estimateChessboardSharpness(gray, (9, 6), points)[0][0])
        if not np.isfinite(blur) or blur > args.max_edge_blur:
            continue
        if any(float(np.sqrt(np.mean((points-p)**2))) < 8 for p in corners):
            continue
        objects.append(obj.copy()); corners.append(points)
        frames.append(index-1); sharpness.append(blur)
    cap.release()
    if len(corners) < 15:
        raise ValueError(f'Only {len(corners)} sharp distinct views; need at least 15')
    # Interleaved held-out views; their poses are fitted but K/D remain fixed.
    test = list(range(0, len(corners), 5))
    train = [i for i in range(len(corners)) if i not in test]
    _, kt, dt, _, _ = calibrate([objects[i] for i in train], [corners[i] for i in train], (640, 480))
    held = []
    for i in test:
        ok, rv, tv = cv2.solvePnP(objects[i], corners[i], kt, dt)
        if not ok:
            raise ValueError('Held-out pose failed')
        held.extend(errors([objects[i]], [corners[i]], kt, dt, [rv], [tv]))
    rms, k, d, rv, tv = calibrate(objects, corners, (640, 480))
    per_view = errors(objects, corners, k, d, rv, tv)
    focal_change = float(np.max(np.abs(np.diag(k)[:2]/np.diag(kt)[:2]-1)))
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rv])
    spread = float(np.degrees(np.max(np.arccos(np.clip(np.abs(normals @ normals.T), 0, 1)))))
    all_points = np.concatenate(corners).reshape(-1, 2)
    span = np.ptp(all_points, axis=0) / [640, 480]
    passed = bool(rms < .6 and max(held) < 1 and focal_change < .05 and
                  spread > 20 and min(span) > .5 and
                  100 < k[0, 0] < 2000 and 100 < k[1, 1] < 2000 and
                  0 < k[0, 2] < 640 and 0 < k[1, 2] < 480)
    result = dict(method='opencv_checkerboard_video', image_size=[640, 480],
                  rotation_degrees=180, pattern_inner_corners=[9, 6],
                  camera_matrix=k.tolist(), distortion_coefficients=d.reshape(-1).tolist(),
                  validated=passed, status='image_validation_passed' if passed else 'needs_more_views',
                  physical_distance_validated=False,
                  source_video=str(Path(args.video).resolve()), frame_indices=frames,
                  retained_views=len(corners), rms_px=float(rms), per_view_rmse_px=per_view,
                  held_out_frame_indices=[frames[i] for i in test], held_out_rmse_px=held,
                  train_vs_full_focal_relative_change=focal_change,
                  max_board_normal_separation_deg=spread, image_coverage_span=span.tolist(),
                  edge_sharpness_px=sharpness, max_edge_blur_px=args.max_edge_blur,
                  notes=['Unit squares: pose translations are not metric.',
                         'Validation is image-based, not physical distance accuracy.',
                         'Same resolution, sensor mode, rotation, crop and focus required.',
                         'No autonomous driving enabled.'])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status', 'retained_views', 'rms_px',
          'held_out_rmse_px', 'train_vs_full_focal_relative_change',
          'max_board_normal_separation_deg', 'image_coverage_span', 'camera_matrix',
          'distortion_coefficients')}, indent=2))


if __name__ == '__main__':
    main()
