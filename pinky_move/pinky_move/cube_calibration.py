"""Offline two-cube pose fitting. No ROS, motor commands, or old homography.

Reference frame: x right, y forward along floor, z up, metres.
Its origin is the camera ground projection ONLY if the pair is centred and
parallel to the robot. These placement assumptions require physical validation.
K/D must describe the actual rotated 640x480 stream, not the raw sensor image.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def object_points():
    # Left/right cube centres separated by 3 cm + 5 cm clear gap.
    return np.array([(x + dx, .27, .015 + dz)
                     for x in (-.04, .04)
                     for dx, dz in ((-.012, .012), (.012, .012),
                                    (.012, -.012), (-.012, -.012))], np.float64)


def detect_points(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
    if ids is None or len(ids) != 2:
        raise ValueError('Exactly two visible tags required')
    ordered = []
    # Duplicate IDs are intentional: distinguish cubes by image location.
    for quad in sorted(corners, key=lambda q: float(q[..., 0].mean())):
        points = quad.reshape(4, 2)
        top = points[np.argsort(points[:, 1])[:2]]
        bottom = points[np.argsort(points[:, 1])[2:]]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        ordered.extend((top[0], top[1], bottom[1], bottom[0]))
    return np.asarray(ordered, np.float64)


def fit(points, intrinsics, camera_height_cm=None):
    if intrinsics.get('image_size') != [640, 480] or intrinsics.get('rotation_degrees') != 180:
        raise ValueError('Intrinsics must match the rotated 640x480 production stream')
    if intrinsics.get('validated') is not True:
        raise ValueError('Measured and validated intrinsics required; guessed focal length forbidden')
    k = np.asarray(intrinsics['camera_matrix'], np.float64)
    d = np.asarray(intrinsics['distortion_coefficients'], np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or
            k[0, 0] <= 0 or k[1, 1] <= 0 or
            not np.allclose(k[2], [0, 0, 1]) or
            d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all()):
        raise ValueError('Invalid camera matrix/distortion')
    world = object_points()
    ok, rv, tv = cv2.solvePnP(world, points, k, d)
    if not ok:
        raise ValueError('Pose fit failed')
    rotation, _ = cv2.Rodrigues(rv)
    camera = -rotation.T @ tv
    unconstrained_height_cm = float(camera[2, 0] * 100)
    if camera_height_cm is not None:
        from scipy.optimize import least_squares
        if not np.isfinite(camera_height_cm) or camera_height_cm <= 0:
            raise ValueError('Camera height must be positive and finite')
        height = camera_height_cm / 100.

        def residual(parameters):
            rot, _ = cv2.Rodrigues(parameters[:3])
            centre = np.array([parameters[3], parameters[4], height])
            trans = -rot @ centre
            projected, _ = cv2.projectPoints(world, parameters[:3], trans, k, d)
            return (projected.reshape(-1, 2)-points).reshape(-1)

        result = least_squares(residual, np.r_[rv.reshape(3), camera[:2, 0]],
                               max_nfev=2000)
        if not result.success:
            raise ValueError('Height-constrained fit failed')
        rv = result.x[:3]
        rotation, _ = cv2.Rodrigues(rv)
        camera = np.array([[result.x[3]], [result.x[4]], [height]])
        tv = -rotation @ camera
    projected, _ = cv2.projectPoints(world, rv, tv, k, d)
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-points)**2, axis=1))))
    if camera[2, 0] <= 0 or np.any((rotation @ world.T + tv)[2] <= 0):
        raise ValueError('Nonphysical pose')
    return dict(status=('candidate_requires_physical_validation' if error <= 2
                        else 'rejected_reprojection_error'),
                measured_camera_height_cm=camera_height_cm,
                unconstrained_camera_height_cm=unconstrained_height_cm,
                camera_ground_origin_offset_cm=(camera[:2, 0]*100).tolist(),
                camera_matrix=k.tolist(), distortion_coefficients=d.reshape(-1).tolist(),
                reference_to_optical_rotation=rotation.tolist(),
                reference_to_optical_translation_m=tv.reshape(3).tolist(),
                camera_position_reference_m=camera.reshape(3).tolist(),
                reprojection_rmse_px=error)


def floor_point(u, v, calibration):
    """Candidate diagnostic only: return floor x-right/y-forward metres."""
    if not (0 <= u < 640 and 0 <= v < 480):
        raise ValueError('Pixel outside calibrated image')
    k = np.asarray(calibration['camera_matrix'], float)
    d = np.asarray(calibration['distortion_coefficients'], float)
    r = np.asarray(calibration['reference_to_optical_rotation'], float)
    t = np.asarray(calibration['reference_to_optical_translation_m'], float)
    uv = cv2.undistortPoints(np.array([[[u, v]]], float), k, d).reshape(2)
    origin = -r.T @ t
    ray = r.T @ np.r_[uv, 1.]
    if ray[2] >= -1e-8 or origin[2] <= 0:
        raise ValueError('Ray does not intersect visible floor')
    result = origin + (-origin[2] / ray[2]) * ray
    if not np.isfinite(result).all():
        raise ValueError('Invalid floor intersection')
    return result[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--intrinsics', help='Validated JSON K/D for rotated stream')
    parser.add_argument('--output', required=True)
    parser.add_argument('--camera-height-cm', type=float)
    args = parser.parse_args()
    frame = cv2.imread(args.image)
    if frame is None or frame.shape[:2] != (480, 640):
        parser.error('A 640x480 image is required')
    points = detect_points(frame)
    data = dict(method='two_apriltag_cubes', version=1, image_size=[640, 480],
                rotation_degrees=180, status='missing_camera_intrinsics',
                source_image=str(Path(args.image).resolve()),
                cube_edge_cm=3., tag_black_edge_cm=2.4, inner_gap_cm=5.,
                front_plane_forward_cm=27., image_points=points.tolist(),
                object_points_m=object_points().tolist(),
                assumptions=['tags centred on cube faces', 'both faces vertical and coplanar',
                             'cube pair aligned to camera ground projection'],
                robot_frame_validated=False, steering_supported=False)
    if args.intrinsics:
        data.update(fit(points, json.loads(Path(args.intrinsics).read_text()),
                        args.camera_height_cm))
    with open(args.output, 'x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    print(data['status'])


if __name__ == '__main__':
    main()
