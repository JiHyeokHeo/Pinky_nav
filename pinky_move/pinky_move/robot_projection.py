"""Intrinsics + extrinsics -> floor points in the specified robot frame.

Optical axes here describe the already-upright, 180-degree-rotated stream.
The mapping to the mechanical camera link is a mounting assumption, NOT an
optical transform supplied by this URDF; physical verification is required.
"""
import argparse
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def origin_matrix(xyz, rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    t = np.eye(4)
    t[:3, :3] = rz @ ry @ rx
    t[:3, 3] = xyz
    return t


def fixed_transform(xml, base='base_footprint', tip='front_camera_link'):
    joints = {j.find('child').get('link'): j for j in ET.fromstring(xml).findall('joint')}
    chain, visited = [], set()
    while tip != base:
        if tip in visited or tip not in joints:
            raise ValueError('Missing or cyclic URDF chain')
        visited.add(tip)
        j = joints[tip]
        if j.get('type') != 'fixed':
            raise ValueError('Only fixed camera chains are supported')
        origin = j.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0]*3
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0]*3
        chain.append(origin_matrix(xyz, rpy))
        tip = j.find('parent').get('link')
    result = np.eye(4)
    for t in reversed(chain):
        result = result @ t
    return result


def validate_calibration(calibration):
    """Reject malformed geometry before publishing any motion command.

    ``base_from_upright_optical`` maps an already-oriented optical image frame
    into ``frame_id``. In base_link the floor is usually BELOW z=0; it must be
    supplied explicitly. Intrinsics and distortion must match the input image.
    """
    k = np.asarray(calibration['camera_matrix'], float)
    d = np.asarray(calibration['distortion_coefficients'], float)
    t = np.asarray(calibration['base_from_upright_optical'], float)
    size = np.asarray(calibration['image_size'], float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or d.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(d).all() or size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size <= 0)):
        raise ValueError('Invalid camera intrinsics or image size')
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., atol=1e-5)):
        raise ValueError('Invalid rigid optical transform')
    frame = calibration.get('frame_id', 'base_footprint')  # legacy diagnostic files
    if frame not in ('base_link', 'base_footprint'):
        raise ValueError('Unsupported robot frame')
    if frame == 'base_link' and 'ground_plane_z_m' not in calibration:
        raise ValueError('base_link calibration needs explicit ground_plane_z_m')
    ground = float(calibration.get('ground_plane_z_m', 0.))
    if not np.isfinite(ground) or t[2, 3] <= ground:
        raise ValueError('Camera must be above the specified floor')


def robot_floor_point(u, v, calibration, image_size, max_forward_m=2.0):
    """Undistort a pixel, rotate its ray, and intersect the ray with the floor.

    No BEV raster or guessed pixel scale is used. Output is (forward, left) in
    metres. The distance cutoff is a numerical bound, NOT accuracy validation.
    """
    if list(image_size) != calibration['image_size']:
        raise ValueError('Calibration resolution mismatch')
    if not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        raise ValueError('Pixel outside image')
    k = np.asarray(calibration['camera_matrix'], float)
    d = np.asarray(calibration['distortion_coefficients'], float)
    t = np.asarray(calibration['base_from_upright_optical'], float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Invalid optical transform')
    ray_xy = cv2.undistortPoints(np.array([[[u, v]]], float), k, d).reshape(2)
    ray = t[:3, :3] @ np.r_[ray_xy, 1.]
    origin = t[:3, 3]
    ground = float(calibration.get('ground_plane_z_m', 0.))
    if not np.isfinite(max_forward_m) or max_forward_m <= 0:
        raise ValueError('Invalid projection distance limit')
    if origin[2] <= ground or ray[2] >= -1e-6:
        raise ValueError('No visible floor intersection')
    point = origin + (ground-origin[2])/ray[2] * ray
    # Numerical/display guard only, not a physically calibrated range.
    if not np.isfinite(point).all() or not 0 < point[0] <= max_forward_m:
        raise ValueError('Outside diagnostic display range')
    return point[:2]


def robot_floor_pixel(x_m, y_m, calibration, image_size):
    """Inverse projection, using the SAME robot frame and floor height."""
    if list(image_size) != calibration['image_size']:
        raise ValueError('Calibration resolution mismatch')
    if not np.isfinite([x_m, y_m]).all() or x_m <= 0:
        raise ValueError('Invalid floor target')
    t = np.asarray(calibration['base_from_upright_optical'], float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Invalid optical transform')
    r = t[:3, :3].T
    translation = -r @ t[:3, 3]
    point = np.array([x_m, y_m, float(calibration.get('ground_plane_z_m', 0.))])
    if (r @ point + translation)[2] <= 1e-6:
        raise ValueError('Target behind camera')
    rv, _ = cv2.Rodrigues(r)
    pixels, _ = cv2.projectPoints(point.reshape(1, 3), rv, translation,
                                 np.asarray(calibration['camera_matrix'], float),
                                 np.asarray(calibration['distortion_coefficients'], float))
    u, v = pixels.reshape(2)
    if not np.isfinite([u, v]).all() or not (0 <= u < image_size[0] and 0 <= v < image_size[1]):
        raise ValueError('Target outside image')
    return float(u), float(v)


def draw_metric_target(frame, target, calibration):
    """Draw only a current, visible target; do not clamp offscreen points."""
    if target is None:
        return False
    height, width = frame.shape[:2]
    inferred = target.get('inferred', False)
    colour = (0, 165, 255) if inferred else (255, 0, 255)
    if target.get('held'):
        colour = (0, 255, 255)
    centre_path = target.get('center_path', [])
    for p, q in zip(centre_path, centre_path[1:]):
        try:
            a = robot_floor_pixel(*p, calibration, (width, height))
            b = robot_floor_pixel(*q, calibration, (width, height))
        except ValueError:
            continue
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), colour, 1, cv2.LINE_AA)
    if inferred:
        for key, dashed, line_colour in [('actual_curve', False, (255, 180, 0)),
                                         ('virtual_curve', True, (0, 165, 255))]:
            points = target.get(key, [])
            for i in range(len(points)-1):
                if dashed and i % 4 >= 2:
                    continue
                try:
                    p = robot_floor_pixel(*points[i], calibration, (width, height))
                    q = robot_floor_pixel(*points[i+1], calibration, (width, height))
                except ValueError:
                    continue
                cv2.line(frame, tuple(map(int, p)), tuple(map(int, q)), line_colour, 2, cv2.LINE_AA)
        cv2.putText(frame, f"INFERRED: {target.get('visible_side', '?')} visible, age={target.get('inference_age_s', 0):.1f}s",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
    try:
        u, v = robot_floor_pixel(target['x_m'], target['y_m'], calibration, (width, height))
    except ValueError:
        return False
    centre = (int(u), int(v))
    cv2.circle(frame, centre, 9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(frame, centre, 6, colour, -1, cv2.LINE_AA)
    cv2.drawMarker(frame, centre, (255, 255, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    label = ('Held ' if target.get('held') else 'Inferred ' if inferred else '') + f"Target X={target['x_m']*100:.0f} Y={target['y_m']*100:+.1f}cm"
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
    anchor = (max(2, min(int(u)+14, width-text_width-3)), max(16, min(int(v)-14, height-5)))
    cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True, help='Expanded URDF, not xacro')
    parser.add_argument('--intrinsics', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-frame', choices=['base_link', 'base_footprint'], default='base_link')
    args = parser.parse_args()
    with open(args.intrinsics) as f:
        intrinsics = json.load(f)
    if not intrinsics.get('validated') or intrinsics.get('rotation_degrees') != 180:
        raise ValueError('Validated upright-stream intrinsics required')
    with open(args.urdf) as f:
        xml = f.read()
    mechanical = fixed_transform(xml, base=args.base_frame)
    footprint_to_base = fixed_transform(xml, base='base_footprint', tip=args.base_frame)
    if not np.allclose(footprint_to_base[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError('Tilted base frame requires a general plane model')
    ground_z = -float(footprint_to_base[2, 3])
    optical = np.eye(4)
    optical[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    data = dict(method='intrinsics_urdf_floor', status='mount_axes_require_validation',
                image_size=intrinsics['image_size'], rotation_degrees=180,
                frame_id=args.base_frame, ground_plane_z_m=ground_z,
                camera_matrix=intrinsics['camera_matrix'],
                distortion_coefficients=intrinsics['distortion_coefficients'],
                base_from_upright_optical=(mechanical @ optical).tolist(),
                base_from_camera_link=mechanical.tolist(),
                intrinsics_source=args.intrinsics, expanded_urdf_source=args.urdf,
                measured_lens_height_cm=6., robot_frame_validated=False,
                steering_supported=False,
                assumptions=['camera link origin represents optical centre',
                             'upright optical z follows mechanical x',
                             'upright optical x follows mechanical -y',
                             'upright optical y follows mechanical -z',
                             'flat floor; height expressed in frame_id',
                             'no second 180-degree rotation after stream intrinsics'])
    with open(args.output, 'x') as f:
        json.dump(data, f, indent=2); f.write('\n')
    print('URDF camera xyz (cm):', (mechanical[:3, 3]*100).tolist())


if __name__ == '__main__':
    main()
