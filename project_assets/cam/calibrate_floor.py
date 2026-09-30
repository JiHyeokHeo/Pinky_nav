"""Reproduce an approximate ground-plane calibration from two measured photos.

Coordinates are manually identified solid checker cells, refined with threshold
contours (not decoded AprilTag corners). Measurements must be horizontal floor
distances to the FRONT BLACK GRID EDGE, not the white paper edge or slant range.
Run: python3 calibrate_floor.py
"""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCES = [
    ('cam_20260919_151012.jpg', 26.0,
     [[293,276],[359,277],[357,266],[298,266],
      [238,264],[295,264],[298,255],[245,255],
      [361,264],[418,263],[412,255],[358,256]],
     [[-1.75,3.5],[1.75,3.5],[1.75,7],[-1.75,7],
      [-5.25,7],[-1.75,7],[-1.75,10.5],[-5.25,10.5],
      [1.75,7],[5.25,7],[5.25,10.5],[1.75,10.5]]),
    ('cam_20260919_151047.jpg', 14.8,
     [[281,346],[389,349],[383,320],[292,319],
      [302,290],[377,295],[374,280],[307,280],
      [377,278],[443,278],[435,266],[375,267],
      [240,277],[305,277],[308,267],[249,267]],
     [[-1.75,3.5],[1.75,3.5],[1.75,7],[-1.75,7],
      [-1.75,10.5],[1.75,10.5],[1.75,14],[-1.75,14],
      [1.75,14],[5.25,14],[5.25,17.5],[1.75,17.5],
      [-5.25,14],[-1.75,14],[-1.75,17.5],[-5.25,17.5]])]


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64)[None], matrix)[0]


def main():
    samples, pixels, ground, matrices = [], [], [], []
    for name, distance, p, q in REFERENCES:
        im = cv2.imread(str(ROOT / name))
        if im is None or im.shape[:2] != (480,640):
            raise ValueError(f'Expected 640x480 reference image: {name}')
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        q[:,1] += distance
        h, _ = cv2.findHomography(p, q, 0)
        fit = project(p,h)-q
        samples.append(dict(image=name,camera_to_black_grid_front_cm=distance,
                            pixel_points=p.tolist(),ground_points_cm=q.tolist(),
                            homography=h.tolist(),fit_rmse_cm=float(np.sqrt(np.mean(np.sum(fit**2,axis=1))))))
        pixels.append(p); ground.append(q); matrices.append(h)
    cross = []
    for i in range(2):
        errors = project(pixels[1-i], matrices[i])[:,1]-ground[1-i][:,1]
        cross.append(float(np.sqrt(np.mean(errors**2))))
    p, q = np.concatenate(pixels), np.concatenate(ground)
    h, _ = cv2.findHomography(p,q,0)
    error = project(p,h)-q
    # Lateral board alignment is not measured. Only forward distance is exposed.
    data = dict(version=2,status='approximate_requires_physical_validation',
                method='two_photo_checker_ground_homography',image_size=[640,480],
                square_size_cm=3.5,image_to_ground_homography=h.tolist(),
                coordinates=dict(x='board-relative lateral right; NOT robot lateral',
                                 y='forward floor distance from camera ground projection, cm'),
                reference_frames=samples,
                validation=dict(fit_forward_rmse_cm=float(np.sqrt(np.mean(error[:,1]**2))),
                                cross_photo_forward_rmse_cm=cross,
                                independent_distance_validation=False),
                recommended_forward_range_cm=[18.3,36.5],
                robot_wheel_offset_cm=None,
                assumptions=['26.0 and 14.8 cm measured horizontally to front black grid edge',
                             'Solid checker cells are 3.5 cm with no gap between adjacent grid cells',
                             'Camera mounting, focus, image rotation and resolution remain unchanged',
                             'Board front edge approximately perpendicular to driving direction',
                             'Lens distortion is not calibrated; fit error is not physical accuracy'],
                urdf_reference='/home/tory/ws/pinky_pro/pinky_description/urdf/pinky.urdf.xacro',
                urdf_note='Nominal camera offset not applied: actual mount and distance origin unverified')
    (ROOT/'calibration.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data['validation'],indent=2))
    print('Saved',ROOT/'calibration.json')


if __name__ == '__main__':
    main()
