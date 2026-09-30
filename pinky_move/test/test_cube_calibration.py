import cv2
import numpy as np
import pytest
from pinky_move.cube_calibration import object_points, fit, floor_point


def test_synthetic_pose_and_floor():
    k = np.array([[550., 0, 320], [0, 550, 240], [0, 0, 1]])
    r = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
    t = -r @ np.array([0., 0., .08])
    rv, _ = cv2.Rodrigues(r)
    pixels, _ = cv2.projectPoints(object_points(), rv, t, k, np.zeros(5))
    data = fit(pixels.reshape(-1, 2), dict(image_size=[640, 480],
               rotation_degrees=180, validated=True, camera_matrix=k,
               distortion_coefficients=np.zeros(5)))
    assert data['reprojection_rmse_px'] < 1e-4
    fixed = fit(pixels.reshape(-1, 2), dict(image_size=[640, 480],
                rotation_degrees=180, validated=True, camera_matrix=k,
                distortion_coefficients=np.zeros(5)), camera_height_cm=8)
    assert fixed['camera_position_reference_m'][2] == .08
    assert fixed['reprojection_rmse_px'] < 1e-4
    pixel, _ = cv2.projectPoints(np.array([[.02, .30, 0.]]), rv, t, k, np.zeros(5))
    assert np.allclose(floor_point(*pixel.reshape(2), data), [.02, .30], atol=1e-5)
    with pytest.raises(ValueError):
        floor_point(320, 100, data)


def test_missing_intrinsics_rejected():
    with pytest.raises(ValueError):
        fit(np.zeros((8, 2)), {})


def test_geometry_uses_black_border_not_cube_edge():
    p = object_points()
    assert np.isclose(p[1, 0]-p[0, 0], .024)
    assert np.isclose(p[4:, 0].mean()-p[:4, 0].mean(), .08)
