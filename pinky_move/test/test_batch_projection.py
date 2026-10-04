import numpy as np
import pytest
from pinky_move.robot_projection import robot_floor_point, robot_floor_points
from test_metric_lane import calibration


def test_batch_matches_scalar_including_invalid_rays():
    rng = np.random.default_rng(17)
    pixels = np.vstack((rng.uniform([0, 0], [640, 480], (2000, 2)),
                        [[-1, 200], [640, 300], [200, 480], [np.nan, 300]]))
    c = calibration()
    for maximum in (.2, 2.):
        expected = np.full((len(pixels), 2), np.nan)
        for i, (u, v) in enumerate(pixels):
            try:
                expected[i] = robot_floor_point(u, v, c, (640, 480), maximum)
            except ValueError:
                pass
        actual = robot_floor_points(pixels, c, (640, 480), maximum)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)


def test_batch_keeps_shape_empty_and_rejects_resolution_mismatch():
    c = calibration()
    assert robot_floor_points(np.empty((0, 2)), c, (640, 480)).shape == (0, 2)
    with pytest.raises(ValueError, match='resolution mismatch'):
        robot_floor_points([[320, 400]], c, (320, 240))
    with pytest.raises(ValueError, match='distance'):
        robot_floor_points([[320, 400]], c, (640, 480), -1)
