"""Geometry and domain isolation of Gazebo course generation."""
import numpy as np
import pytest

from pinky_move.lane_simulation import COURSES, course_points, offset_polyline


@pytest.mark.parametrize('course', COURSES)
def test_course_contains_continuous_two_boundaries(course):
    points = course_points(course)
    left, right = offset_polyline(points,.10), offset_polyline(points,-.10)
    assert points.ndim == 2 and points.shape[1] == 2
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.allclose((left+right)/2,points)
    assert np.all(np.linalg.norm(left-right,axis=1)>=.20-1e-10)


def test_right_and_left_turns_are_mirrored():
    left, right = course_points('left90'), course_points('right90')
    assert np.allclose(left[:,0],right[:,0])
    assert np.allclose(left[:,1],-right[:,1])


def test_invalid_course_rejected():
    with pytest.raises(ValueError):
        course_points('hardware')
