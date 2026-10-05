"""Geometry and domain isolation of Gazebo course generation."""
import numpy as np
import pytest

from pinky_move.lane_simulation import COURSES, course_points, offset_polyline, road_boundaries, stripe_mesh


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


def test_sharp_s_has_four_alternating_right_angle_turns():
    delta = np.diff(course_points('s_sharp'), axis=0)
    assert np.allclose(np.sum(delta[:-1]*delta[1:], axis=1), 0.)
    cross = delta[:-1, 0]*delta[1:, 1]-delta[:-1, 1]*delta[1:, 0]
    assert np.array_equal(np.sign(cross), [1., -1., -1., 1.])


@pytest.mark.parametrize('course', COURSES)
@pytest.mark.parametrize('target_lane', [0, 1])
def test_two_lanes_have_three_stripes_and_correct_target_pair(course, target_lane):
    points = course_points(course)
    boundaries = road_boundaries(points, .20, 2, target_lane)
    assert len(boundaries) == 3
    assert np.allclose((boundaries['left']+boundaries['right'])/2, points)
    assert all(np.isfinite(curve).all() for curve in boundaries.values())
    assert all(np.linalg.norm(np.diff(curve, axis=0), axis=1).min() > 0.
               for curve in boundaries.values())
    # At the straight entry, three stripes are spaced by one lane width.
    assert np.allclose(np.diff(sorted(c[0, 1] for c in boundaries.values())), .20)


@pytest.mark.parametrize('count,target', [(0, 0), (3, 0), (1, 1), (2, -1), (2, 2)])
def test_invalid_road_selection_rejected(count, target):
    with pytest.raises(ValueError):
        road_boundaries(course_points('two_lines'), .20, count, target)


@pytest.mark.parametrize('course', COURSES)
def test_paint_triangles_share_joint_vertices_without_gaps(course, tmp_path):
    import cv2
    points = course_points(course)
    for name, curve in road_boundaries(points, .20, 2).items():
        path = tmp_path/(name+'.obj')
        vertices, faces = stripe_mesh(curve, path)
        assert path.is_file()
        assert len(vertices) == 2*len(curve)
        assert len(faces) == 2*(len(curve)-1)
        for index in range(len(curve)-2):
            # Two neighbouring segment meshes share exactly their joint edge.
            a = set(faces[2*index:2*index+2].ravel())
            b = set(faces[2*index+2:2*index+4].ravel())
            assert a & b == {2*index+2, 2*index+3}
        image = np.zeros((1600, 1600), np.uint8)
        pixels = np.rint((vertices+[.6, 1.5])*400).astype(np.int32)
        for face in faces:
            cv2.fillPoly(image, [pixels[face]], 1)
        assert cv2.connectedComponents(image)[0] == 2  # background + ONE stripe


def test_right_angle_ribbon_has_exact_miter_not_square_caps(tmp_path):
    vertices, _ = stripe_mesh([[0., 0.], [1., 0.], [1., 1.]], tmp_path/'corner.obj')
    np.testing.assert_allclose(vertices[2:4], [[.99, .01], [1.01, -.01]])


@pytest.mark.parametrize('lane', [0, 1])
def test_sharp_three_stripe_road_can_actually_generate_all_meshes(lane, tmp_path):
    points = course_points('s_sharp')
    for name, stripe in road_boundaries(points, .20, 2, lane).items():
        assert np.linalg.norm(np.diff(stripe, axis=0), axis=1).min() > .020
        assert np.all(np.sum(np.diff(stripe, axis=0)*np.diff(points, axis=0), axis=1) > 0.)
        stripe_mesh(stripe, tmp_path/(name+'.obj'))


def test_too_wide_road_rejects_folded_stripe_before_launch():
    with pytest.raises(ValueError, match='too tight'):
        road_boundaries(course_points('s_sharp'), .4, 2, 1)
