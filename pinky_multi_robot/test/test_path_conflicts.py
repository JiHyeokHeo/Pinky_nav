import unittest

from pinky_multi_robot.path_conflicts import path_distance, point_path_distance


class PathConflictsTest(unittest.TestCase):
    def test_crossing_paths(self):
        self.assertEqual(
            path_distance([(0, 0), (1, 1)], [(0, 1), (1, 0)]), 0.0)

    def test_parallel_paths(self):
        self.assertAlmostEqual(
            path_distance([(0, 0), (1, 0)], [(0, 1), (1, 1)]), 1.0)

    def test_collinear_but_disjoint(self):
        self.assertAlmostEqual(
            path_distance([(0, 0), (1, 0)], [(2, 0), (3, 0)]), 1.0)

    def test_waiting_point_on_priority_path(self):
        self.assertEqual(point_path_distance((0.5, 0), [(0, 0), (1, 0)]), 0.0)

    def test_empty_path_rejected(self):
        with self.assertRaises(ValueError):
            path_distance([], [(0, 0)])


if __name__ == '__main__':
    unittest.main()
